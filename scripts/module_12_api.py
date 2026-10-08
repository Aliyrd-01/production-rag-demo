"""
Module 12 — Production RAG API (FastAPI).

Запуск:
  python scripts/module_12_api.py
  python scripts/module_12_api.py --port 8080

Эндпоинты:
  GET  /health              — проверка
  POST /query               — RAG-запрос, ответ + источники
  GET  /query/stream/{q}    — стриминг ответа (SSE)
  POST /ingest              — добавить документ (заглушка)
"""

import os, sys, time, json, asyncio, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

try:
    import pymysql
except Exception as e:
    print(f"[mysql] pymysql import failed: {e}")
    pymysql = None

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

from common import get_chat_llm, get_fastembed_dense, get_embeddings
from module_5_retrieval import search_dense, search_sparse, search_hybrid, search_hybrid_rerank
from module_7_generation import format_context, generate_citations, check_faithfulness, CITATIONS_PROMPT

# ─── Langfuse ─────────────────────────────────────────────────────────────
_langfuse_handler = None
_langfuse_client = None
try:
    from langfuse import Langfuse, propagate_attributes
    from langfuse.langchain import CallbackHandler as _CallbackHandlerBase
except Exception as e:
    print(f"[langfuse] import failed: {e}")
    _CallbackHandlerBase = None


if _CallbackHandlerBase is not None:

    class FilledUsageCallbackHandler(_CallbackHandlerBase):
        """CallbackHandler, который подставляет примерный usage (по символам),
        если провайдер (Zen/OpenRouter) не вернул token usage в ответе."""

        def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
            if not self._has_usage(response):
                try:
                    inp_text = self._flatten_input(kwargs.get("inputs"))
                    out_text = self._flatten_output(response)
                    est_in = max(1, len(inp_text) // 4)
                    est_out = max(1, len(out_text) // 4)
                    if response.llm_output is None:
                        response.llm_output = {}
                    response.llm_output["token_usage"] = {
                        "input": est_in,
                        "output": est_out,
                        "total": est_in + est_out,
                    }
                except Exception:
                    pass
            return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        @staticmethod
        def _has_usage(response) -> bool:
            def _positive(total: int) -> bool:
                return bool(total) and total > 0

            for gen in response.generations or []:
                for g in gen:
                    gi = getattr(g, "generation_info", None) or {}
                    um = gi.get("usage_metadata")
                    m = getattr(g, "message", None)
                    mm = None
                    if m is not None:
                        mm = getattr(m, "usage_metadata", None)
                        rm = getattr(m, "response_metadata", None)
                        if isinstance(rm, dict) and _positive((rm.get("usage") or {}).get("input_tokens")):
                            return True
                    um = um or mm
                    if um is not None:
                        for k in ("total", "total_tokens", "input", "output", "input_tokens", "output_tokens", "prompt_tokens", "completion_tokens"):
                            if _positive(um.get(k)):
                                return True
            rlo = response.llm_output or {}
            for key in ("token_usage", "usage"):
                tu = rlo.get(key)
                if isinstance(tu, dict):
                    for k in ("total", "total_tokens", "input", "output", "input_tokens", "output_tokens", "prompt_tokens", "completion_tokens"):
                        if _positive(tu.get(k)):
                            return True
            return False

        @staticmethod
        def _flatten_input(inp) -> str:
            if inp is None:
                return ""
            if isinstance(inp, str):
                return inp
            if isinstance(inp, (list, tuple)):
                parts = []
                for it in inp:
                    c = getattr(it, "content", it) if not isinstance(it, (str, dict)) else it
                    if isinstance(c, (list, tuple)):
                        parts.append(" ".join(str(x) for x in c))
                    else:
                        parts.append(str(c))
                return "\n".join(parts)
            return str(inp)

        @staticmethod
        def _flatten_output(response) -> str:
            gen = response.generations[-1] if response.generations else None
            if not gen:
                return ""
            g = gen[-1]
            text = getattr(g, "text", None)
            if text:
                return text
            msg = getattr(g, "message", None)
            if msg is not None:
                return str(msg.content or "")
            return ""

else:
    FilledUsageCallbackHandler = None


try:
    _langfuse_on = os.getenv("ENABLE_LANGFUSE", "").lower() in ("1", "true", "yes")
    if (_langfuse_client is None and _langfuse_on
            and os.getenv("LANGFUSE_PUBLIC_KEY") and FilledUsageCallbackHandler is not None):
        _langfuse_client = Langfuse()
        _langfuse_handler = FilledUsageCallbackHandler()
        print(f"[langfuse] tracing to {os.getenv('LANGFUSE_HOST','https://cloud.langfuse.com')}")
except Exception as e:
    print(f"[langfuse] init failed: {e}")

LC_CONFIG = {"callbacks": [_langfuse_handler]} if _langfuse_handler else {}

app = FastAPI(title="RAG Production API", version="1.0.0")

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "")
CACHE_SIM_THRESHOLD = float(os.getenv("CACHE_SIM_THRESHOLD", "0.92"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS.split(",")],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

llm = get_chat_llm(temperature=0)

# ─── Semantic Cache ────────────────────────────────────────────────────────

_cache_entries = []
_CACHE_MAX = int(os.getenv("CACHE_MAX", "256"))
_CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "0") == "1"


def _cache_embed(text: str):
    dense = get_fastembed_dense()
    vecs = list(dense.embed([text]))
    return vecs[0]


def _cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-12)


def _cache_evict():
    now = time.time()
    _cache_entries[:] = [e for e in _cache_entries if now - e["ts"] < _CACHE_TTL]
    while len(_cache_entries) > _CACHE_MAX:
        _cache_entries.pop(0)


def _cache_lookup(question: str, filters: dict | None = None):
    if not CACHE_ENABLED:
        return None
    _cache_evict()
    if not _cache_entries:
        return None
    key = json.dumps({"q": question, "f": filters}, sort_keys=True, default=str)
    qv = _cache_embed(question)
    for entry in _cache_entries:
        if entry.get("key") == key:
            return entry["result"]
        if not filters and _cosine_sim(qv, entry["vector"]) >= CACHE_SIM_THRESHOLD:
            if not entry.get("filters"):
                return entry["result"]
    return None


def _cache_store(question: str, result: dict, filters: dict | None = None):
    if not CACHE_ENABLED:
        return
    _cache_entries.append({
        "key": json.dumps({"q": question, "f": filters}, sort_keys=True, default=str),
        "filters": bool(filters),
        "vector": _cache_embed(question),
        "ts": time.time(),
        "result": result,
    })
    _cache_evict()

# ─── MySQL Logging ────────────────────────────────────────────────────────

MYSQL_CONFIG = {
    "host": os.getenv("RAG_MYSQL_HOST", "auth-db936.hstgr.io"),
    "port": int(os.getenv("RAG_MYSQL_PORT", "3306")),
    "user": os.getenv("RAG_MYSQL_USER", "u543957720_crypto"),
    "password": os.getenv("RAG_MYSQL_PASS", ""),
    "database": os.getenv("RAG_MYSQL_DB", "u543957720_cryptoprice"),
    "charset": "utf8mb4",
}
MYSQL_LOG_ENABLED = bool(pymysql and MYSQL_CONFIG["password"])


def _sources_to_json(task, sources):
    out = []
    for r in sources:
        s = getattr(r, "source", "")
        l = getattr(r, "lang", "")
        out.append({"source": s, "lang": l})
    return json.dumps(out, ensure_ascii=False)


def _log_query_mysql(question, answer, sources_json, latency_ms, faithfulness, cache_hit, client_ip):
    if not MYSQL_LOG_ENABLED:
        return
    try:
        conn = pymysql.connect(**MYSQL_CONFIG)
        try:
            with conn.cursor() as c:
                c.execute(
                    "INSERT INTO rag_query_logs "
                    "(question, answer, sources, latency_ms, faithfulness, cache_hit, client_ip) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (question, answer, sources_json, int(latency_ms), faithfulness, int(cache_hit), client_ip),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"[mysql] log write failed: {e}")


# ─── Faithfulness Background ──────────────────────────────────────────────

_faith_results = {}


async def _run_faithfulness(task_id: str, context: str, answer: str, config=None):
    loop = asyncio.get_running_loop()
    config = config or {}

    def _faith_check():
        with propagate_attributes(trace_name="rag_query"):
            return check_faithfulness(context, answer, llm, config)

    faith = await loop.run_in_executor(None, _faith_check)
    _faith_results[task_id] = round(faith, 3)


def _check_auth(request: Request):
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        raise HTTPException(401, "Unauthorized — provide X-API-Key header")

# ─── Rate limiting ───────────────────────────────────────────────────────

from collections import defaultdict
from datetime import datetime, timezone

_rate = defaultdict(list)
RATE_LIMIT = 20
RATE_WINDOW = 60


def _check_rate(client_ip: str):
    now = time.time()
    timestamps = _rate.get(client_ip, [])
    timestamps = [t for t in timestamps if now - t < RATE_WINDOW]
    if len(timestamps) >= RATE_LIMIT:
        raise HTTPException(429, f"Rate limit: {RATE_LIMIT} req/{RATE_WINDOW}s")
    timestamps.append(now)
    if timestamps:
        _rate[client_ip] = timestamps
    else:
        _rate.pop(client_ip, None)


# ─── Schemas ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    strategy: str = Field(default="hybrid", pattern="^(dense_only|sparse_only|hybrid|hybrid_rerank)$")
    filters: dict | None = Field(default=None, description='e.g. {"source": "news", "lang": "en"}')


class SourceItem(BaseModel):
    id: str
    score: float
    text_snippet: str = ""
    source: str = ""
    lang: str = ""


class QueryResponse(BaseModel):
    question: str
    answer: str
    faithfulness: float = 0.0
    sources: list[SourceItem]
    unsupported: bool = False
    cached: bool = False
    faithfulness_task_id: str = ""
    latency_ms: float = 0
    latency_retrieval_ms: float = 0
    latency_generation_ms: float = 0


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10)
    source: str = Field(default="api")


class IngestResponse(BaseModel):
    status: str
    n_chunks: int


# ─── UI ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def ui():
    here = os.path.dirname(os.path.abspath(__file__))
    path = None
    # Walk up to find index.html (handles both final/scripts/ and scripts/ launch dirs)
    d = here
    for _ in range(6):
        for rel in [os.path.join("index.html"), os.path.join("final", "index.html")]:
            candidate = os.path.normpath(os.path.join(d, rel))
            if os.path.isfile(candidate):
                path = candidate
                break
        if path:
            break
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    if path:
        return FileResponse(path, media_type="text/html")
    return HTMLResponse(f"<h2>index.html not found (searched from {here})</h2>", status_code=404)


# ─── Health ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    qdrant_ok = False
    try:
        from module_5_retrieval import get_client
        get_client().get_collections()
        qdrant_ok = True
    except Exception:
        pass
    ok = qdrant_ok
    return {"status": "ok" if ok else "degraded", "qdrant": qdrant_ok, "llm": True}


# ─── Query ───────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request, background_tasks: BackgroundTasks):
    _check_auth(request)
    client_ip = request.client.host if request.client else "unknown"
    _check_rate(client_ip)
    t0 = time.time()

    # Единый root span/trace в Langfuse для всего RAG-запроса
    lf_cm = None
    lf_obj = None
    q_config = {}
    if _langfuse_client:
        lf_cm = _langfuse_client.start_as_current_observation(
            name="rag_query",
            as_type="chain",
            input={"question": req.question, "top_k": req.top_k, "strategy": req.strategy},
        )
        lf_obj = lf_cm.__enter__()
        q_handler = FilledUsageCallbackHandler(
            trace_context={
                "trace_id": lf_obj.trace_id,
                "parent_span_id": lf_obj.id,
            }
        )
        q_config = {"callbacks": [q_handler]}

    cached = _cache_lookup(req.question, req.filters)
    if cached:
        cached["cached"] = True
        if lf_cm:
            lf_cm.__exit__(None, None, None)
        if _langfuse_client:
            _langfuse_client.flush()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _log_query_mysql,
                                   req.question, cached.get("answer", ""),
                                   _sources_to_json(None, [s if isinstance(s, SourceItem) else SourceItem(**s) for s in cached.get("sources", [])]),
                                   cached.get("latency_ms", 0), cached.get("faithfulness", 0.0), 1, client_ip)
        return QueryResponse(**cached)

    loop = asyncio.get_running_loop()
    strategy_fn = {
        "dense_only": search_dense,
        "sparse_only": search_sparse,
        "hybrid": search_hybrid,
        "hybrid_rerank": search_hybrid_rerank,
    }[req.strategy]

    # Спан retrieval внутри трейса rag_query
    retr_cm = None
    if _langfuse_client and lf_obj:
        retr_cm = _langfuse_client.start_as_current_observation(
            name="retrieval",
            as_type="retriever",
            trace_context={"trace_id": lf_obj.trace_id, "parent_span_id": lf_obj.id},
            input={"query": req.question, "top_k": req.top_k, "strategy": req.strategy},
        )
        retr_cm.__enter__()

    docs = await loop.run_in_executor(None, strategy_fn, req.question, req.top_k, req.filters)
    t1 = time.time()

    if retr_cm:
        retr_cm.__exit__(None, None, None)
    print(f"  docs={len(docs)}")

    if not docs:
        if lf_cm:
            lf_cm.__exit__(None, None, None)
        if _langfuse_client:
            _langfuse_client.flush()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _log_query_mysql,
                                   req.question, "No relevant documents found.",
                                   "[]", round((t1 - t0) * 1000, 1), 0.0, 0, client_ip)
        return QueryResponse(
            question=req.question,
            answer="No relevant documents found. Cannot answer the question.",
            sources=[],
            unsupported=True,
            latency_ms=round((t1 - t0) * 1000, 1),
            latency_retrieval_ms=round((t1 - t0) * 1000, 1),
        )

    context = format_context(docs)

    def _gen_traced():
        with propagate_attributes(trace_name="rag_query"):
            return generate_citations(req.question, context, llm, q_config)

    answer = await asyncio.get_running_loop().run_in_executor(None, _gen_traced)
    t2 = time.time()

    task_id = uuid.uuid4().hex[:12]
    background_tasks.add_task(_run_faithfulness, task_id, context, answer, q_config)

    resp = QueryResponse(
        question=req.question,
        answer=answer,
        sources=[
            SourceItem(
                id=d[0], score=round(d[2], 3), text_snippet=d[1][:200],
                source=d[3].get("source", ""), lang=d[3].get("lang", ""),
            )
            for d in docs
        ],
        faithfulness_task_id=task_id,
        latency_ms=round((t2 - t0) * 1000, 1),
        latency_retrieval_ms=round((t1 - t0) * 1000, 1),
        latency_generation_ms=round((t2 - t1) * 1000, 1),
    )
    _cache_store(req.question, resp.model_dump(), req.filters)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _log_query_mysql,
                               req.question, answer, _sources_to_json(None, resp.sources),
                               resp.latency_ms, 0.0, 0, client_ip)
    if lf_obj:
        lf_obj.update(output=resp.model_dump())
    if lf_cm:
        lf_cm.__exit__(None, None, None)
    if _langfuse_client:
        _langfuse_client.flush()
    return resp


# ─── Faithfulness Poll ────────────────────────────────────────────────────

class FaithfulnessResult(BaseModel):
    task_id: str
    faithfulness: float
    done: bool


@app.get("/faithfulness/{task_id}", response_model=FaithfulnessResult)
async def get_faithfulness(task_id: str):
    if task_id in _faith_results:
        return FaithfulnessResult(task_id=task_id, faithfulness=_faith_results[task_id], done=True)
    return FaithfulnessResult(task_id=task_id, faithfulness=0.0, done=False)


# ─── Logs ─────────────────────────────────────────────────────────────────

class LogRow(BaseModel):
    id: int
    ts: str
    question: str
    answer: str
    latency_ms: float
    faithfulness: float
    cache_hit: bool
    client_ip: str


class LogsResponse(BaseModel):
    count: int
    rows: list[LogRow]


@app.get("/logs", response_model=LogsResponse)
async def get_logs(request: Request, limit: int = 20):
    _check_auth(request)
    if not MYSQL_LOG_ENABLED:
        return LogsResponse(count=0, rows=[])
    try:
        conn = pymysql.connect(**MYSQL_CONFIG)
        try:
            with conn.cursor() as c:
                c.execute(
                    "SELECT id, ts, question, answer, latency_ms, faithfulness, cache_hit, client_ip "
                    "FROM rag_query_logs ORDER BY id DESC LIMIT %s",
                    (min(max(limit, 1), 100),),
                )
                rows = c.fetchall()
        finally:
            conn.close()
    except Exception as e:
        print(f"[mysql] logs read failed: {e}")
        return LogsResponse(count=0, rows=[])
    return LogsResponse(
        count=len(rows),
        rows=[
            LogRow(id=r[0], ts=str(r[1]), question=r[2], answer=r[3],
                   latency_ms=float(r[4] or 0), faithfulness=float(r[5] or 0),
                   cache_hit=bool(r[6]), client_ip=r[7] or "")
            for r in rows
        ],
    )


# ─── Stream ──────────────────────────────────────────────────────────────

async def _stream_gen(question: str, top_k: int = 5):
    docs = await asyncio.get_running_loop().run_in_executor(
        None, search_hybrid, question, top_k
    )
    context = format_context(docs)
    prompt = CITATIONS_PROMPT.invoke({"context": context, "question": question})

    yield f"data: {json.dumps({'type': 'meta', 'n_docs': len(docs)})}\n\n"

    async for chunk in llm.astream(prompt, config=LC_CONFIG):
        if chunk.content:
            yield f"data: {json.dumps({'type': 'token', 'text': chunk.content})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.get("/query/stream/{question:path}")
async def query_stream(question: str, request: Request):
    _check_auth(request)
    client_ip = request.client.host if request.client else "unknown"
    _check_rate(client_ip)
    return StreamingResponse(
        _stream_gen(question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Ingest ──────────────────────────────────────────────────────────────

from qdrant_client.http import models as qm


def _chunk_text(text: str, chunk_size=512, overlap=50):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, length_function=len
    )
    return splitter.split_text(text)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    from module_5_retrieval import get_sparse, get_client, COLLECTION
    loop = asyncio.get_running_loop()
    chunks = await loop.run_in_executor(None, _chunk_text, req.text)
    if not chunks:
        raise HTTPException(400, "Text too short to chunk")
    dense = get_embeddings("nomic-embed-text")
    sparse = get_sparse()
    client = get_client()
    dense_vecs = [dense.embed_query(c) for c in chunks]
    sparse_vecs = list(sparse.embed(chunks))
    points = []
    for i, (chunk, dv, sv) in enumerate(zip(chunks, dense_vecs, sparse_vecs)):
        points.append(qm.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dv.tolist(),
                "sparse": qm.SparseVector(
                    indices=sv.indices.tolist(),
                    values=sv.values.tolist(),
                ),
            },
            payload={
                "doc_id": f"{req.source}#{i}",
                "page_content": chunk,
                "source": req.source,
            },
        ))
    client.upsert(collection_name=COLLECTION, points=points)
    return IngestResponse(status="ok", n_chunks=len(points))


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"API: http://{args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)
