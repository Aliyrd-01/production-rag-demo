"""
Module 5 — Retrieval: Qdrant Hybrid Search + Reranking.

Стратегии:
  1. dense_only  — nomic-embed-text (768d) via FastEmbed
  2. sparse_only — Qdrant/bm25 via FastEmbed
  3. hybrid      — dense + sparse → Qdrant RRF fusion
  4. hybrid_rerank — hybrid → CrossEncoder (bge-reranker)

Benchmark: 10 вопросов из MS MARCO, R@5, R@10, MRR, latency.

Запуск:
  python scripts/module_5_retrieval.py                    # benchmark
  python scripts/module_5_retrieval.py --interactive      # интерактив
"""

import math
import os, sys, time, json, argparse
from functools import lru_cache
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass  # под Streamlit stdout — обёртка, reconfigure может отсутствовать
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from common import get_fastembed_dense, get_fastembed_sparse, get_fastembed_reranker, get_embeddings


BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(BASE, "..", "..", "data"))
CORPUS_PATH = os.path.join(DATA, "corpus", "msmarco", "corpus.jsonl")
QRELS_PATH = os.path.join(DATA, "corpus", "msmarco", "qrels", "train.tsv")
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_v2_q_test")

_client = None


def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=False,
            timeout=90,
            check_compatibility=False,
        )
    return _client


def get_dense():
    return get_embeddings("fastembed")


def get_sparse():
    return get_fastembed_sparse()


def _fix_mojibake(text: str) -> str:
    """Repair UTF-8 text that was decoded as cp1252 (e.g. "Colorâ€"" -> "Color—").

    The MS MARCO corpus was indexed with a broken encoding, so retrieved payloads
    contain mojibake. Repairing on read fixes both the LLM context and the UI
    without reindexing the collection.
    """
    if not text:
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _row(point):
    """Uniform (doc_id, text, score, payload) row with mojibake repaired."""
    payload = point.payload or {}
    return (
        str(payload.get("doc_id", payload.get("id", point.id))),
        _fix_mojibake(payload.get("page_content", "")),
        float(point.score),
        payload,
    )


_IDF_CACHE = {}


def _load_idf():
    """BM25 IDF-статистика, посчитанная по корпусу (eval/build_idf.py).

    Коллекция создана без modifier=IDF, поэтому Qdrant не применяет IDF сам.
    Без него sparse-вектор — это term-frequency: частые слова вроде "what" весят
    столько же, сколько редкие термины, и sparse-поиск деградирует (Hit@5 0.46).
    """
    if _IDF_CACHE:
        return _IDF_CACHE
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "idf_stats.json"
    )
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        _IDF_CACHE["n_docs"] = raw["n_docs"]
        _IDF_CACHE["df"] = {int(k): v for k, v in raw["df"].items()}
        _IDF_CACHE["avg_len"] = sum(_IDF_CACHE["df"].values()) / max(len(_IDF_CACHE["df"]), 1)
    except Exception as e:
        print(f"  ⚠ IDF stats unavailable ({e}), using raw term-frequency")
        return None
    return _IDF_CACHE


def apply_bm25_idf(svec, k1: float = 1.2, b: float = 0.75):
    """Перевзвешивает sparse-вектор запроса формулой BM25 (idf + длина запроса).

    svec — результат Qdrant/bm25 из fastembed (уже с tf-saturation).
    Домножаем на idf и нормируем по длине запроса, чтобы короткие запросы
    не получали систематически больший вес.
    """
    stats = _load_idf()
    if not stats:
        return svec

    n_docs = stats["n_docs"]
    df = stats["df"]
    avg_len = stats["avg_len"] or 1.0

    indices = svec.indices.tolist() if hasattr(svec.indices, "tolist") else list(svec.indices)
    values = svec.values.tolist() if hasattr(svec.values, "tolist") else list(svec.values)

    norm = k1 * (1 - b + b * len(indices) / avg_len)

    out_idx, out_val = [], []
    for idx, tf in zip(indices, values):
        freq = df.get(int(idx), 0)
        idf = max(math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)), 1e-6)
        out_idx.append(int(idx))
        out_val.append(float(tf) * idf / norm)

    return qm.SparseVector(indices=out_idx, values=out_val)


@lru_cache(maxsize=512)
def _dense_query_vec(query: str):
    return tuple(get_dense().embed_query(query))


def _dense_query(query: str):
    return list(_dense_query_vec(query))


@lru_cache(maxsize=512)
def _sparse_query_tuples(query: str):
    svec = list(get_sparse().embed([query]))[0]
    out = apply_bm25_idf(svec)
    idx = out.indices.tolist() if hasattr(out.indices, "tolist") else list(out.indices)
    vals = out.values.tolist() if hasattr(out.values, "tolist") else list(out.values)
    return tuple(int(i) for i in idx), tuple(float(v) for v in vals)


def _sparse_query(query: str):
    idx, vals = _sparse_query_tuples(query)
    return qm.SparseVector(indices=list(idx), values=list(vals))


def get_cross_encoder():
    return get_fastembed_reranker()


TEST_QUERIES = [
    {"query": "what teas are good for what",            "qids": ["900736"]},
    {"query": "what was the immediate impact of the success of the manhattan project", "qids": ["1185869"]},
    {"query": "what color is amber urine",              "qids": ["597651"]},
    {"query": "is autoimmune hepatitis a bile acid synthesis disorder", "qids": ["403613"]},
    {"query": "elegxo meaning",                         "qids": ["1183785"]},
    {"query": "how much does an average person make for tutoring", "qids": ["312651"]},
    {"query": "can you use a calculator on the compass test", "qids": ["80385"]},
    {"query": "what does physical medicine do",          "qids": ["645590"]},
    {"query": "weather history in amsterdam",            "qids": ["543163"]},
    {"query": "justice is designed to repair the harm to victim the community and the offender", "qids": ["1185868"]},
]


def load_ground_truth():
    gt = {}
    with open(QRELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                gt.setdefault(parts[0], set()).add(str(parts[1]))
    return gt


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    top_k = set(str(rid) for rid in retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mrr_at_k(retrieved_ids, relevant_ids, k):
    for i, rid in enumerate(retrieved_ids[:k]):
        if str(rid) in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def _to_filter(filters: dict | None):
    if not filters:
        return None
    conditions = []
    for key, val in filters.items():
        if not val:
            continue
        if isinstance(val, dict):
            range_kw = {}
            if "gte" in val: range_kw["gte"] = val["gte"]
            if "lte" in val: range_kw["lte"] = val["lte"]
            if "gt" in val: range_kw["gt"] = val["gt"]
            if "lt" in val: range_kw["lt"] = val["lt"]
            if range_kw:
                conditions.append(qm.FieldCondition(key=key, range=qm.Range(**range_kw)))
            continue
        conditions.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=val)))
    return qm.Filter(must=conditions) if conditions else None


def search_dense(query: str, k: int = 10, filters: dict | None = None):
    client = get_client()
    results = client.query_points(
        collection_name=COLLECTION,
        query=_dense_query(query),
        using="dense",
        limit=max(k * 4, 20),
        with_payload=True,
        query_filter=_to_filter(filters),
    )
    return _dedupe_by_doc_id([_row(p) for p in results.points], k)


def search_sparse(query: str, k: int = 10, filters: dict | None = None):
    client = get_client()
    results = client.query_points(
        collection_name=COLLECTION,
        query=_sparse_query(query),
        using="sparse",
        limit=max(k * 4, 20),
        with_payload=True,
        query_filter=_to_filter(filters),
    )
    return _dedupe_by_doc_id([_row(p) for p in results.points], k)


def search_hybrid(query: str, k: int = 10, filters: dict | None = None):
    client = get_client()
    qfilter = _to_filter(filters)
    try:
        results = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                qm.Prefetch(query=_dense_query(query), using="dense", limit=k * 4),
                qm.Prefetch(query=_sparse_query(query), using="sparse", limit=k * 4),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=max(k * 4, 20), with_payload=True, query_filter=qfilter,
        )
    except Exception as e:
        print(f"  ⚠ Sparse model failed ({e}), falling back to dense-only")
        results = client.query_points(
            collection_name=COLLECTION,
            query=_dense_query(query), using="dense",
            limit=max(k * 4, 20), with_payload=True, query_filter=qfilter,
        )
    return _dedupe_by_doc_id([_row(p) for p in results.points], k)


def _rrf_merge(rank_lists, k: int = 10, k_rrf: int = 60):
    """Reciprocal Rank Fusion для нескольких списков документов."""
    scores, docs = {}, {}
    for rl in rank_lists:
        for rank, d in enumerate(rl):
            did = d[0]
            scores[did] = scores.get(did, 0.0) + 1.0 / (k_rrf + rank + 1)
            docs.setdefault(did, d)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out = []
    for did, sc in ranked[:k]:
        d = docs[did]
        out.append((d[0], d[1], sc) + (d[3:] if len(d) > 3 else ()))
    return out


_HYDE_PROMPT = (
    "Write a short passage of 2-3 sentences that would answer the question below. "
    "Write it as if it were an excerpt from a reference document: state facts directly, "
    "no preamble, no hedging, do not mention that it is hypothetical, do not ask questions. "
    "Reply with the passage only.\n\n"
    "Question: {query}\n\nPassage:"
)


def make_hypothetical_answer(query: str, llm=None) -> str:
    """HyDE: LLM пишет гипотетический ответ, чтобы он векторизовался ближе к реальному."""
    from common import get_chat_llm
    llm = llm or get_chat_llm(temperature=0)
    try:
        out = llm.invoke(_HYDE_PROMPT.format(query=query))
        return (out.content or "").strip()
    except Exception as e:
        print("  ⚠ HyDE generation failed (%s), falling back to plain hybrid" % e)
        return ""


def _search_hybrid_hyde_sequential(query: str, k: int = 10, filters: dict | None = None, llm=None):
    hypo = make_hypothetical_answer(query, llm=llm)
    if not hypo:
        return search_hybrid(query, k=k, filters=filters)
    base = search_hybrid(query, k=k, filters=filters)
    hy = search_hybrid(hypo, k=k, filters=filters)
    return _rrf_merge([base, hy], k=k)


def search_hybrid_hyde(query: str, k: int = 10, filters: dict | None = None, llm=None):
    """Hybrid over the question AND a HyDE passage, fused server-side (RRF).

    One Qdrant round trip with four prefetches instead of two sequential searches,
    which removes the latency penalty that made HyDE expensive.
    """
    hypo = make_hypothetical_answer(query, llm=llm)
    if not hypo:
        return search_hybrid(query, k=k, filters=filters)
    client = get_client()
    qfilter = _to_filter(filters)
    try:
        results = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                qm.Prefetch(query=_dense_query(query), using="dense", limit=k * 4),
                qm.Prefetch(query=_sparse_query(query), using="sparse", limit=k * 4),
                qm.Prefetch(query=_dense_query(hypo), using="dense", limit=k * 4),
                qm.Prefetch(query=_sparse_query(hypo), using="sparse", limit=k * 4),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=max(k * 4, 20), with_payload=True, query_filter=qfilter,
        )
        return _dedupe_by_doc_id([_row(p) for p in results.points], k)
    except Exception as e:
        print(f"  ⚠ parallel HyDE failed ({e}), falling back to sequential")
        return _search_hybrid_hyde_sequential(query, k=k, filters=filters, llm=llm)


def _dedupe_by_doc_id(docs, k: int, max_chars: int = 1600):
    """Group retrieved chunks by doc_id and merge their text.

    The collection stores chunks; one document can hold several. The old behaviour
    dropped every chunk but the best, so the LLM never saw the rest of a document.
    Now distinct chunk texts of the same doc_id are merged in chunk order (capped at
    max_chars), keeping the best score. Neighbour chunks only merge when both were
    retrieved, so behaviour degrades gracefully where a doc has a single chunk.
    """
    groups, order = {}, []
    for d in docs:
        did = d[0]
        if did not in groups:
            groups[did] = {"best": d, "chunks": []}
            order.append(did)
        g = groups[did]
        if d[2] > g["best"][2]:
            g["best"] = d
        cid = (d[3] or {}).get("chunk_id")
        g["chunks"].append((cid if isinstance(cid, int) else 0, d[1]))

    out = []
    for did in order:
        g = groups[did]
        best = g["best"]
        seen, parts, total = set(), [], 0
        for _, text in sorted(g["chunks"], key=lambda x: x[0]):
            t = (text or "").strip()
            if not t or t in seen:
                continue
            seen.add(t)
            if parts and total + len(t) > max_chars:
                break
            parts.append(t)
            total += len(t)
        merged = "\n".join(parts) if parts else best[1]
        out.append((best[0], merged, best[2]) + tuple(best[3:]))
        if len(out) >= k:
            break
    return out


def _rows(results):
    return [_row(p) for p in results.points]


def _weighted_merge(lists, k=10, k_rrf=60):
    scores, docs = {}, {}
    for docs_list, w in lists:
        if w <= 0:
            continue
        for rank, d in enumerate(_dedupe_by_doc_id(docs_list, len(docs_list))):
            did = d[0]
            scores[did] = scores.get(did, 0.0) + w / (k_rrf + rank + 1)
            docs.setdefault(did, d)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out = []
    for did, sc in ranked[:k]:
        d = docs[did]
        out.append((d[0], d[1], sc) + (d[3:] if len(d) > 3 else ()))
    return out


def search_hybrid_weighted(query: str, k: int = 10, filters: dict | None = None,
                           w_dense: float = 1.0, w_sparse: float = 1.0, pool: int = 40):
    """RRF fusion с весами: чем ниже w_sparse, тем меньше шума от BM25."""
    client = get_client()
    qfilter = _to_filter(filters)
    lists = []

    if w_dense > 0:
        r = client.query_points(
            collection_name=COLLECTION, query=get_dense().embed_query(query),
            using="dense", limit=pool, with_payload=True, query_filter=qfilter,
        )
        lists.append((_rows(r), w_dense))

    if w_sparse > 0:
        try:
            svec = list(get_sparse().embed([query]))[0]
            r = client.query_points(
                collection_name=COLLECTION,
                query=apply_bm25_idf(svec),
                using="sparse", limit=pool, with_payload=True, query_filter=qfilter,
            )
            lists.append((_rows(r), w_sparse))
        except Exception as e:
            print("  ⚠ Sparse model failed (%s), dense-only" % e)

    if not lists:
        return []
    return _weighted_merge(lists, k=k)


def search_dense_hyde(query: str, k: int = 10, filters: dict | None = None, llm=None):
    """Dense по исходному вопросу + dense по гипотетическому ответу, слитые RRF."""
    hypo = make_hypothetical_answer(query, llm=llm)
    if not hypo:
        return search_dense(query, k=k, filters=filters)
    base = search_dense(query, k=k, filters=filters)
    hy = search_dense(hypo, k=k, filters=filters)
    return _rrf_merge([base, hy], k=k)


def rerank_cross_encoder(query: str, docs, top_n: int = 5):
    ce = get_cross_encoder()
    if not docs:
        return []
    pairs = [(query, d[1]) for d in docs]
    scores = [float(s) for s in ce.predict(pairs)]
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [(d[0], d[1], s) + (d[3:] if len(d) > 3 else ()) for d, s in ranked[:top_n]]


_LLM_RERANK_PROMPT = (
    "You are a search relevance judge. Given the query and a list of candidate "
    "passages, score EACH passage 0.0-1.0 on how relevant it is to answering the query. "
    "Return ONLY a JSON array of numbers, one per passage, in the same order.\n\n"
    "Query: {query}\n\nPassages:\n{passages}"
)


def rerank_llm(query: str, docs, top_n: int = 5, llm=None):
    """Rerank candidates with the main chat LLM (big-pickle)."""
    if not docs:
        return []
    from common import get_chat_llm
    llm = llm or get_chat_llm(temperature=0)
    passages = "\n\n".join(f"[{i+1}] {d[1][:800]}" for i, d in enumerate(docs))
    prompt = _LLM_RERANK_PROMPT.format(query=query, passages=passages)
    import re, json as _json
    raw = llm.invoke(prompt).content
    try:
        scores = _json.loads(re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.M))
    except Exception:
        m = re.search(r"\[[\d.,\s]+\]", raw)
        scores = _json.loads(m.group(0)) if m else [0.5] * len(docs)
    if not isinstance(scores, list):
        scores = [0.5] * len(docs)
    scores = [float(s) if isinstance(s, (int, float)) else 0.5 for s in scores]
    while len(scores) < len(docs):
        scores.append(0.5)
    scores = scores[:len(docs)]
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [(d[0], d[1], s) + (d[3:] if len(d) > 3 else ()) for d, s in ranked[:top_n]]


def search_hybrid_rerank(query: str, k: int = 10, filters: dict | None = None, rerank_top: int = 5):
    docs = search_hybrid(query, k=k, filters=filters)
    return rerank_llm(query, docs, top_n=rerank_top)


def search_hybrid_hyde_ce(query: str, k: int = 10, filters: dict | None = None,
                          llm=None, rerank_top: int = 5):
    """HyDE-enhanced hybrid, then CrossEncoder rerank (best of both)."""
    docs = search_hybrid_hyde(query, k=max(k, 20), filters=filters, llm=llm)
    return rerank_cross_encoder(query, docs, top_n=rerank_top)


_REWRITE_PROMPT = (
    "Rewrite the search query below into {n} different short queries that express "
    "the same information need with different wording. Return ONLY the queries, "
    "one per line, no numbering, no extra text.\n\nQuery: {query}\n"
)


def make_rewrites(query: str, n: int = 3, llm=None) -> list:
    """Multi-query: ask the LLM for n paraphrases of the question."""
    from common import get_chat_llm
    llm = llm or get_chat_llm(temperature=0)
    try:
        raw = llm.invoke(_REWRITE_PROMPT.format(n=n, query=query)).content or ""
    except Exception as e:
        print("  ⚠ rewrite generation failed (%s)" % e)
        return []
    out = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-*0123456789). ").strip()
        if line and line.lower() != query.lower():
            out.append(line)
        if len(out) >= n:
            break
    return out


def search_multi_query(query: str, k: int = 10, filters: dict | None = None, llm=None):
    """Hybrid for the original question plus LLM rewrites, merged with RRF."""
    queries = [query] + make_rewrites(query, llm=llm)
    if len(queries) == 1:
        return search_hybrid(query, k=k, filters=filters)
    lists = [search_hybrid(q, k=max(k, 20), filters=filters) for q in queries]
    return _rrf_merge(lists, k=k)


def _diversify(docs, k: int, max_overlap: float = 0.8):
    """Lexical diversity (MMR proxy without vectors): drop near-duplicate chunks."""
    import re as _re

    def toks(s):
        return set(_re.findall(r"[a-z0-9]+", (s or "").lower()))

    picked, picked_toks = [], []
    for d in docs:
        t = toks(d[1])
        if not t:
            continue
        if any(len(t & pt) / max(1, len(t | pt)) > max_overlap for pt in picked_toks):
            continue
        picked.append(d)
        picked_toks.append(t)
        if len(picked) >= k:
            break
    return picked or docs[:k]


def search_hybrid_diverse(query: str, k: int = 10, filters: dict | None = None):
    """Hybrid over a wider pool, then lexical MMR to remove near-duplicates."""
    pool = search_hybrid(query, k=max(k * 3, 20), filters=filters)
    return _diversify(pool, k)


def trace_hybrid(query: str, k: int = 10):
    """Per-branch candidates for the retrieval-trace UI (dense vs sparse vs fused)."""
    client = get_client()
    dense = client.query_points(
        collection_name=COLLECTION, query=_dense_query(query), using="dense",
        limit=max(k * 4, 20), with_payload=True,
    )
    sparse_rows = []
    try:
        sparse = client.query_points(
            collection_name=COLLECTION, query=_sparse_query(query), using="sparse",
            limit=max(k * 4, 20), with_payload=True,
        )
        sparse_rows = [_row(p) for p in sparse.points]
    except Exception:
        sparse_rows = []
    try:
        fused = search_hybrid(query, k=k)
    except Exception:
        fused = []
    return {
        "dense": _dedupe_by_doc_id([_row(p) for p in dense.points], k),
        "sparse": _dedupe_by_doc_id(sparse_rows, k) if sparse_rows else [],
        "fused": fused,
    }


def run_benchmark():
    gt_raw = load_ground_truth()
    client = get_client()
    info = client.get_collection(COLLECTION)
    print(f"Collection: {COLLECTION} — {info.points_count} points, status={info.status}")

    # Build set of doc_ids that exist in our collection
    print("Collecting doc_ids from collection...")
    our_doc_ids = set()
    next_offset = None
    while True:
        res = client.scroll(COLLECTION, limit=5000, offset=next_offset, with_payload=["doc_id"], with_vectors=False)
        for p in res[0]:
            did = p.payload.get("doc_id")
            if did:
                our_doc_ids.add(str(did))
        next_offset = res[1]
        if next_offset is None:
            break
    print(f"  {len(our_doc_ids)} unique doc_ids in collection")

    # Filter ground truth: only keep docs that exist in our collection
    gt = {}
    for qid, docset in gt_raw.items():
        filtered = docset & our_doc_ids
        if filtered:
            gt[qid] = filtered
    print(f"  {len(gt)} queries with relevant docs in collection")

    strategies = {
        "dense_only":  search_dense,
        "sparse_only": search_sparse,
        "hybrid":      search_hybrid,
    }

    results = {name: {"r5": [], "r10": [], "mrr": [], "latency": []} for name in strategies}
    tested = 0

    for qi, item in enumerate(TEST_QUERIES):
        query = item["query"]
        qid = item["qids"][0]
        relevant = gt.get(qid, set())
        if not relevant:
            print(f"\n[{qi+1}/10] \"{query}\"  SKIP (no relevant docs in collection)")
            continue
        tested += 1
        print(f"\n[{qi+1}/10] \"{query}\"  relevant={len(relevant)} docs")

        for name, fn in strategies.items():
            t0 = time.time()
            docs = fn(query)
            latency = (time.time() - t0) * 1000

            doc_ids = [d[0] for d in docs]
            docs = [d[:3] for d in docs]
            r5 = recall_at_k(doc_ids, relevant, 5)
            r10 = recall_at_k(doc_ids, relevant, 10)
            mrr = mrr_at_k(doc_ids, relevant, 10)

            results[name]["r5"].append(r5)
            results[name]["r10"].append(r10)
            results[name]["mrr"].append(mrr)
            results[name]["latency"].append(latency)

            print(f"  {name:15s}  R@5={r5:.2f}  R@10={r10:.2f}  MRR={mrr:.3f}  {latency:.0f}ms")

    print(f"\nResults ({tested} queries with relevant docs in collection):")
    print("=" * 70)
    print(f"{'Strategy':15s}  {'R@5':>6s}  {'R@10':>6s}  {'MRR':>6s}  {'Latency':>10s}")
    print("-" * 70)
    for name, m in results.items():
        n = len(m["r5"])
        avg_r5 = sum(m["r5"]) / n if n else 0
        avg_r10 = sum(m["r10"]) / n if n else 0
        avg_mrr = sum(m["mrr"]) / n if n else 0
        avg_lat = sum(m["latency"]) / n if n else 0
        print(f"  {name:15s}  {avg_r5:5.2f}  {avg_r10:5.2f}  {avg_mrr:5.3f}  {avg_lat:8.0f}ms")
    print("=" * 70)


def interactive_demo():
    print("\n=== Module 5: Retrieval Strategies (interactive) ===")
    print("Commands: /help /strategies /quit\n")

    strategies = ["dense_only", "sparse_only", "hybrid"]

    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/help":
            print("Commands:")
            print("  <query>       — search with all strategies")
            print("  /rerank <q>   — hybrid search + CrossEncoder rerank")
            print("  /strategies   — list active strategies")
            print("  /quit         — exit")
            continue
        if q == "/strategies":
            print(f"Active: {', '.join(strategies)}")
            continue

        do_rerank = q.startswith("/rerank ")
        query = q[len("/rerank "):] if do_rerank else q

        for name in strategies:
            t0 = time.time()
            if name == "dense_only":
                docs = search_dense(query)
            elif name == "sparse_only":
                docs = search_sparse(query)
            elif name == "hybrid":
                docs = search_hybrid(query)
            else:
                docs = []
            latency = (time.time() - t0) * 1000

            print(f"\n[{name}] {len(docs)} docs, {latency:.0f}ms")
            for i, d in enumerate(docs[:5]):
                print(f"  [{i}] (id={d[0]}) {d[1][:120]}")
                pl = d[3]
                src = pl.get("source", "")
                lang = pl.get("lang", "")
                if src or lang:
                    print(f"       source={src} lang={lang}")

        if do_rerank:
            print(f"\n--- CrossEncoder rerank (bge-reranker, top 5) ---")
            base = search_hybrid(query)
            t0 = time.time()
            reranked = rerank_cross_encoder(query, base, top_n=5)
            ms = (time.time() - t0) * 1000
            print(f"  Speed: {ms:.0f}ms")
            for i, d in enumerate(reranked):
                print(f"  [{i}] (id={d[0]}) {d[1][:120]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="interactive demo")
    args = parser.parse_args()

    if args.interactive:
        interactive_demo()
    else:
        run_benchmark()
