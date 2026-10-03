"""
Module 5 — Retrieval: Qdrant Hybrid Search + Reranking.

Стратегии:
  1. dense_only  — nomic-embed-text (768d) via Ollama
  2. sparse_only — Qdrant/bm25 via FastEmbed
  3. hybrid      — dense + sparse → Qdrant RRF fusion
  4. hybrid_rerank — hybrid → CrossEncoder (bge-reranker)

Benchmark: 10 вопросов из MS MARCO, R@5, R@10, MRR, latency.

Запуск:
  python scripts/module_5_retrieval.py                    # benchmark
  python scripts/module_5_retrieval.py --interactive      # интерактив
"""

import os, sys, time, json, argparse
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
    dense = get_dense()
    dvec = dense.embed_query(query)
    results = client.query_points(
        collection_name=COLLECTION,
        query=dvec,
        using="dense",
        limit=k,
        with_payload=True,
        query_filter=_to_filter(filters),
    )
    out = []
    for p in results.points:
        payload = p.payload or {}
        out.append((
            str(payload.get("doc_id", payload.get("id", p.id))),
            payload.get("page_content", ""),
            float(p.score),
            payload,
        ))
    return out


def search_sparse(query: str, k: int = 10, filters: dict | None = None):
    client = get_client()
    sparse = get_sparse()
    svec = list(sparse.embed([query]))[0]
    sparse_vector = qm.SparseVector(
        indices=svec.indices.tolist(),
        values=svec.values.tolist(),
    )
    results = client.query_points(
        collection_name=COLLECTION,
        query=sparse_vector,
        using="sparse",
        limit=k,
        with_payload=True,
        query_filter=_to_filter(filters),
    )
    out = []
    for p in results.points:
        payload = p.payload or {}
        out.append((
            str(payload.get("doc_id", payload.get("id", p.id))),
            payload.get("page_content", ""),
            float(p.score),
            payload,
        ))
    return out


def search_hybrid(query: str, k: int = 10, filters: dict | None = None):
    client = get_client()
    dense = get_dense()
    dvec = dense.embed_query(query)
    qfilter = _to_filter(filters)
    try:
        sparse = get_sparse()
        svec = list(sparse.embed([query]))[0]
        sparse_vector = qm.SparseVector(
            indices=svec.indices.tolist(),
            values=svec.values.tolist(),
        )
        results = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                qm.Prefetch(query=dvec, using="dense", limit=k * 4),
                qm.Prefetch(query=sparse_vector, using="sparse", limit=k * 4),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=k, with_payload=True, query_filter=qfilter,
        )
    except Exception as e:
        print(f"  ⚠ Sparse model failed ({e}), falling back to dense-only")
        results = client.query_points(
            collection_name=COLLECTION,
            query=dvec, using="dense",
            limit=k, with_payload=True, query_filter=qfilter,
        )
    out = []
    for p in results.points:
        payload = p.payload or {}
        out.append((
            str(payload.get("doc_id", payload.get("id", p.id))),
            payload.get("page_content", ""),
            float(p.score),
            payload,
        ))
    return out


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
