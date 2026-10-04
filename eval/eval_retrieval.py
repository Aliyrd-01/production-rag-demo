"""
Retrieval eval for rag_v2_q_test against MS MARCO qrels.

Usage:
    python eval/eval_retrieval.py                          # default strategies
    python eval/eval_retrieval.py --strategies hybrid,hybrid_hyde --k 20
    python eval/eval_retrieval.py --show-failures

Ground truth: eval/eval_set.json (built by build_eval_set.py).
Queries run serially so the reported latency is honest.
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_SET = os.path.join(HERE, "eval_set.json")
RESULTS = os.path.join(HERE, "results_retrieval.json")


def hit_rate_at_k(ranked, relevant, k):
    return 1.0 if set(ranked[:k]) & set(relevant) else 0.0


def mrr_at_k(ranked, relevant, k):
    rel = set(relevant)
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    rel = set(relevant)
    gains = [1.0 if d in rel else 0.0 for d in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    n_rel = min(len(rel), k)
    if n_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg else 0.0


def build_strategies(names):
    from module_5_retrieval import (
        search_dense,
        search_sparse,
        search_hybrid,
        search_hybrid_hyde,
        search_dense_hyde,
        search_hybrid_weighted,
        rerank_cross_encoder,
    )

    def dense_only(q, k):
        return [d[0] for d in search_dense(q, k=k)]

    def sparse_only(q, k):
        return [d[0] for d in search_sparse(q, k=k)]

    def hybrid(q, k):
        return [d[0] for d in search_hybrid(q, k=k)]

    def hybrid_hyde(q, k):
        return [d[0] for d in search_hybrid_hyde(q, k=k)]

    def dense_hyde(q, k):
        return [d[0] for d in search_dense_hyde(q, k=k)]

    def hybrid_ce(q, k):
        docs = search_hybrid(q, k=max(k, 20))
        return [d[0] for d in rerank_cross_encoder(q, docs, top_n=k)]

    def make_weighted(ws):
        def fn(q, k):
            return [d[0] for d in search_hybrid_weighted(q, k=k, w_dense=1.0, w_sparse=ws)]
        return fn

    table = {
        "dense_only": dense_only,
        "sparse_only": sparse_only,
        "hybrid": hybrid,
        "hybrid_hyde": hybrid_hyde,
        "dense_hyde": dense_hyde,
        "hybrid_ce": hybrid_ce,
        "w0.0": make_weighted(0.0),
        "w0.10": make_weighted(0.10),
        "w0.15": make_weighted(0.15),
        "w0.25": make_weighted(0.25),
        "w0.35": make_weighted(0.35),
        "w0.50": make_weighted(0.50),
        "w0.75": make_weighted(0.75),
        "w1.0": make_weighted(1.0),
    }
    missing = [n for n in names if n not in table]
    if missing:
        raise SystemExit("unknown strategies: %s (available: %s)" % (missing, ", ".join(table)))
    return [(n, table[n]) for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="dense_only,sparse_only,hybrid")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="only first N eval queries")
    ap.add_argument("--show-failures", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    with open(EVAL_SET, encoding="utf-8") as f:
        data = json.load(f)
    items = data["items"][: args.limit] if args.limit else data["items"]
    print("collection: %s | eval queries: %d | k=%d" % (data["collection"], len(items), args.k))
    print("-" * 78)

    rows = []
    all_out = {}
    for name, fn in build_strategies(args.strategies.split(",")):
        h5 = h10 = mrr10 = ndcg10 = 0.0
        lat = []
        misses = []
        for it in items:
            t0 = time.time()
            ranked = fn(it["query"], args.k)
            lat.append(time.time() - t0)
            h5 += hit_rate_at_k(ranked, it["relevant"], 5)
            h10 += hit_rate_at_k(ranked, it["relevant"], 10)
            mrr10 += mrr_at_k(ranked, it["relevant"], 10)
            ndcg10 += ndcg_at_k(ranked, it["relevant"], 10)
            if not set(ranked[:10]) & set(it["relevant"]):
                misses.append({"qid": it["qid"], "query": it["query"], "relevant": it["relevant"]})

        n = len(items)
        row = {
            "strategy": name,
            "k": args.k,
            "n": n,
            "hit@5": round(h5 / n, 4),
            "hit@10": round(h10 / n, 4),
            "mrr@10": round(mrr10 / n, 4),
            "ndcg@10": round(ndcg10 / n, 4),
            "avg_latency_s": round(sum(lat) / n, 3),
            "p95_latency_s": round(sorted(lat)[int(n * 0.95) - 1], 3),
        }
        rows.append(row)
        all_out[name] = misses
        print(
            "%-14s Hit@5 %.3f  Hit@10 %.3f  MRR@10 %.3f  NDCG@10 %.3f  "
            "lat %.2fs (p95 %.2fs)"
            % (name, row["hit@5"], row["hit@10"], row["mrr@10"], row["ndcg@10"],
               row["avg_latency_s"], row["p95_latency_s"])
        )

    print("-" * 78)
    if args.show_failures:
        for name, misses in all_out.items():
            print("\n### %s — %d misses in top-10" % (name, len(misses)))
            for m in misses[:12]:
                print("  [%s] %s" % (m["qid"], m["query"][:80]))

    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            hist = json.load(f)
    else:
        hist = []
    hist.append({"tag": args.tag or "run", "rows": rows})
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)
    print("\nsaved -> %s" % RESULTS)


if __name__ == "__main__":
    main()
