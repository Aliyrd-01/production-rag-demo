"""
Builds a reproducible retrieval eval set for the rag_v2_q_test collection.

Ground truth comes from MS MARCO qrels (train.tsv): binary relevance per
(query-id, corpus-id). Only queries whose relevant passage actually exists in
our collection are usable, because the collection holds 310 746 of the ~8.8M
MS MARCO passages, so most train qrels point at documents we do not have.

Output: eval/eval_set.json  +  eval/collection_doc_ids.txt (cache)
"""

import io
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

from module_5_retrieval import get_client, COLLECTION

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
DATA = r"D:\AI\LangChain\data\corpus\msmarco"
CACHE = os.path.join(HERE, "collection_doc_ids.txt")
OUT = os.path.join(HERE, "eval_set.json")
N_TARGET = 100
SEED = 42


def collect_doc_ids():
    if os.path.exists(CACHE):
        with io.open(CACHE, encoding="utf-8") as f:
            ids = {ln.strip() for ln in f if ln.strip()}
        print("doc_id cache hit: %d ids" % len(ids))
        return ids

    print("scrolling collection (one-off, cached afterwards)...", flush=True)
    client = get_client()
    ids = set()
    offset = None
    pages = 0
    while True:
        recs, offset = client.scroll(
            collection_name=COLLECTION,
            limit=8192,
            offset=offset,
            with_payload=["doc_id"],
            with_vectors=False,
        )
        for r in recs:
            did = (r.payload or {}).get("doc_id")
            if did is not None:
                ids.add(str(did))
        pages += 1
        print("  page %d -> %d doc_ids" % (pages, len(ids)), flush=True)
        if offset is None:
            break

    with io.open(CACHE, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(ids)))
    print("cached %d doc_ids -> %s" % (len(ids), CACHE))
    return ids


def load_queries():
    out = {}
    with io.open(os.path.join(DATA, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            qid = str(o.get("_id", "")).strip()
            txt = (o.get("text") or "").strip()
            if qid and txt:
                out[qid] = txt
    return out


def load_qrels():
    gt = {}
    with io.open(os.path.join(DATA, "qrels", "train.tsv"), encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if score not in ("1", "2"):
                continue
            gt.setdefault(qid, set()).add(did)
    return gt


def main():
    present = collect_doc_ids()
    queries = load_queries()
    gt = load_qrels()
    print("queries: %d | qrel qids: %d | doc_ids in collection: %d" % (len(queries), len(gt), len(present)))

    eligible = []
    for qid, rel in gt.items():
        if qid not in queries:
            continue
        hit = rel & present
        if hit:
            eligible.append((qid, queries[qid], sorted(hit)))

    print("eligible queries (relevant passage present in collection): %d" % len(eligible))

    rng = random.Random(SEED)
    rng.shuffle(eligible)
    sample = eligible[:N_TARGET]

    payload = {
        "collection": COLLECTION,
        "source": "MS MARCO qrels train.tsv",
        "seed": SEED,
        "n": len(sample),
        "items": [{"qid": q, "query": t, "relevant": h} for q, t, h in sample],
    }
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("wrote %s with %d queries" % (OUT, len(sample)))
    for q, t, h in sample[:5]:
        print("  [%s] %s -> %s" % (q, t[:70], h))


if __name__ == "__main__":
    main()
