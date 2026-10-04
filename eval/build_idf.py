"""Считает document frequency по sparse-векторам корпуса.

Зачем: коллекция rag_v2_q_test создана без modifier=IDF, поэтому Qdrant считает
dot product по term-frequency без IDF-взвешивания и без нормализации длины.
Из-за этого sparse_only даёт Hit@5 0.46 и любое смешивание с dense ухудшает
результат. Пересоздавать коллекцию не хотим (переиндексация на 310k точек),
поэтому применяем BM25-веса на клиенте.

Результат: idf_stats.json = {"n_docs": N, "df": {term_index: doc_count}}.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from module_5_retrieval import COLLECTION, get_client

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "idf_stats.json")
BATCH = 2000


def main():
    client = get_client()

    df = {}
    n_docs = 0
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=BATCH,
            offset=offset,
            with_payload=False,
            with_vectors=["sparse"],
        )
        if not points:
            break

        for p in points:
            vec = (p.vector or {}).get("sparse")
            if not vec:
                continue
            indices = vec.indices if hasattr(vec, "indices") else vec["indices"]
            # один документ = один doc_id; у чанков разные индексы term,
            # но для DF считаем каждый point как отдельный документ коллекции.
            for idx in indices:
                df[int(idx)] = df.get(int(idx), 0) + 1
            n_docs += 1

        print(f"  {n_docs:,} points | {len(df):,} unique terms", flush=True)
        if offset is None:
            break

    stats = {"n_docs": n_docs, "df": {str(k): v for k, v in df.items()}}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(stats, f)

    print(f"\nsaved -> {OUT}")
    print(f"documents : {n_docs:,}")
    print(f"terms     : {len(df):,}")
    top = sorted(df.values(), reverse=True)[:5]
    print(f"most common term df: {top}")


if __name__ == "__main__":
    main()
