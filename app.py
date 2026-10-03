"""
Production RAG — Streamlit Community Cloud demo.

Стек: Qdrant Cloud (hybrid dense+sparse, RRF) + FastEmbed + LLM-пул с цитатами.
Датасет: rag_v2_q_test (MS MARCO, 310 746 пассажей) — индекс не пересобирается.

Локально:  streamlit run app.py
Облако:   share.streamlit.io (секреты — в App Settings → Secrets)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

import streamlit as st

st.set_page_config(page_title="Production RAG", page_icon="🔎", layout="centered")


# ─── Secrets → os.environ ───────────────────────────────────────────────────
# В облаке .env нет: секреты лежат в st.secrets. common.py и module_5_retrieval.py
# читают os.getenv(), поэтому переносим секреты в окружение ДО их импорта.
def load_secrets_to_env() -> list:
    if hasattr(st, "secrets") and st.secrets:
        try:
            items = dict(st.secrets)
        except Exception:
            items = {}
        loaded = []
        for k, v in items.items():
            if isinstance(v, str) and v and not os.environ.get(k):
                os.environ[k] = v
                loaded.append(k)
        if loaded:
            return loaded
    return []


load_secrets_to_env()


@st.cache_resource(show_spinner="Loading models & clients...")
def get_backend(strategy: str):
    """Инициализация retrieval-бэкенда один раз на процесс."""
    from module_5_retrieval import COLLECTION, get_client
    client = get_client()
    info = client.get_collection(COLLECTION)

    if strategy in ("hybrid", "dense_only", "sparse_only"):
        from module_5_retrieval import search_dense, search_sparse, search_hybrid
        fn = {"hybrid": search_hybrid, "dense_only": search_dense, "sparse_only": search_sparse}[strategy]
        return fn, COLLECTION, info.points_count, None

    if strategy == "hybrid_llm_rerank":
        from module_5_retrieval import search_hybrid_rerank
        return search_hybrid_rerank, COLLECTION, info.points_count, None

    if strategy == "hybrid_cross_encoder":
        from module_5_retrieval import search_hybrid, get_cross_encoder
        get_cross_encoder()  # тяжёлая модель — грузим только при выборе этой стратегии
        return search_hybrid, COLLECTION, info.points_count, "bge-reranker-base"

    raise ValueError(strategy)


@st.cache_resource(show_spinner="Loading LLM pool...")
def get_llm():
    from common import get_chat_llm
    return get_chat_llm(temperature=0)


def run_query(question: str, strategy: str, top_k: int, rerank_top: int):
    from module_7_generation import format_context, generate_citations
    from common import describe_llm

    fn, collection, points, reranker = get_backend(strategy)
    llm = get_llm()

    t0 = time.time()
    if strategy == "hybrid_llm_rerank":
        docs = fn(question, k=top_k, filters=None, rerank_top=rerank_top)
    else:
        docs = fn(question, top_k, None)
        if strategy == "hybrid_cross_encoder":
            from module_5_retrieval import rerank_cross_encoder
            docs = rerank_cross_encoder(question, docs, top_n=rerank_top)
    t1 = time.time()

    if not docs:
        return {
            "answer": "No relevant documents found — the question is not covered by the corpus.",
            "sources": [],
            "unsupported": True,
            "latency_ms": (t1 - t0) * 1000,
            "retrieval_ms": (t1 - t0) * 1000,
            "generation_ms": 0.0,
            "collection": collection,
            "points": points,
            "reranker": reranker,
            "llm": describe_llm(llm),
        }

    context = format_context(docs)
    answer = generate_citations(question, context, llm)
    t2 = time.time()
    llm_info = describe_llm(llm)

    return {
        "answer": answer,
        "sources": [
            {"id": d[0], "score": round(float(d[2]), 3), "snippet": d[1][:300]}
            for d in docs
        ],
        "unsupported": False,
        "latency_ms": (t2 - t0) * 1000,
        "retrieval_ms": (t1 - t0) * 1000,
        "generation_ms": (t2 - t1) * 1000,
        "collection": collection,
        "points": points,
        "reranker": reranker,
        "llm": llm_info,
    }


# ─── UI ─────────────────────────────────────────────────────────────────────

st.title("🔎 Production RAG")
st.caption("Hybrid retrieval (dense + sparse, RRF) → LLM answer with citations. Qdrant Cloud · FastEmbed · LangChain")

STRATEGIES = {
    "hybrid": "Hybrid — dense + sparse (RRF)",
    "dense_only": "Dense only (Nomic v1.5)",
    "sparse_only": "Sparse only (BM25)",
    "hybrid_llm_rerank": "Hybrid + LLM rerank",
    "hybrid_cross_encoder": "Hybrid + CrossEncoder rerank",
}

EXAMPLES = [
    "what color is amber urine",
    "what causes dark amber urine",
    "what do elevated liver enzymes mean",
    "how does a bill become law in the united states",
    "how much does an average person make for tutoring",
    "is autoimmune hepatitis a bile acid synthesis disorder",
]

with st.sidebar:
    st.header("Retrieval")
    strategy = st.selectbox(
        "Strategy", list(STRATEGIES), index=0,
        format_func=lambda k: STRATEGIES[k],
    )
    top_k = st.slider("Candidates (top K)", 1, 20, 8)
    need_rerank = strategy in ("hybrid_llm_rerank", "hybrid_cross_encoder")
    rerank_top = st.slider("After rerank", 1, 10, 5) if need_rerank else 5

    st.divider()
    st.markdown("**Corpus**")
    st.code("rag_v2_q_test", language=None)
    st.caption("MS MARCO · 310 746 passages · dense 768d (COSINE/INT8) + sparse BM25")

    st.divider()
    st.markdown("**Try**")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True, key=f"ex_{q}"):
            st.session_state["q"] = q

question = st.text_input(
    "Question", key="q",
    placeholder="e.g. what color is amber urine",
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Retrieving + generating (first run downloads the embedding model, ~1 min)..."):
            result = run_query(question.strip(), strategy, top_k, rerank_top)
    except Exception as e:
        st.error(f"**{type(e).__name__}:** {e}")
        st.stop()

    if result["unsupported"]:
        st.warning("**No answer in corpus**")
        st.write(result["answer"])
    else:
        st.markdown("### Answer")
        st.write(result["answer"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Total", f"{result['latency_ms'] / 1000:.1f}s")
    m2.metric("Retrieval", f"{result['retrieval_ms'] / 1000:.1f}s")
    m3.metric("Generation", f"{result['generation_ms'] / 1000:.1f}s")

    info = result["llm"]
    served = info["served"]
    llm_line = f"**LLM:** `{info['provider']}` · served `{served}`"
    if info["served"] != info["requested"]:
        llm_line += f" (requested `{info['requested']}`)"
    st.caption(llm_line)

    with st.expander(f"Sources ({len(result['sources'])})", expanded=True):
        for i, s in enumerate(result["sources"], 1):
            st.markdown(f"**[{i}]** `msmarco#{s['id']}` · score `{s['score']}`")
            st.text(s["snippet"])

    st.divider()
    c = st.caption(
        f"collection `{result['collection']}` · {result['points']:,} passages · "
        f"strategy `{strategy}` · top_k {top_k}"
    )
    if result["reranker"]:
        c.caption(f"reranker `{result['reranker']}`")