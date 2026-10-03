"""
Production RAG — Streamlit app (cloud demo).

Деплой: Streamlit Community Cloud (share.streamlit.io)
Запуск локально:  streamlit run app.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

import streamlit as st

st.set_page_config(page_title="Production RAG", page_icon="🔎", layout="centered")


@st.cache_resource
def get_llm():
    from common import get_chat_llm
    return get_chat_llm(temperature=0)


@st.cache_resource
def get_strategy(name: str):
    from module_5_retrieval import COLLECTION, search_dense, search_sparse, search_hybrid
    return {
        "hybrid": search_hybrid,
        "dense_only": search_dense,
        "sparse_only": search_sparse,
    }[name], COLLECTION


def run_query(question: str, strategy: str, top_k: int):
    fn, collection = get_strategy(strategy)
    from module_7_generation import format_context, generate_citations

    t0 = time.time()
    docs = fn(question, top_k, None)
    t1 = time.time()

    if not docs:
        return {
            "answer": "No relevant documents found. Cannot answer the question.",
            "sources": [],
            "unsupported": True,
            "latency_ms": round((t1 - t0) * 1000, 1),
            "retrieval_ms": round((t1 - t0) * 1000, 1),
            "generation_ms": 0.0,
            "collection": collection,
        }

    context = format_context(docs)
    answer = generate_citations(question, context, get_llm())
    t2 = time.time()

    return {
        "answer": answer,
        "sources": [
            {"id": d[0], "score": round(float(d[2]), 3), "snippet": d[1][:220]}
            for d in docs
        ],
        "unsupported": False,
        "latency_ms": round((t2 - t0) * 1000, 1),
        "retrieval_ms": round((t1 - t0) * 1000, 1),
        "generation_ms": round((t2 - t1) * 1000, 1),
        "collection": collection,
    }


# ─── UI ────────────────────────────────────────────────────────────────────

st.title("🔎 Production RAG")
st.caption("Hybrid search + LLM answer with citations — Qdrant Cloud, FastEmbed embeddings, cross-encoder-ready")

with st.sidebar:
    st.header("Settings")
    strategy = st.selectbox(
        "Search strategy",
        ["hybrid", "dense_only", "sparse_only"],
        index=0,
        help="hybrid = dense + sparse with RRF fusion",
    )
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    st.divider()
    st.markdown("**Examples**")
    for q in [
        "what color is amber urine",
        "how to reset my password?",
        "what causes dark amber urine",
        "what do elevated liver enzymes mean",
        "how does a bill become law in the united states",
    ]:
        if st.button(q, use_container_width=True):
            st.session_state["q"] = q

question = st.text_input(
    "Question",
    key="q",
    placeholder="e.g. what color is amber urine",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving and generating..."):
        try:
            result = run_query(question.strip(), strategy, top_k)
        except Exception as e:
            st.error(f"Error: {type(e).__name__}: {e}")
            st.stop()

    if result["unsupported"]:
        st.warning("**No answer in corpus**")
        st.write(result["answer"])
    else:
        st.markdown("### Answer")
        st.write(result["answer"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Latency", f"{result['latency_ms'] / 1000:.1f}s")
    c2.metric("Retrieval", f"{result['retrieval_ms'] / 1000:.1f}s")
    c3.metric("Generation", f"{result['generation_ms'] / 1000:.1f}s")

    with st.expander(f"Sources ({len(result['sources'])})", expanded=True):
        for i, s in enumerate(result["sources"], 1):
            st.markdown(f"**[{i}]** `{s['id']}` · score `{s['score']}`")
            st.text(s["snippet"])

    st.divider()
    st.caption(
        f"collection `{result['collection']}` · strategy `{strategy}` · top_k {top_k}"
    )