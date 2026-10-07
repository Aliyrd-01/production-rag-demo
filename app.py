"""
Production RAG — Streamlit demo.

Стек: Qdrant (hybrid dense+sparse, RRF) + FastEmbed + LLM-пул с цитатами.
Датасет: rag_v2_q_test (MS MARCO, 310 746 пассажей) — индекс не пересобирается.

Локально:  streamlit run app.py
Облако:   share.streamlit.io (секреты — в App Settings → Secrets)
Docker:   см. Dockerfile / tools/start-rag-public.cmd
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

import streamlit as st

st.set_page_config(page_title="Production RAG", page_icon="🔎", layout="centered")


# ─── Secrets → os.environ ───────────────────────────────────────────────────
def load_secrets_to_env() -> list:
    """Секреты из st.secrets -> os.environ (модули читают os.getenv()).

    ВАЖНО: обращаться к st.secrets можно только внутри try — у st.secrets нет
    дешёвого __bool__, проверка вне try роняет скрипт.
    """
    try:
        items = dict(st.secrets)
    except Exception:
        return []
    loaded = []
    for k, v in items.items():
        if isinstance(v, str) and v and not os.environ.get(k):
            os.environ[k] = v
            loaded.append(k)
    return loaded


load_secrets_to_env()


# ─── Backend ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Qdrant client...")
def get_collection_info():
    from module_5_retrieval import COLLECTION, get_client
    info = get_client().get_collection(COLLECTION)
    return COLLECTION, info.points_count


@st.cache_resource(show_spinner="Loading LLM pool...")
def get_llm():
    from common import get_chat_llm
    return get_chat_llm(temperature=0)


def _apply_meta_filter(docs, langs=(), sources=()):
    """Client-side metadata filter (the collection has no payload indexes)."""
    if not langs and not sources:
        return docs
    out = []
    for d in docs:
        pl = d[3] or {}
        if langs and pl.get("lang") not in langs:
            continue
        if sources and pl.get("source") not in sources:
            continue
        out.append(d)
    return out


def retrieve_docs(strategy, question, top_k, rerank_top, langs=(), sources=()):
    """Dispatch a strategy to the retrieval functions (llm fetched lazily)."""
    import module_5_retrieval as m

    llm = None
    if strategy in ("dense_hyde", "hybrid_hyde", "hybrid_hyde_cross_encoder", "multi_query"):
        llm = get_llm()

    if strategy == "dense_only":
        docs = m.search_dense(question, top_k, None)
    elif strategy == "sparse_only":
        docs = m.search_sparse(question, top_k, None)
    elif strategy == "hybrid":
        docs = m.search_hybrid(question, top_k, None)
    elif strategy == "hybrid_diverse":
        docs = m.search_hybrid_diverse(question, top_k, None)
    elif strategy == "hybrid_weighted":
        docs = m.search_hybrid_weighted(question, top_k, None)
    elif strategy == "hybrid_llm_rerank":
        docs = m.search_hybrid_rerank(question, top_k, None, rerank_top=rerank_top)
    elif strategy == "hybrid_cross_encoder":
        base = m.search_hybrid(question, max(top_k, 20), None)
        docs = m.rerank_cross_encoder(question, base, top_n=rerank_top)
    elif strategy == "dense_hyde":
        docs = m.search_dense_hyde(question, top_k, None, llm=llm)
    elif strategy == "hybrid_hyde":
        docs = m.search_hybrid_hyde(question, top_k, None, llm=llm)
    elif strategy == "hybrid_hyde_cross_encoder":
        docs = m.search_hybrid_hyde_ce(question, top_k, None, llm=llm, rerank_top=rerank_top)
    elif strategy == "multi_query":
        docs = m.search_multi_query(question, top_k, None, llm=llm)
    else:
        raise ValueError(strategy)

    return _apply_meta_filter(docs, langs, sources)


@st.cache_data(show_spinner=False, max_entries=256)
def cached_retrieve(strategy, question, top_k, rerank_top, langs=(), sources=()):
    return retrieve_docs(strategy, question, top_k, rerank_top, langs, sources)


def _run_retrieval(strategy, question, top_k, rerank_top, langs, sources):
    """Cached retrieval with a graceful fallback to hybrid if a strategy fails."""
    try:
        docs = cached_retrieve(strategy, question, top_k, rerank_top, langs, sources)
        return docs, None
    except Exception as e:
        if strategy == "hybrid":
            raise
        docs = cached_retrieve("hybrid", question, top_k, rerank_top, langs, sources)
        return docs, f"{type(e).__name__}: {e}"


def _log_feedback(question, answer, strategy, verdict):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "feedback.jsonl")
    rec = {"ts": time.time(), "question": question, "strategy": strategy,
           "verdict": verdict, "answer": (answer or "")[:2000]}
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ─── UI ─────────────────────────────────────────────────────────────────────
st.title("🔎 Production RAG")
st.caption("Hybrid retrieval (dense + sparse, RRF) → LLM answer with citations. "
           "Qdrant Cloud · FastEmbed · LangChain")

STRATEGIES = {
    # Первый = дефолт. hybrid влезает в минимальную RAM.
    "hybrid": "Hybrid — dense + sparse (RRF) + IDF  ← default",
    "hybrid_diverse": "Hybrid + MMR diversity",
    "hybrid_cross_encoder": "Hybrid + CrossEncoder rerank",
    "hybrid_hyde": "Hybrid + HyDE (hypothetical answer)",
    "hybrid_hyde_cross_encoder": "Hybrid + HyDE + CrossEncoder  (best quality)",
    "multi_query": "Multi-query (LLM rewrites) + hybrid",
    "dense_only": "Dense only (Nomic v1.5)",
    "sparse_only": "Sparse only (BM25 + IDF)",
    "hybrid_weighted": "Hybrid — weighted RRF",
    "hybrid_llm_rerank": "Hybrid + LLM rerank",
    "dense_hyde": "Dense + HyDE",
}
NEEDS_RERANK = ("hybrid_cross_encoder", "hybrid_hyde_cross_encoder",
                "hybrid_llm_rerank")

EXAMPLES = [
    "what color is amber urine",
    "what causes dark amber urine",
    "what do elevated liver enzymes mean",
    "how does a bill become law in the united states",
    "how much does an average person make for tutoring",
    "is autoimmune hepatitis a bile acid synthesis disorder",
]


def load_benchmarks() -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "results_retrieval.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        return {}
    latest = {}
    for run in history:
        for row in run.get("rows", []):
            latest[row["strategy"]] = {**row, "tag": run.get("tag", "")}
    return latest


def load_generation_benchmarks() -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "results_generation.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


with st.sidebar:
    st.header("Retrieval")
    strategy = st.selectbox("Strategy", list(STRATEGIES), index=0,
                            format_func=lambda k: STRATEGIES[k])
    top_k = st.slider("Candidates (top K)", 1, 20, 8)
    need_rerank = strategy in NEEDS_RERANK
    rerank_top = st.slider("After rerank", 1, 10, 5) if need_rerank else 5

    st.divider()
    st.markdown("**Answering**")
    min_score = st.slider("Min top score (0 = off)", 0.0, 1.0, 0.0, 0.05,
                          help="Refuse to answer when the best retrieved score is below this.")
    strict_grounding = st.checkbox("Strict grounding (flag uncited answers)", value=False)

    st.divider()
    st.markdown("**Metadata filter**")
    lang_filter = st.multiselect("Language", ["en"], default=[])
    source_filter = st.multiselect("Source", ["msmarco"], default=[])

    st.divider()
    st.markdown("**Corpus**")
    try:
        collection, points = get_collection_info()
    except Exception as e:  # noqa: BLE001
        collection, points = "rag_v2_q_test", 0
        st.error(f"Qdrant unavailable: {e}")
    st.code(collection, language=None)
    st.caption(f"MS MARCO · {points:,} passages · dense 768d (COSINE/INT8) + sparse BM25 + client-side IDF")

    st.divider()
    st.markdown("**Try**")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True, key=f"ex_{q}"):
            st.session_state["q"] = q

tab_chat, tab_bench, tab_gen = st.tabs(["Chat", "Retrieval benchmarks", "Generation benchmarks"])

with tab_chat:
    question = st.text_input("Question", key="q", placeholder="e.g. what color is amber urine")

    if st.button("Ask", type="primary", disabled=not question.strip()):
        q = question.strip()
        langs = tuple(lang_filter)
        sources = tuple(source_filter)

        t0 = time.time()
        try:
            with st.spinner("Retrieving..."):
                docs, warn = _run_retrieval(strategy, q, top_k, rerank_top, langs, sources)
        except Exception as e:  # noqa: BLE001
            st.error(f"**{type(e).__name__}:** {e}")
            docs, warn = None, None
        t1 = time.time()

        if docs is None:
            pass
        else:
            if warn:
                st.warning(f"`{strategy}` failed ({warn}) — fell back to **hybrid**.")

            top_score = max((float(d[2]) for d in docs), default=0.0)
            too_weak = (not docs) or (min_score > 0 and top_score < min_score)
            if too_weak:
                st.warning("**No answer in corpus** — retrieval too weak or empty. "
                           "The question may be out of the indexed corpus.")
            else:
                from module_7_generation import (
                    format_context, generate_citations_stream, sanitize_context,
                    extract_citations, verify_grounding, count_tokens,
                )

                llm = get_llm()
                context = sanitize_context(format_context(docs))

                st.markdown("### Answer")
                t_gen0 = time.time()
                ttft = {"v": None}

                def _gen():
                    for chunk in generate_citations_stream(q, context, llm):
                        if ttft["v"] is None:
                            ttft["v"] = time.time() - t_gen0
                        yield chunk

                try:
                    answer = st.write_stream(_gen())
                except Exception:
                    from module_7_generation import generate_citations
                    answer = generate_citations(q, context, llm)
                    st.write(answer)
                t2 = time.time()

                answer = answer or ""
                _retr_ms = (t1 - t0) * 1000
                _gen_ms = (t2 - t_gen0) * 1000

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total", f"{(t2 - t0):.1f}s")
                m2.metric("Retrieval", f"{_retr_ms / 1000:.1f}s")
                m3.metric("Generation", f"{_gen_ms / 1000:.1f}s")
                m4.metric("TTFT", f"{ttft['v']:.1f}s" if ttft["v"] else "—")

                from common import describe_llm
                info = describe_llm(llm)
                line = f"**LLM:** `{info['provider']}` · served `{info['served']}`"
                if info["served"] != info["requested"]:
                    line += f" (requested `{info['requested']}`)"
                st.caption(line)

                # Grounding: mark which sources were actually cited
                cited = {str(did) for _, did in extract_citations(answer)}
                issues, cited_ids, _ = verify_grounding(
                    answer, [str(d[0]) for d in docs], len(docs))
                with st.expander(f"Sources ({len(docs)})", expanded=True):
                    for i, d in enumerate(docs, 1):
                        mark = "  ✅ cited" if str(d[0]) in cited else ""
                        st.markdown(f"**[{i}]** `msmarco#{d[0]}` · score `{round(float(d[2]), 3)}`{mark}")
                        st.text(d[1][:400])
                if strict_grounding and not cited:
                    st.warning("Strict grounding: the answer cites no source — treat it as unverified.")
                elif issues:
                    st.caption(f"⚠️ citation issues: {issues}")

                # Cost estimate (Qubax qwen3-235b ballpark)
                tok_in = count_tokens(context + q)
                tok_out = count_tokens(answer)
                cost = tok_in * (0.004 / 1_000_000) + tok_out * (0.0162 / 1_000_000)
                st.caption(f"~{tok_in} in / ~{tok_out} out tokens · est. **${cost:.6f}** per query")

                f1, f2, _sp = st.columns([1, 1, 4])
                if f1.button("👍 Helpful", key="fb_up"):
                    _log_feedback(q, answer, strategy, "up")
                    st.toast("Thanks — logged.")
                if f2.button("👎 Not helpful", key="fb_down"):
                    _log_feedback(q, answer, strategy, "down")
                    st.toast("Thanks — logged.")

                st.divider()
                st.caption(f"collection `{collection}` · {points:,} passages · "
                           f"strategy `{strategy}` · top_k {top_k}")

    if question.strip():
        with st.expander("Run all strategies (A/B on this question)"):
            if st.button("Compare all", key="ab_all"):
                rows = []
                for name, label in STRATEGIES.items():
                    t0 = time.time()
                    try:
                        ds, _w = _run_retrieval(name, question.strip(), top_k, rerank_top, (), ())
                        dt = time.time() - t0
                        rows.append({"strategy": name, "docs": len(ds), "s": round(dt, 2),
                                     "top score": round(max((float(d[2]) for d in ds), default=0.0), 3),
                                     "top doc": (ds[0][0] if ds else "—")})
                    except Exception as e:  # noqa: BLE001
                        rows.append({"strategy": name, "docs": 0, "s": round(time.time() - t0, 2),
                                     "top score": 0.0, "top doc": f"err: {type(e).__name__}"})
                st.dataframe(rows, use_container_width=True, hide_index=True)

        with st.expander("Retrieval trace (dense vs sparse vs fused)"):
            if st.button("Trace", key="trace_btn"):
                try:
                    from module_5_retrieval import trace_hybrid
                    tr = trace_hybrid(question.strip(), k=top_k)
                    c1, c2, c3 = st.columns(3)
                    for col, key, title in ((c1, "dense", "Dense"), (c2, "sparse", "Sparse"), (c3, "fused", "Fused (RRF)")):
                        with col:
                            st.markdown(f"**{title}**")
                            for d in tr[key][:8]:
                                st.caption(f"`{d[0]}` · {round(float(d[2]), 3)}")
                                st.text((d[1] or "")[:120])
                except Exception as e:  # noqa: BLE001
                    st.error(f"Trace failed: {e}")

    st.caption(f"Strategies available: {len(STRATEGIES)} · measured on 100 MS MARCO "
               "questions — see the benchmarks tabs.")


# ─── Retrieval benchmarks ───────────────────────────────────────────────────
BENCH_ORDER = [
    "dense_only", "sparse_only", "hybrid", "hybrid_diverse",
    "dense_hyde", "hybrid_hyde", "hybrid_ce", "w0.0", "w0.25",
]
BENCH_LABELS = {
    "dense_only": "Dense only (Nomic v1.5)",
    "sparse_only": "Sparse only (BM25)",
    "hybrid": "Hybrid — dense + sparse (RRF)",
    "hybrid_diverse": "Hybrid + MMR diversity",
    "dense_hyde": "Dense + HyDE",
    "hybrid_hyde": "Hybrid + HyDE",
    "hybrid_ce": "Hybrid + CrossEncoder",
    "w0.0": "Weighted RRF (w_sparse=0.0)",
    "w0.25": "Weighted RRF (w_sparse=0.25)",
}


def render_benchmarks():
    st.subheader("Retrieval benchmarks")
    st.caption("Offline evaluation on 100 MS MARCO questions (qrels train.tsv). "
               "Documents are deduplicated by doc_id — the index stores chunks.")

    bench = load_benchmarks()
    if not bench:
        st.info("No results yet — run `python eval/eval_retrieval.py`.")
        return

    rows = []
    for key in BENCH_ORDER:
        if key not in bench:
            continue
        r = bench[key]
        rows.append({
            "Strategy": BENCH_LABELS.get(key, key),
            "Hit@5": r.get("hit@5"), "Hit@10": r.get("hit@10"),
            "MRR@10": r.get("mrr@10"), "NDCG@10": r.get("ndcg@10"),
            "avg s": r.get("avg_latency_s"), "p95 s": r.get("p95_latency_s"),
        })

    st.dataframe(rows, use_container_width=True, hide_index=True, column_config={
        "Hit@5": st.column_config.ProgressColumn("Hit@5", min_value=0.0, max_value=1.0, format="%.3f"),
        "Hit@10": st.column_config.NumberColumn("Hit@10", format="%.3f"),
        "MRR@10": st.column_config.NumberColumn("MRR@10", format="%.3f"),
        "NDCG@10": st.column_config.NumberColumn("NDCG@10", format="%.3f"),
        "avg s": st.column_config.NumberColumn("avg s", format="%.2f"),
        "p95 s": st.column_config.NumberColumn("p95 s", format="%.2f"),
    })

    if rows:
        best = max(rows, key=lambda r: (r["Hit@10"] or 0, r["NDCG@10"] or 0))
        st.success(f"**Best by Hit@10:** {best['Strategy']} — Hit@10 {best['Hit@10']:.3f}, "
                   f"NDCG@10 {best['NDCG@10']:.3f}")

    with st.expander("How to read this"):
        st.markdown(
            "- **Hit@k** — доля вопросов, где релевантный документ попал в топ-k.\n"
            "- **MRR@10** — средний обратный ранг первого релевантного документа.\n"
            "- **NDCG@10** — качество ранжирования целиком.\n"
            "- **avg / p95 s** — задержка retrieval на одном запросе.")


with tab_bench:
    render_benchmarks()


def render_generation():
    st.subheader("Generation benchmarks")
    st.caption("LLM-judged answer quality on the same eval questions "
               "(faithfulness / answer relevancy / citation validity).")
    gen = load_generation_benchmarks()
    if not gen:
        st.info("No results yet — run `python eval/eval_generation.py --limit 30`.")
        return
    rows = gen.get("rows", [])
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(f"judge: `{gen.get('judge', '?')}` · n = {gen.get('n', '?')} · "
               f"{gen.get('ts', '')}")


with tab_gen:
    render_generation()
