import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import TOP_K_RETRIEVE, TOP_K_RERANK_CANDIDATES
from core.document_loader import load_document, chunk_document
from core.vectorstore import build_vectorstore
from core.bm25_retriever import BM25Retriever
from core.pipelines import PIPELINE_REGISTRY
from core.metrics import run_pipeline_with_metrics
from core.eval_generator import generate_eval_set
from core.llm import reset_token_tracker, get_token_usage

st.set_page_config(page_title="RAG Lab", layout="wide")
st.title("🧪 RAG Lab — Compare Retrieval Strategies")
st.caption(
    "Same document, same question, different pipelines. See what each technique actually buys you."
)

# ---------------------------------------------------------------------------
# Sidebar: document setup
# ---------------------------------------------------------------------------
st.sidebar.header("1. Document")
doc_source = st.sidebar.radio("Source", ["Use sample document", "Upload your own (.txt)"])

if doc_source == "Upload your own (.txt)":
    st.sidebar.caption(
        "Any parsed plain-text .txt"
    )
    uploaded = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])
    text = uploaded.read().decode("utf-8") if uploaded else None
else:
    st.sidebar.caption("Default sample document: *The Metamorphosis* by Franz Kafka.")
    text = load_document(os.path.join("data", "sample_doc.txt"))

if text is None:
    st.info("Upload a .txt file in the sidebar to get started.")
    st.stop()

doc_signature = str(hash(text))

if "doc_signature" not in st.session_state or st.session_state.doc_signature != doc_signature:
    with st.spinner(
        "Chunking document and building indexes "
        "(first run downloads the embedding + reranker models, ~1-2 min)..."
    ):
        docs = chunk_document(text)
        vectorstore = build_vectorstore(docs)
        bm25 = BM25Retriever(docs)
    st.session_state.doc_signature = doc_signature
    st.session_state.docs = docs
    st.session_state.vectorstore = vectorstore
    st.session_state.bm25 = bm25

st.sidebar.success(f"Indexed {len(st.session_state.docs)} chunks")

# ---------------------------------------------------------------------------
# Sidebar: advanced retrieval settings — tune these to widen the gap between
# naive and enhanced pipelines. Lower top_k / higher candidate pool = harder.
# ---------------------------------------------------------------------------
with st.sidebar.expander("⚙️ Advanced retrieval settings"):
    top_k_retrieve = st.slider("Top-k chunks retrieved", 2, 10, TOP_K_RETRIEVE)
    top_k_candidates = st.slider(
        "Candidate pool size (reranker / hybrid+rerank)", 10, 60, TOP_K_RERANK_CANDIDATES
    )
    st.caption(
        "Lower top-k and/or a wider candidate pool = harder retrieval task = "
        "bigger visible gap between naive and enhanced pipelines."
    )
    fast_mode = st.checkbox(
        "⚡ Fast mode — skip LLM-judge scoring", value=False,
        help="Skips the faithfulness/relevance LLM calls entirely. Only retrieval "
             "metrics (recall/precision/MRR) are computed. Roughly halves token spend "
             "and latency — good for iterating on retrieval settings before a final run.",
    )

# ---------------------------------------------------------------------------
# Sidebar: pipeline selection
# ---------------------------------------------------------------------------
st.sidebar.header("2. Pipelines to compare")
pipeline_labels = {k: v.name for k, v in PIPELINE_REGISTRY.items()}
selected = st.sidebar.multiselect(
    "Choose strategies",
    options=list(pipeline_labels.keys()),
    default=["naive", "rerank", "hybrid_rerank"],
    format_func=lambda k: pipeline_labels[k],
)
for key in selected:
    st.sidebar.caption(f"**{pipeline_labels[key]}** — {PIPELINE_REGISTRY[key].description}")


def get_pipeline(key):
    cls = PIPELINE_REGISTRY[key]
    return cls(
        st.session_state.vectorstore,
        st.session_state.bm25,
        top_k=top_k_retrieve,
        top_k_candidates=top_k_candidates,
    )


def radar_chart(rows, name_key="pipeline"):
    fig = go.Figure()
    for row in rows:
        fig.add_trace(
            go.Scatterpolar(
                r=[
                    row.get("faithfulness") or 0,
                    row.get("answer_relevance") or 0,
                    row.get("context_relevance") or 0,
                ],
                theta=["Faithfulness", "Answer Relevance", "Context Relevance"],
                fill="toself",
                name=row[name_key],
            )
        )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True)
    return fig


# ---------------------------------------------------------------------------
# Main tabs — Batch Evaluation listed first so it's the default selected tab.
# ---------------------------------------------------------------------------
tab_batch, tab_single = st.tabs(["📊 Batch Evaluation", "🔍 Single Question"])

with tab_batch:
    st.write(
        "Generate a ground-truth evaluation set directly from your loaded document "
        "(default: *The Metamorphosis* by Franz Kafka, or your own upload), then run "
        "every selected pipeline against it and compare real retrieval metrics "
        "(Recall, Precision, MRR) alongside the LLM-judged scores."
    )

    st.markdown("##### Step 1 — Generate the evaluation set")
    st.caption(
        "Four question categories are generated on purpose, so no single pipeline is "
        "favored by construction: **lexical** (keyword overlap → favors hybrid/BM25), "
        "**paraphrased** (different wording, same meaning → favors dense/rerank), "
        "**vague** (short, underspecified → favors query rewrite/HyDE), and "
        "**multi-hop** (needs two chunks combined → favors hybrid/multi-query/rerank). "
        "Generation uses the cheap helper model, but still costs 1 call per question."
    )

    c1, c2, c3, c4 = st.columns(4)
    n_lexical = c1.number_input("Lexical", min_value=0, max_value=15, value=3)
    n_paraphrased = c2.number_input("Paraphrased", min_value=0, max_value=15, value=3)
    n_vague = c3.number_input("Vague", min_value=0, max_value=15, value=3)
    n_multi_hop = c4.number_input("Multi-hop", min_value=0, max_value=15, value=3)
    st.caption(
        "Lexical questions tend to score 1.0 recall for every pipeline — they don't "
        "discriminate much. Weighting toward vague/multi-hop shows a bigger spread."
    )

    gen_btn = st.button("Generate evaluation set from this document")

    if gen_btn:
        total_q = n_lexical + n_paraphrased + n_vague + n_multi_hop
        if total_q == 0:
            st.warning("Set at least one question count above 0.")
        else:
            progress = st.progress(0.0)
            reset_token_tracker()
            with st.spinner("Generating questions from document chunks via LLM..."):
                eval_set = generate_eval_set(
                    st.session_state.docs,
                    n_lexical=n_lexical,
                    n_paraphrased=n_paraphrased,
                    n_vague=n_vague,
                    n_multi_hop=n_multi_hop,
                    progress_callback=progress.progress,
                )
            gen_tokens = get_token_usage()["total_tokens"]
            st.session_state.eval_set = eval_set
            st.session_state.eval_set_doc_signature = doc_signature
            st.success(f"Generated {len(eval_set)} questions using ~{gen_tokens} tokens.")

    eval_set = st.session_state.get("eval_set")
    stale = st.session_state.get("eval_set_doc_signature") != doc_signature
    if eval_set and stale:
        st.warning("The document changed since this eval set was generated — regenerate for accurate ground truth.")

    if eval_set:
        with st.expander(f"View generated eval set ({len(eval_set)} questions)"):
            for q in eval_set:
                st.markdown(f"- `[{q['category']}]` {q['question']}  \n  *expected answer: {q['answer']}*")
        st.download_button(
            "Download eval set as JSON",
            data=json.dumps(eval_set, indent=2),
            file_name="eval_set.json",
            mime="application/json",
        )

    st.markdown("##### Step 2 — Run the comparison")
    batch_btn = st.button("Run batch evaluation", type="primary")

    if batch_btn and not selected:
        st.warning("Select at least one pipeline in the sidebar.")
    elif batch_btn and not eval_set:
        st.warning("Generate an evaluation set first (Step 1).")
    elif batch_btn:
        all_results = []
        progress = st.progress(0.0)
        total = max(len(selected) * len(eval_set), 1)
        done = 0
        for key in selected:
            pipeline = get_pipeline(key)
            for q in eval_set:
                result = run_pipeline_with_metrics(
                    pipeline,
                    q["question"],
                    expected_chunk_ids=q["expected_chunk_ids"],
                    category=q["category"],
                    skip_judge=fast_mode,
                )
                all_results.append(result)
                done += 1
                progress.progress(done / total)

        df = pd.DataFrame(all_results)

        st.subheader("Overall aggregate (across all categories)")
        agg = (
            df.groupby("pipeline")
            .agg(
                recall_at_k=("recall_at_k", "mean"),
                precision_at_k=("precision_at_k", "mean"),
                mrr=("mrr", "mean"),
                faithfulness=("faithfulness", "mean"),
                answer_relevance=("answer_relevance", "mean"),
                context_relevance=("context_relevance", "mean"),
                latency_sec=("latency_sec", "mean"),
                total_tokens=("total_tokens", "sum"),
            )
            .round(2)
            .reset_index()
        )
        st.dataframe(agg, use_container_width=True, hide_index=True)
        st.plotly_chart(radar_chart(agg.to_dict("records")), use_container_width=True)

        st.subheader("Breakdown by question category — this is where pipelines diverge")
        st.caption(
            "Recall@k per pipeline per category. Look for which pipeline wins on which "
            "category rather than a single overall 'best' — that's usually the more honest answer."
        )
        pivot = df.pivot_table(
            index="pipeline", columns="category", values="recall_at_k", aggfunc="mean"
        ).round(2)
        st.dataframe(pivot, use_container_width=True)

        low_signal = pivot.std(axis=0)[pivot.std(axis=0) < 0.05].index.tolist()
        if low_signal:
            st.caption(
                f"⚠️ Little to no spread across pipelines this run in: **{', '.join(low_signal)}**. "
                "Those categories aren't discriminating right now — consider generating fewer of "
                "them next time, or lowering top-k to make retrieval harder."
            )

        with st.expander("Per-question raw results"):
            st.dataframe(
                df[[
                    "pipeline", "category", "question",
                    "recall_at_k", "precision_at_k", "mrr",
                    "faithfulness", "answer_relevance", "context_relevance",
                    "latency_sec", "total_tokens",
                ]],
                use_container_width=True,
                hide_index=True,
            )

with tab_single:
    st.caption(
        "Using the sample document (*The Metamorphosis* by Franz Kafka) unless you "
        "uploaded your own in the sidebar."
    )
    question = st.text_input("Ask a question about the document")
    run_btn = st.button("Run comparison", type="primary")

    if run_btn and not selected:
        st.warning("Select at least one pipeline in the sidebar.")
    elif run_btn and question:
        results = []
        cols = st.columns(len(selected))
        for i, key in enumerate(selected):
            pipeline = get_pipeline(key)
            with st.spinner(f"Running {pipeline_labels[key]}..."):
                result = run_pipeline_with_metrics(pipeline, question, skip_judge=fast_mode)
            results.append(result)
            with cols[i]:
                st.subheader(pipeline_labels[key])
                st.write(result["answer"])
                st.caption(
                    f"⏱ {result['latency_sec']}s · 🔁 {result['num_llm_calls']} LLM calls · "
                    f"🔢 {result['total_tokens']} tokens ({result['pipeline_tokens']} pipeline + {result['judge_tokens']} judge)"
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Faithful.", result.get("faithfulness"))
                m2.metric("Relevance", result.get("answer_relevance"))
                m3.metric("Ctx Rel.", result.get("context_relevance"))
                if result["extras"]:
                    with st.expander("Intermediate steps"):
                        st.json(result["extras"])
                with st.expander("Retrieved chunks"):
                    for c in result["chunks"]:
                        st.markdown(f"> {c.page_content[:350]}...")

        st.divider()
        st.subheader("Comparison")
        df = pd.DataFrame(
            [
                {
                    "Pipeline": r["pipeline"],
                    "Faithfulness": r.get("faithfulness"),
                    "Answer Relevance": r.get("answer_relevance"),
                    "Context Relevance": r.get("context_relevance"),
                    "Latency (s)": r["latency_sec"],
                    "LLM Calls": r["num_llm_calls"],
                    "Tokens": r["total_tokens"],
                }
                for r in results
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.plotly_chart(radar_chart(results), use_container_width=True)
