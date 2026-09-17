import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.document_loader import load_document, chunk_document
from core.vectorstore import build_vectorstore
from core.bm25_retriever import BM25Retriever
from core.pipelines import PIPELINE_REGISTRY
from core.metrics import run_pipeline_with_metrics

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
    uploaded = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])
    text = uploaded.read().decode("utf-8") if uploaded else None
else:
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
# Sidebar: pipeline selection
# ---------------------------------------------------------------------------
st.sidebar.header("2. Pipelines to compare")
pipeline_labels = {k: v.name for k, v in PIPELINE_REGISTRY.items()}
selected = st.sidebar.multiselect(
    "Choose strategies",
    options=list(pipeline_labels.keys()),
    default=["naive", "hybrid", "rerank"],
    format_func=lambda k: pipeline_labels[k],
)
for key in selected:
    st.sidebar.caption(f"**{pipeline_labels[key]}** — {PIPELINE_REGISTRY[key].description}")


def get_pipeline(key):
    cls = PIPELINE_REGISTRY[key]
    return cls(st.session_state.vectorstore, st.session_state.bm25)


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
# Main tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔍 Single Question", "📊 Batch Evaluation"])

with tab1:
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
                result = run_pipeline_with_metrics(pipeline, question)
            results.append(result)
            with cols[i]:
                st.subheader(pipeline_labels[key])
                st.write(result["answer"])
                st.caption(f"⏱ {result['latency_sec']}s · 🔁 {result['num_llm_calls']} LLM calls")
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
                }
                for r in results
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.plotly_chart(radar_chart(results), use_container_width=True)

with tab2:
    st.write(
        "Run every selected pipeline over a fixed evaluation question set "
        "(`data/eval_questions.json`) and compare aggregate scores."
    )
    eval_path = os.path.join("data", "eval_questions.json")
    eval_questions = []
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_questions = json.load(f)

    st.caption(f"{len(eval_questions)} evaluation questions loaded.")
    with st.expander("View evaluation questions"):
        for q in eval_questions:
            st.markdown(f"- {q['question']}")

    batch_btn = st.button("Run batch evaluation")

    if batch_btn and not selected:
        st.warning("Select at least one pipeline in the sidebar.")
    elif batch_btn:
        all_results = []
        progress = st.progress(0.0)
        total = max(len(selected) * len(eval_questions), 1)
        done = 0
        for key in selected:
            pipeline = get_pipeline(key)
            for q in eval_questions:
                result = run_pipeline_with_metrics(pipeline, q["question"])
                all_results.append(result)
                done += 1
                progress.progress(done / total)

        df = pd.DataFrame(all_results)
        agg = (
            df.groupby("pipeline")
            .agg(
                faithfulness=("faithfulness", "mean"),
                answer_relevance=("answer_relevance", "mean"),
                context_relevance=("context_relevance", "mean"),
                latency_sec=("latency_sec", "mean"),
                num_llm_calls=("num_llm_calls", "mean"),
            )
            .round(2)
            .reset_index()
        )

        st.subheader("Aggregate results")
        st.dataframe(agg, use_container_width=True, hide_index=True)
        st.plotly_chart(radar_chart(agg.to_dict("records")), use_container_width=True)

        with st.expander("Per-question raw results"):
            st.dataframe(
                df[["pipeline", "question", "faithfulness", "answer_relevance", "context_relevance", "latency_sec"]],
                use_container_width=True,
                hide_index=True,
            )
