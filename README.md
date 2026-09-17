# 🧪 RAG Lab

A side-by-side laboratory for comparing RAG (Retrieval-Augmented Generation) techniques
on the same document and the same question: Naive RAG, Query Rewriting, Hybrid Search
(BM25 + dense), Cross-Encoder Reranking, HyDE, Multi-Query, and a "full stack" pipeline
that combines several of these — plus an LLM-judged metrics panel (faithfulness, answer
relevance, context relevance) and latency/cost tracking for each.

100% free stack: local embeddings (sentence-transformers), local reranker
(cross-encoder), Chroma as an in-memory vector store, and Groq for LLM calls
(free tier).

## Setup

```bash
cd rag_lab
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste in a free Groq API key from https://console.groq.com/keys

streamlit run app.py
```

First run will download the embedding model (~90MB) and reranker model (~90MB) from
HuggingFace — that's the "1-2 min" spinner you'll see. After that it's cached locally.

## How it's organized

```
rag_lab/
├── app.py                  # Streamlit UI (single question + batch eval tabs)
├── config.py                # model names, chunk size, top-k, etc.
├── data/
│   └── sample_doc.txt         # short sample document (swap for your own long doc)
└── core/
    ├── document_loader.py     # chunking
    ├── vectorstore.py         # Chroma + local embeddings
    ├── bm25_retriever.py      # keyword search for hybrid retrieval
    ├── reranker.py             # cross-encoder reranking
    ├── llm.py                  # Groq wrapper: rewrite / HyDE / multi-query / generate
    ├── pipelines.py             # the actual RAG variants — start here
    ├── eval_generator.py        # auto-generates ground-truth eval questions per chunk
    └── metrics.py                # LLM-judge scoring + retrieval metrics + latency
```

**`core/pipelines.py`** is the most important file — every technique is a small class
implementing `run(question) -> {chunks, answer, num_llm_calls, ...}`, so it's easy to
read what each variant actually does differently, and easy to add a new one.

## Pipelines included

| Key | What it does |
|---|---|
| `naive` | Baseline: embed query, top-k dense retrieval, generate. |
| `query_rewrite` | LLM rewrites the question into a sharper search query before retrieval. |
| `hybrid` | BM25 (keyword) + dense embeddings, merged via reciprocal rank fusion. |
| `rerank` | Retrieve a wide candidate set, then re-score with a cross-encoder. |
| `hyde` | Generate a hypothetical answer, embed *that* for retrieval instead of the question. |
| `multi_query` | Generate 3 paraphrased queries, retrieve for each, merge & dedupe. |
| `hybrid_rerank` | Hybrid retrieval + reranking together — the "kitchen sink" pipeline. |

## Metrics

**LLM-judged (every run, single question or batch):**
- **Faithfulness** — is the answer actually supported by the retrieved context (no hallucination)?
- **Answer relevance** — does the answer address the question that was asked?
- **Context relevance** — is what got retrieved actually relevant, or mostly noise?

**Ground-truth retrieval metrics (batch mode only, via the generated eval set):**
- **Recall@k** — of the chunk(s) a question was actually generated from, how many did retrieval find?
- **Precision@k** — of what got retrieved, how much was actually relevant?
- **MRR** — how high up the ranking was the first relevant chunk?

### Generating an unbiased eval set (Batch Evaluation tab)

Rather than hand-writing eval questions (easy to accidentally bias toward whichever
pipeline you like), the Batch Evaluation tab generates them straight from your loaded
document's chunks via the LLM, across four deliberately different categories so no
single pipeline wins by construction:

| Category | Tests | Should favor |
|---|---|---|
| `lexical` | question reuses the chunk's own wording | Hybrid / BM25 |
| `paraphrased` | question uses different vocabulary, same meaning | Dense / Reranker |
| `vague` | short, underspecified, conversational phrasing | Query Rewrite / HyDE |
| `multi_hop` | needs two separate chunks combined to answer | Hybrid / Multi-Query / Reranker |

Because each generated question records exactly which `chunk_id`(s) it came from, batch
results include real Recall/Precision/MRR, not just LLM-judged scores — and the
category breakdown table shows which pipeline wins on which *kind* of difficulty,
instead of collapsing everything into one "best overall" number, which is usually the
more honest way to present RAG comparisons.

Use a longer, denser document for this — the more plausible-looking distractor chunks
in the corpus, the more naive RAG's weaknesses (especially on `multi_hop` and `vague`)
actually show up. `data/sample_doc.txt` is intentionally short; swap in your own via
the sidebar's "Upload your own" option for a real stress test.

## Extending it

- **Add a new pipeline**: subclass `BasePipeline` in `core/pipelines.py`, implement
  `run()`, add it to `PIPELINE_REGISTRY`. It'll automatically show up in the sidebar.
- **Add retrieval-only metrics** (recall@k, MRR): you'll need labeled "relevant chunk
  ids" per question — add an `expected_chunk_ids` field to `eval_questions.json` and a
  new function in `core/metrics.py`.
- **Swap in RAGAS**: for a more rigorous eval suite, `pip install ragas` and replace
  `judge_response()` in `core/metrics.py` with RAGAS's `faithfulness` /
  `answer_relevancy` / `context_precision` metrics — they use the same LLM-judge idea
  but are more battle-tested.
- **Try a different document**: use the "Upload your own" option in the sidebar, or
  swap out `data/sample_doc.txt`. Longer, more repetitive/technical documents tend to
  show bigger gaps between naive RAG and the enhanced pipelines.
