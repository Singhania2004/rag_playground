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
actually show up. 

## Cost control

Two Groq models are used, not one:

- **`GROQ_MODEL`** (default `openai/gpt-oss-120b`) — only for final answer generation, where quality matters most.
- **`GROQ_HELPER_MODEL`** (default `openai/gpt-oss-20b`) — for everything else: query rewriting, HyDE, multi-query generation, LLM-judge scoring, and eval-question generation. These are simple tasks; a small fast model does fine, and the judge call alone roughly doubles context-token usage per pipeline-question pair, so this split matters more than anything else for cost.

Every result reports `pipeline_tokens` (what the technique would actually cost in production) separately from `judge_tokens` (evaluation overhead), plus a total. There's also a **⚡ Fast mode** toggle in the sidebar's advanced settings that skips the LLM judge entirely and only computes retrieval metrics — useful while you're tuning top-k/candidate-pool settings, before spending tokens on a final scored run.

## Tuning the gap between pipelines

Sidebar → **⚙️ Advanced retrieval settings** exposes:
- **Top-k chunks retrieved** — lower = harder retrieval task = bigger visible gap.
- **Candidate pool size** (for reranker / hybrid+rerank) — higher = reranker has more to work with, especially on multi-hop questions where a needed chunk might rank outside the top-k on dense similarity alone but still be recoverable further down.

After a batch run, the category breakdown table flags any category where every pipeline scored about the same (near-zero spread) — that category isn't teaching you anything that run, so consider weighting the eval-set generation counts away from it next time

