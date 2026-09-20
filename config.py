import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Used only for the final answer generation, where quality matters most.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Used for everything else: query rewriting, HyDE, multi-query, LLM-judge scoring,
# and eval-question generation. These are simpler tasks and a fast/cheap model does
# fine — this is the single biggest lever for reducing token spend, since the judge
# call alone roughly doubles context-token usage per pipeline-question pair.
GROQ_HELPER_MODEL = os.getenv("GROQ_HELPER_MODEL", "openai/gpt-oss-20b")

# --- Embeddings (local, free) ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Reranker (local, free) ---
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Retrieval ---
# Lower TOP_K_RETRIEVE and/or higher TOP_K_RERANK_CANDIDATES = harder retrieval task =
# bigger visible gap between naive and enhanced pipelines. Both are also exposed as
# sidebar sliders in the app, so these are just the defaults.
TOP_K_RETRIEVE = 4
TOP_K_RERANK_CANDIDATES = 30
