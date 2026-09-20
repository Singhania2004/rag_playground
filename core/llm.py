from contextvars import ContextVar

from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_HELPER_MODEL

_llm_cache = {}


def get_llm(temperature: float = 0.0, use_helper_model: bool = False):
    """Cache one ChatGroq instance per (model, temperature) combo.

    use_helper_model=True routes to the small/fast/cheap model, used for every task
    that isn't final answer generation (rewriting, HyDE, multi-query, judging, eval
    question generation).
    """
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    model = GROQ_HELPER_MODEL if use_helper_model else GROQ_MODEL
    key = (model, temperature)
    if key not in _llm_cache:
        _llm_cache[key] = ChatGroq(api_key=GROQ_API_KEY, model=model, temperature=temperature)
    return _llm_cache[key]


# ---------------------------------------------------------------------------
# Token tracking. ContextVar rather than a plain module-level dict so that
# concurrent Streamlit sessions (each run in their own thread) don't clobber
# each other's counters.
# ---------------------------------------------------------------------------
_token_usage_var: ContextVar = ContextVar("_token_usage_var", default=None)


def reset_token_tracker():
    _token_usage_var.set({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


def get_token_usage() -> dict:
    return _token_usage_var.get() or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _track_usage(response):
    tracker = _token_usage_var.get()
    if tracker is None:
        return
    # Newer langchain AIMessage objects expose usage_metadata directly.
    usage = getattr(response, "usage_metadata", None)
    if usage:
        tracker["prompt_tokens"] += usage.get("input_tokens", 0) or 0
        tracker["completion_tokens"] += usage.get("output_tokens", 0) or 0
        tracker["total_tokens"] += usage.get("total_tokens", 0) or 0
        return
    # Fallback: Groq's raw usage block surfaced via response_metadata.
    meta = getattr(response, "response_metadata", {}) or {}
    token_usage = meta.get("token_usage", {}) or {}
    tracker["prompt_tokens"] += token_usage.get("prompt_tokens", 0) or 0
    tracker["completion_tokens"] += token_usage.get("completion_tokens", 0) or 0
    tracker["total_tokens"] += token_usage.get("total_tokens", 0) or 0


def invoke_llm(prompt: str, use_helper_model: bool = False, temperature: float = 0.0) -> str:
    """Single entry point for every LLM call in the app, so token tracking is
    never accidentally skipped."""
    llm = get_llm(temperature=temperature, use_helper_model=use_helper_model)
    response = llm.invoke(prompt)
    _track_usage(response)
    return response.content.strip()


# ---------------------------------------------------------------------------
# Task-specific prompts. Everything except generate_answer uses the helper model.
# ---------------------------------------------------------------------------

def rewrite_query(query: str) -> str:
    """Rewrite a possibly vague/conversational question into a sharper search query."""
    prompt = f"""You rewrite user questions into clearer, more specific search queries for a
retrieval system. Keep it to a single line. No explanation, no quotes, no preamble.

Original question: {query}
Rewritten query:"""
    return invoke_llm(prompt, use_helper_model=True)


def generate_multi_queries(query: str, n: int = 3) -> list[str]:
    """Generate n paraphrased variants of the question for multi-query retrieval."""
    prompt = f"""Generate {n} different rephrasings of the question below, each focusing on a
different angle or wording, useful for document retrieval. Return ONLY the {n}
questions, one per line, no numbering, no extra text.

Question: {query}"""
    text = invoke_llm(prompt, use_helper_model=True)
    lines = [l.strip("-• ").strip() for l in text.split("\n") if l.strip()]
    return lines[:n]


def generate_hypothetical_answer(query: str) -> str:
    """HyDE: generate a hypothetical passage that would answer the question, to embed
    in place of the raw query."""
    prompt = f"""Write a short hypothetical passage (3-4 sentences) that would answer the
question below, as if it were an excerpt from a reference document. Do not mention
that it's hypothetical, and do not add any preamble.

Question: {query}"""
    return invoke_llm(prompt, use_helper_model=True)


def generate_answer(query: str, context_chunks: list[str]) -> str:
    """Final answer generation — the one call that stays on the main, higher-quality model."""
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""Answer the question using ONLY the context below. If the answer is not
contained in the context, say "I cannot find this in the provided document." Do not
use outside knowledge.

Context:
{context}

Question: {query}

Answer:"""
    return invoke_llm(prompt, use_helper_model=False)
