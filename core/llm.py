from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL

_llm = None


def get_llm(temperature: float = 0.0):
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=temperature)
    return _llm


def rewrite_query(query: str) -> str:
    """Rewrite a possibly vague/conversational question into a sharper search query."""
    llm = get_llm()
    prompt = f"""You rewrite user questions into clearer, more specific search queries for a
retrieval system. Keep it to a single line. No explanation, no quotes, no preamble.

Original question: {query}
Rewritten query:"""
    response = llm.invoke(prompt)
    return response.content.strip()


def generate_multi_queries(query: str, n: int = 3) -> list[str]:
    """Generate n paraphrased variants of the question for multi-query retrieval."""
    llm = get_llm()
    prompt = f"""Generate {n} different rephrasings of the question below, each focusing on a
different angle or wording, useful for document retrieval. Return ONLY the {n}
questions, one per line, no numbering, no extra text.

Question: {query}"""
    response = llm.invoke(prompt)
    lines = [l.strip("-• ").strip() for l in response.content.strip().split("\n") if l.strip()]
    return lines[:n]


def generate_hypothetical_answer(query: str) -> str:
    """HyDE: generate a hypothetical passage that would answer the question, to embed
    in place of the raw query (hypothetical answers tend to be closer in embedding
    space to real answer-bearing passages than short questions are)."""
    llm = get_llm()
    prompt = f"""Write a short hypothetical passage (3-4 sentences) that would answer the
question below, as if it were an excerpt from a reference document. Do not mention
that it's hypothetical, and do not add any preamble.

Question: {query}"""
    response = llm.invoke(prompt)
    return response.content.strip()


def generate_answer(query: str, context_chunks: list[str]) -> str:
    llm = get_llm()
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""Answer the question using ONLY the context below. If the answer is not
contained in the context, say "I cannot find this in the provided document." Do not
use outside knowledge.

Context:
{context}

Question: {query}

Answer:"""
    response = llm.invoke(prompt)
    return response.content.strip()
