from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, docs, top_k: int = 5):
    """Re-score a candidate list with a cross-encoder and keep the top_k."""
    if not docs:
        return docs
    reranker = get_reranker()
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:top_k]]
