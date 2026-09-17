import re

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Simple keyword-based retriever, used standalone in Hybrid RAG."""

    def __init__(self, docs):
        self.docs = docs
        self.tokenized_corpus = [self._tokenize(d.page_content) for d in docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def _tokenize(text: str):
        return re.findall(r"\w+", text.lower())

    def retrieve(self, query: str, k: int = 5):
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in ranked_idx]
