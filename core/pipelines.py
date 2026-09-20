from core.llm import (
    rewrite_query,
    generate_answer,
    generate_hypothetical_answer,
    generate_multi_queries,
)
from core.reranker import rerank
from config import TOP_K_RETRIEVE, TOP_K_RERANK_CANDIDATES


def reciprocal_rank_fusion(ranked_lists, k: int = 60):
    """Merge multiple ranked lists of Documents into one ranked list.
    Standard RRF: score(d) = sum(1 / (k + rank_in_list)) across lists.
    """
    scores = {}
    doc_map = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = doc.page_content
            doc_map[key] = doc
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


class BasePipeline:
    name = "base"
    description = ""

    def __init__(self, vectorstore, bm25_retriever=None, top_k=TOP_K_RETRIEVE, top_k_candidates=TOP_K_RERANK_CANDIDATES):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
        self.top_k_candidates = top_k_candidates

    def dense_retrieve(self, query, k=None):
        k = k or self.top_k
        return self.vectorstore.similarity_search(query, k=k)

    def run(self, question: str) -> dict:
        """Must return {"chunks": [...], "answer": str, "num_llm_calls": int, ...extras}"""
        raise NotImplementedError


class NaiveRAG(BasePipeline):
    name = "Naive RAG"
    description = "Embed the raw query, take top-k chunks, stuff into the prompt. The baseline."

    def run(self, question):
        chunks = self.dense_retrieve(question)
        answer = generate_answer(question, [c.page_content for c in chunks])
        return {"chunks": chunks, "answer": answer, "num_llm_calls": 1}


class QueryRewriteRAG(BasePipeline):
    name = "Query Rewriter + RAG"
    description = "LLM rewrites the question into a sharper search query before retrieval."

    def run(self, question):
        rewritten = rewrite_query(question)
        chunks = self.dense_retrieve(rewritten)
        answer = generate_answer(question, [c.page_content for c in chunks])
        return {
            "chunks": chunks,
            "answer": answer,
            "num_llm_calls": 2,
            "rewritten_query": rewritten,
        }


class HybridRAG(BasePipeline):
    name = "Hybrid RAG (BM25 + Dense)"
    description = "Combine keyword search (BM25) and dense embedding search via reciprocal rank fusion."

    def run(self, question):
        dense_chunks = self.dense_retrieve(question)
        bm25_chunks = self.bm25_retriever.retrieve(question, k=self.top_k)
        merged = reciprocal_rank_fusion([dense_chunks, bm25_chunks])[: self.top_k]
        answer = generate_answer(question, [c.page_content for c in merged])
        return {"chunks": merged, "answer": answer, "num_llm_calls": 1}


class RerankRAG(BasePipeline):
    name = "RAG + Reranker"
    description = "Retrieve a wide candidate set, then re-score with a cross-encoder for precision."

    def run(self, question):
        candidates = self.dense_retrieve(question, k=self.top_k_candidates)
        reranked = rerank(question, candidates, top_k=self.top_k)
        answer = generate_answer(question, [c.page_content for c in reranked])
        return {"chunks": reranked, "answer": answer, "num_llm_calls": 1}


class HyDERAG(BasePipeline):
    name = "HyDE RAG"
    description = "Generate a hypothetical answer first, then retrieve using ITS embedding instead of the question's."

    def run(self, question):
        hypothetical = generate_hypothetical_answer(question)
        chunks = self.dense_retrieve(hypothetical)
        answer = generate_answer(question, [c.page_content for c in chunks])
        return {
            "chunks": chunks,
            "answer": answer,
            "num_llm_calls": 2,
            "hypothetical_doc": hypothetical,
        }


class MultiQueryRAG(BasePipeline):
    name = "Multi-Query RAG"
    description = "Generate several paraphrased queries, retrieve for each, and merge/dedupe results."

    def run(self, question):
        queries = generate_multi_queries(question, n=3)
        all_chunks = []
        seen = set()
        for q in queries + [question]:
            for c in self.dense_retrieve(q):
                if c.page_content not in seen:
                    seen.add(c.page_content)
                    all_chunks.append(c)
        top_chunks = all_chunks[: self.top_k]
        answer = generate_answer(question, [c.page_content for c in top_chunks])
        return {
            "chunks": top_chunks,
            "answer": answer,
            "num_llm_calls": 2,
            "generated_queries": queries,
        }


class HybridRerankRAG(BasePipeline):
    name = "Hybrid + Reranker (Full Stack)"
    description = "BM25 + dense fusion for wide recall, then cross-encoder reranking for precision."

    def run(self, question):
        dense_chunks = self.dense_retrieve(question, k=self.top_k_candidates)
        bm25_chunks = self.bm25_retriever.retrieve(question, k=self.top_k_candidates)
        merged = reciprocal_rank_fusion([dense_chunks, bm25_chunks])
        reranked = rerank(question, merged, top_k=self.top_k)
        answer = generate_answer(question, [c.page_content for c in reranked])
        return {"chunks": reranked, "answer": answer, "num_llm_calls": 1}


PIPELINE_REGISTRY = {
    "naive": NaiveRAG,
    "query_rewrite": QueryRewriteRAG,
    "hybrid": HybridRAG,
    "rerank": RerankRAG,
    "hyde": HyDERAG,
    "multi_query": MultiQueryRAG,
    "hybrid_rerank": HybridRerankRAG,
}
