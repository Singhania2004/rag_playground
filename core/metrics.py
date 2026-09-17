import json
import time

from core.llm import get_llm

JUDGE_PROMPT = """You are evaluating a RAG (Retrieval-Augmented Generation) system's output.
Be strict and objective.

Question: {question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Rate the following on a scale of 1-5 (integers only):
1. faithfulness: Is the answer fully supported by the retrieved context, with no
   hallucinated or unsupported facts?
2. answer_relevance: Does the answer actually address the question that was asked?
3. context_relevance: Is the retrieved context relevant to the question, as opposed
   to being mostly noise/off-topic?

Respond with ONLY valid JSON in this exact format, no other text, no markdown fences:
{{"faithfulness": <int>, "answer_relevance": <int>, "context_relevance": <int>}}"""


def judge_response(question: str, context_chunks: list[str], answer: str) -> dict:
    llm = get_llm()
    context = "\n\n".join(context_chunks) if context_chunks else "(no context retrieved)"
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    response = llm.invoke(prompt)
    text = response.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        scores = {"faithfulness": None, "answer_relevance": None, "context_relevance": None}
    return scores


def compute_retrieval_metrics(retrieved_docs, expected_chunk_ids) -> dict:
    """Ground-truth retrieval metrics. Compares each pipeline's retrieved chunk_ids
    against the chunk_id(s) an eval question was actually generated from.

    - recall_at_k: fraction of the expected chunks that were retrieved at all
    - precision_at_k: fraction of retrieved chunks that were actually relevant
    - mrr: reciprocal rank of the first relevant chunk retrieved (0 if none found)
    """
    if not expected_chunk_ids:
        return {"recall_at_k": None, "precision_at_k": None, "mrr": None}

    retrieved_ids = [d.metadata.get("chunk_id") for d in retrieved_docs]
    expected_set = set(expected_chunk_ids)
    hits = set(retrieved_ids) & expected_set

    recall = len(hits) / len(expected_set)
    precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0

    mrr = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in expected_set:
            mrr = 1.0 / rank
            break

    return {
        "recall_at_k": round(recall, 2),
        "precision_at_k": round(precision, 2),
        "mrr": round(mrr, 2),
    }


def run_pipeline_with_metrics(pipeline, question: str, expected_chunk_ids=None, category=None) -> dict:
    """Run one pipeline on one question and attach latency, LLM-judged quality scores,
    and (if ground-truth chunk ids are available) real retrieval metrics.
    """
    start = time.time()
    result = pipeline.run(question)
    latency = time.time() - start

    judge_scores = judge_response(
        question, [c.page_content for c in result["chunks"]], result["answer"]
    )
    retrieval_scores = compute_retrieval_metrics(result["chunks"], expected_chunk_ids)

    return {
        "pipeline": pipeline.name,
        "question": question,
        "category": category,
        "answer": result["answer"],
        "chunks": result["chunks"],
        "extras": {k: v for k, v in result.items() if k not in ("chunks", "answer", "num_llm_calls")},
        "latency_sec": round(latency, 2),
        "num_llm_calls": result.get("num_llm_calls", 1),
        **judge_scores,
        **retrieval_scores,
    }
