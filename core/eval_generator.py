import json
import random

from core.llm import get_llm

MIN_CHUNK_LEN = 200  # skip chunks too short/uninformative to write a real question from

SINGLE_HOP_PROMPTS = {
    "lexical": """From the passage below, write ONE specific factual question that reuses
some of the SAME distinctive words or phrases found in the passage, so someone could
find the answer via a keyword search. Also give a short, correct answer using only
this passage.

Passage:
{chunk}

Respond with ONLY valid JSON, no markdown fences:
{{"question": "...", "answer": "..."}}""",

    "paraphrased": """From the passage below, write ONE specific factual question, but
phrase it using DIFFERENT words or synonyms than the passage uses — avoid reusing the
passage's distinctive vocabulary. This should test semantic understanding rather than
keyword overlap. Also give a short, correct answer using only this passage.

Passage:
{chunk}

Respond with ONLY valid JSON, no markdown fences:
{{"question": "...", "answer": "..."}}""",

    "vague": """From the passage below, write ONE short, casual, somewhat underspecified
question that a person might type quickly — use vague references or pronouns instead of
full names where it still makes sense — but the question must still have exactly ONE
correct answer findable in this passage. Also give a short, correct answer using only
this passage.

Passage:
{chunk}

Respond with ONLY valid JSON, no markdown fences:
{{"question": "...", "answer": "..."}}""",
}

MULTI_HOP_PROMPT = """Below are two passages, A and B, from the same document. Write ONE
question that REQUIRES combining information from BOTH passages — someone with only
passage A, or only passage B, could not fully answer it. Give a short, combined correct
answer using only these two passages.

Passage A:
{chunk_a}

Passage B:
{chunk_b}

Respond with ONLY valid JSON, no markdown fences:
{{"question": "...", "answer": "..."}}"""


def _parse_json(text: str):
    text = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _generate_single(chunk_doc, category: str):
    llm = get_llm()
    prompt = SINGLE_HOP_PROMPTS[category].format(chunk=chunk_doc.page_content)
    response = llm.invoke(prompt)
    parsed = _parse_json(response.content)
    if not parsed or "question" not in parsed:
        return None
    return {
        "question": parsed["question"],
        "answer": parsed.get("answer", ""),
        "category": category,
        "expected_chunk_ids": [chunk_doc.metadata["chunk_id"]],
    }


def _generate_multi(chunk_a, chunk_b):
    llm = get_llm()
    prompt = MULTI_HOP_PROMPT.format(chunk_a=chunk_a.page_content, chunk_b=chunk_b.page_content)
    response = llm.invoke(prompt)
    parsed = _parse_json(response.content)
    if not parsed or "question" not in parsed:
        return None
    return {
        "question": parsed["question"],
        "answer": parsed.get("answer", ""),
        "category": "multi_hop",
        "expected_chunk_ids": [chunk_a.metadata["chunk_id"], chunk_b.metadata["chunk_id"]],
    }


def generate_eval_set(
    docs,
    n_lexical: int = 4,
    n_paraphrased: int = 4,
    n_vague: int = 4,
    n_multi_hop: int = 4,
    progress_callback=None,
):
    """Build a ground-truth eval set from the loaded document's chunks, spread across
    four deliberately different question categories so no single pipeline is favored
    by construction. Returns a list of
    {question, answer, category, expected_chunk_ids} dicts.
    """
    usable = [d for d in docs if len(d.page_content) >= MIN_CHUNK_LEN]
    random.shuffle(usable)

    questions = []
    idx = 0
    total = n_lexical + n_paraphrased + n_vague + n_multi_hop
    done = 0

    for category, n in [("lexical", n_lexical), ("paraphrased", n_paraphrased), ("vague", n_vague)]:
        for _ in range(n):
            if idx >= len(usable):
                break
            chunk = usable[idx]
            idx += 1
            qa = _generate_single(chunk, category)
            if qa:
                questions.append(qa)
            done += 1
            if progress_callback:
                progress_callback(done / total)

    remaining = usable[idx:]
    random.shuffle(remaining)
    pair_idx = 0
    for _ in range(n_multi_hop):
        if pair_idx + 1 >= len(remaining):
            break
        a, b = remaining[pair_idx], remaining[pair_idx + 1]
        pair_idx += 2
        qa = _generate_multi(a, b)
        if qa:
            questions.append(qa)
        done += 1
        if progress_callback:
            progress_callback(done / total)

    random.shuffle(questions)
    return questions
