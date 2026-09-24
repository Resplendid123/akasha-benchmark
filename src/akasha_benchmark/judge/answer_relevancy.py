"""answer_relevancy：生成答案对应的问题，再与原问题比较 embedding。"""

from __future__ import annotations

import math
from typing import Any

import httpx

QUESTION_SYSTEM = """Generate the single question that the ANSWER is primarily trying to answer.
Return JSON only in exactly this shape: {\"question\": \"...\"}.
Do not answer the question, explain your choice, or copy the answer verbatim."""
QUESTION_USER = """ANSWER:
{answer}
"""


def build_prompt(answer: str) -> tuple[str, str] | None:
    if not answer.strip():
        return None
    return QUESTION_SYSTEM, QUESTION_USER.format(answer=answer[:6000])


def parse_generated_question(payload: dict[str, Any]) -> str:
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("generated question must be a non-empty string")
    return question.strip()[:1000]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        raise ValueError("embedding vectors must be non-empty and have equal dimensions")
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if not norm_left or not norm_right:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_left * norm_right)))


def embedding_similarity(
    question: str,
    generated_question: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_seconds: float = 120.0,
) -> float:
    response = httpx.post(
        f"{base_url.rstrip('/')}/embeddings",
        json={"model": model, "input": [question, generated_question]},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(timeout_seconds),
    )
    response.raise_for_status()
    body = response.json()
    vectors = [item.get("embedding") for item in body.get("data") or []]
    if len(vectors) != 2 or not all(isinstance(vector, list) for vector in vectors):
        raise ValueError("embedding response must contain two vectors")
    return cosine_similarity(vectors[0], vectors[1])
