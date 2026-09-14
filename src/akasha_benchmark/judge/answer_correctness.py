"""answer_correctness：答案与参考答案在事实上是否一致。

需要参考答案，所以 narrativeqa 上省略。判「说的是不是同一件事」，
不受措辞与长度影响，补 EM 与 F1 在解释性长答案上失真的那个洞。

分四档而不是连续打分：模型输出的浮点数会集中在几个值上，档位间的差异没有意义。
"""

from __future__ import annotations

from typing import Any

SYSTEM = """You compare an ANSWER against a REFERENCE answer for the same
QUESTION and rate factual agreement.

Pick exactly one verdict:
- "correct": states the same facts as the reference. Extra correct detail,
  different wording, and different length are all fine.
- "partial": gets part of it right but misses or garbles a required part. Use
  this for multi-part questions answered only in part.
- "incorrect": contradicts the reference, or answers a different question.
- "no_answer": a refusal or an admission that it could not find the information.
  This is NOT the same as incorrect — the answer makes no factual claim.

Wording never matters. Judge facts only. A one-word answer that matches the
reference is "correct".

Return JSON only, no prose, in exactly this shape:
{"verdict": "correct|partial|incorrect|no_answer", "reason": "<one sentence>"}"""

USER_TEMPLATE = """QUESTION:
{question}

REFERENCE:
{reference}

ANSWER:
{answer}"""

# no_answer 不给分数：拒答既不对也不错，与「答错了」的处置完全不同。
SCORES: dict[str, float | None] = {
    "correct": 1.0,
    "partial": 0.5,
    "incorrect": 0.0,
    "no_answer": None,
}


def parse_verdict(payload: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    verdict = payload.get("verdict")
    if verdict not in SCORES:
        raise ValueError(f"unknown verdict {verdict!r}; expected one of {sorted(SCORES)}")
    return SCORES[verdict], {
        "verdict": verdict,
        "reason": str(payload.get("reason") or "")[:500],
    }


def build_prompt(
    question: str, answer: str, reference: str
) -> tuple[str, str] | None:
    """拼出 ``(system, user)``。缺参考答案或答案为空时返回 ``None``。"""
    if not question.strip() or not answer.strip() or not reference.strip():
        return None
    return SYSTEM, USER_TEMPLATE.format(
        question=question, reference=reference, answer=answer
    )
