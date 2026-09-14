"""answer_relevancy：答案是否在回答这个问题。

让模型逐句判「这句是否服务于该问题」，得分 = 相关句占比。不走 RAGAS 原版的
嵌入相似度（平台的 judge 端点只有 chat），所以绝对值只可同配置比较。

不依赖参考答案：判的是「答没答到点上」而不是「答得对不对」。
"""

from __future__ import annotations

from typing import Any

SYSTEM = """You judge whether an ANSWER actually addresses the QUESTION.

Do this:
1. Split the ANSWER into sentences. Ignore pure boilerplate ("Here is what I
   found", "Let me help you with that").
2. For each sentence decide:
   - "relevant": it contributes to answering the question.
   - "irrelevant": it is off-topic, filler, or answers something else.
3. Judge relevance to the QUESTION only. Factual correctness is NOT your concern
   here — a wrong-but-on-topic sentence is still "relevant".

Return JSON only, no prose, in exactly this shape:
{"sentences": [{"sentence": "<text>", "verdict": "relevant|irrelevant"}]}

If the ANSWER is a refusal or contains no substantive sentences,
return {"sentences": []}."""

USER_TEMPLATE = """QUESTION:
{question}

ANSWER:
{answer}"""

VERDICTS = ("relevant", "irrelevant")


def score_sentences(sentences: list[dict[str, Any]]) -> float | None:
    """相关句占比。没有实质句子时返回 ``None``（拒答上这个指标无定义）。"""
    if not sentences:
        return None
    relevant = sum(1 for s in sentences if s.get("verdict") == "relevant")
    return relevant / len(sentences)


def parse_verdict(payload: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    sentences = payload.get("sentences")
    if not isinstance(sentences, list):
        raise ValueError(f"expected a list under 'sentences', got {type(sentences).__name__}")

    cleaned: list[dict[str, Any]] = []
    for entry in sentences:
        if not isinstance(entry, dict):
            raise ValueError(f"sentence entries must be objects, got {type(entry).__name__}")
        verdict = entry.get("verdict")
        if verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {verdict!r}; expected one of {list(VERDICTS)}")
        cleaned.append(
            {"sentence": str(entry.get("sentence") or "")[:500], "verdict": verdict}
        )

    return score_sentences(cleaned), {
        "sentences": cleaned,
        "sentence_count": len(cleaned),
        "relevant": sum(1 for s in cleaned if s["verdict"] == "relevant"),
    }


def build_prompt(
    question: str, answer: str, response: dict[str, Any]
) -> tuple[str, str] | None:
    """拼出 ``(system, user)``。问题或答案为空时返回 ``None``，该条跳过。

    ``response`` 用不上，保留形参是为了与另外三个判据共用调用签名。
    """
    if not question.strip() or not answer.strip():
        return None
    return SYSTEM, USER_TEMPLATE.format(question=question, answer=answer)
