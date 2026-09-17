"""answer_relevancy：答案是否在回答这个问题。

让模型逐句判「这句是否服务于该问题」，得分 = 相关句占比。不走 RAGAS 原版的
嵌入相似度（平台的 judge 端点只有 chat），所以绝对值只可同配置比较。

不依赖参考答案：判的是「答没答到点上」而不是「答得对不对」。
"""

from __future__ import annotations

import re
from typing import Any

SYSTEM = """You judge whether an ANSWER actually addresses the QUESTION.

The ANSWER has already been split into numbered sentences.
For each numbered sentence decide:
   - "relevant": it contributes to answering the question.
   - "irrelevant": it is off-topic, filler, or answers something else.
   - "ignore": it is a refusal, heading, or pure boilerplate with no substantive claim.
Judge relevance to the QUESTION only. Factual correctness is NOT your concern
   here — a wrong-but-on-topic sentence is still "relevant".

Return JSON only, no prose, in exactly this shape:
{"verdicts": [{"index": 1, "verdict": "relevant|irrelevant|ignore"}]}

Return exactly one verdict for every numbered sentence, in the same order.
Never copy sentence text into the JSON."""

USER_TEMPLATE = """QUESTION:
{question}

NUMBERED ANSWER SENTENCES:
{sentences}"""

VERDICTS = ("relevant", "irrelevant", "ignore")
MAX_SENTENCES = 100
MAX_SENTENCE_CHARS = 1000
_SENTENCE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")


def split_sentences(answer: str) -> list[str]:
    """本地稳定分句，避免让模型把带引号的原文复制进 JSON。"""
    return [
        match.group(0).strip()[:MAX_SENTENCE_CHARS]
        for match in _SENTENCE.finditer(answer)
        if match.group(0).strip()
    ][:MAX_SENTENCES]


def score_sentences(sentences: list[dict[str, Any]]) -> float | None:
    """相关句占比。没有实质句子时返回 ``None``（拒答上这个指标无定义）。"""
    judged = [sentence for sentence in sentences if sentence.get("verdict") != "ignore"]
    if not judged:
        return None
    relevant = sum(1 for sentence in judged if sentence.get("verdict") == "relevant")
    return relevant / len(judged)


def parse_verdict(
    payload: dict[str, Any], sentences: list[str]
) -> tuple[float | None, dict[str, Any]]:
    verdicts = payload.get("verdicts")
    if not isinstance(verdicts, list):
        raise ValueError(f"expected a list under 'verdicts', got {type(verdicts).__name__}")
    if len(verdicts) != len(sentences):
        raise ValueError(f"expected {len(sentences)} verdicts, got {len(verdicts)}")

    by_index: dict[int, str] = {}
    for entry in verdicts:
        if not isinstance(entry, dict):
            raise ValueError(f"verdict entries must be objects, got {type(entry).__name__}")
        index = entry.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or not 1 <= index <= len(sentences):
            raise ValueError(f"invalid sentence index {index!r}; expected 1..{len(sentences)}")
        if index in by_index:
            raise ValueError(f"duplicate sentence index {index}")
        verdict = entry.get("verdict")
        if verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {verdict!r}; expected one of {list(VERDICTS)}")
        by_index[index] = verdict

    cleaned = [
        {"index": index, "sentence": sentence[:500], "verdict": by_index[index]}
        for index, sentence in enumerate(sentences, 1)
    ]

    return score_sentences(cleaned), {
        "sentences": cleaned,
        "sentence_count": len(cleaned),
        "relevant": sum(1 for s in cleaned if s["verdict"] == "relevant"),
        "irrelevant": sum(1 for s in cleaned if s["verdict"] == "irrelevant"),
        "ignored": sum(1 for s in cleaned if s["verdict"] == "ignore"),
    }


def build_prompt(
    question: str, answer: str, response: dict[str, Any]
) -> tuple[str, str, list[str]] | None:
    """拼出 prompt 与本地分句。问题或答案为空时返回 ``None``。

    ``response`` 用不上，保留形参是为了与另外三个判据共用调用签名。
    """
    sentences = split_sentences(answer)
    if not question.strip() or not sentences:
        return None
    numbered = "\n".join(f"[{index}] {sentence}" for index, sentence in enumerate(sentences, 1))
    return SYSTEM, USER_TEMPLATE.format(question=question, sentences=numbered), sentences
