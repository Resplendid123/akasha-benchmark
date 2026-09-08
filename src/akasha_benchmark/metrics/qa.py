"""答案质量：Exact Match 与 token F1，用标准 SQuAD/MRQA 归一化口径。

归一化刻意照抄已发表的做法 —— 小写、去标点、去冠词 ``a``/``an``/``the``、
合并空白 —— 这样数字才能跟 HippoRAG 2 之类的公开结果对得上。
自创口径会让所有对外比较失去意义。

多参考答案取 max：musique 的别名在轮次 1 已并入 ``answers``，
narrativeqa 本身带 2 条人工参考。
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """标准口径：小写 -> 去标点 -> 去冠词 -> 合并空白。顺序不能换。"""
    lowered = text.lower()
    without_punct = lowered.translate(_PUNCT_TABLE)
    without_articles = _ARTICLES.sub(" ", without_punct)
    return _WHITESPACE.sub(" ", without_articles).strip()


def tokenize(text: str) -> list[str]:
    return normalize_answer(text).split()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """词袋级 F1，重复词按出现次数取交集。"""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    # 退化情况，与官方 SQuAD 脚本口径一致：两边都空算完全匹配，
    # 预测为空而参考非空算 0。
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    shared = sum(overlap.values())
    if shared == 0:
        return 0.0
    precision = shared / len(pred_tokens)
    recall = shared / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, references: Sequence[str]) -> dict[str, float]:
    """对多条参考答案取 max。"""
    if not references:
        raise ValueError("score_answer requires at least one reference")
    prediction = prediction or ""
    return {
        "em": max(exact_match(prediction, r) for r in references),
        "f1": max(token_f1(prediction, r) for r in references),
    }


def answer_mode_distribution(modes: Sequence[str | None]) -> dict[str, float]:
    """各 ``answerMode`` 的占比。

    ``no_match`` 率与 ``general`` 兜底率是「检索没喂够料」的直接信号：
    这两条路径返回的 citations 和 retrievedSources 都是空的。
    """
    total = len(modes)
    if not total:
        return {}
    counts = Counter(mode or "missing" for mode in modes)
    return {mode: count / total for mode, count in sorted(counts.items())}
