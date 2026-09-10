"""答案质量：Exact Match 与 token F1，用标准 SQuAD/MRQA 归一化口径。

归一化刻意照抄已发表的做法 —— 小写、去标点、去冠词 ``a``/``an``/``the``、
合并空白 —— 这样数字才能跟 HippoRAG 2 之类的公开结果对得上。
自创口径会让所有对外比较失去意义。

多参考答案取 max：musique 的别名在归一化时已并入 ``answers``，
narrativeqa 本身带 2 条人工参考。

**EM 在这套架构上预期恒为 0，这不是 bug。** EM 要求整串归一化后完全相等，而
Akasha 返回的是解释性散文（"The disease described is yellow fever, which is
caused by ... belonging to the genus **Flavivirus**."），参考答案是
``Flavivirus`` 这样的短跨度，两者不可能相等。2026-09-09 的在线冒烟实测：三条
答案全部实质正确、gold 全部召回（``recall@10`` 与 ``full_coverage@10`` 均为
1.000），EM 仍是 0.000。

所以 **EM 只能当形态探针读，不能当质量指标读**：它变成非 0 意味着生成端开始
输出短跨度答案（换了 answer prompt 或换了模型），而不是意味着答案变对了。
判断答案对不对看 F1、``citation_precision``、``evidence_verifiable_rate``。

F1 受同一稀释效应影响但**仍然是变化的**（那三条分别 0.054 / 0.087 / 0.143），
只是绝对值被解释性 token 压低，只可用于同配置之间比较。
两个指标的计算方式与低分归因详见 metrics.md「答案质量」。
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
    """整串归一化后完全相等才算 1。

    对解释性散文答案预期恒为 0，见模块说明 —— 读它是为了知道生成端的**答案形态**
    有没有变，不是为了知道答案对不对。
    """
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
    """对多条参考答案取 max，两个指标各自独立取 max。

    ``em`` 预期恒为 0，是形态探针；``f1`` 是可读的质量指标。见模块说明。
    """
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
