"""检索指标：Recall@k、nDCG@k、MRR、Hit@k，输入是排序后的 page id 列表。

用 ``retrievedSources`` 算，不用 ``citations``：前者是裁剪前的召回全集
（``ai-knowledge-chat.service.ts:544``），后者已经被「被引 ∩ 有证据」的交集
裁过一遍，拿它算 Recall 会低估检索能力。

相关性是二元的 —— 所有 gold 同权 —— 所以 nDCG 的理想排序就是把全部 gold 排在最前。

有一条影响读数的坑：``no_match`` 和 ``general`` 两种 answerMode 会**无条件**
返回空的 ``retrievedSources``（``ai-knowledge-chat.service.ts:641,667``），
这些行不论检索实际找到什么，分数都是 0。所以汇总时必须把
``knowledge`` 切片单独报一份，见 :func:`aggregate` 的调用方。
"""

from __future__ import annotations

from collections.abc import Sequence
from math import log2
from typing import Any

from ..datasets import Capability, CapabilityError

DEFAULT_KS: tuple[int, ...] = (2, 5, 10, 20)


def require_evidence_capability(dataset: str, capabilities: frozenset[Capability]) -> None:
    """没有 gold 标注就没有检索指标，此时拒绝计算而不是返回 0.0。"""
    if Capability.EVIDENCE_RECALL not in capabilities:
        raise CapabilityError(
            f"{dataset} does not declare EVIDENCE_RECALL: it has no gold documents, so "
            "Recall/nDCG/MRR/Hit are undefined. Returning 0.0 would silently pollute "
            "any aggregate that includes it."
        )


def ranked_doc_ids(retrieved: Sequence[dict[str, Any]], page_to_doc: dict[str, str]) -> list[str]:
    """按排序返回去重后的 doc_id。反查不到的 page 仍然占住它的排名位。

    保留未知 page 的排名位是有意的：直接丢掉它们会让后面的结果整体前移，
    把所有对排名敏感的指标都算高。这些 page 由 :func:`unmapped_page_ids` 单独统计。
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for source in retrieved:
        page_id = source.get("sourcePageId")
        if not page_id:
            continue
        doc_id = page_to_doc.get(page_id)
        # 反查不到就用一个不可能等于任何 gold 的占位键，只为占住名次。
        key = doc_id if doc_id is not None else f"__unmapped__:{page_id}"
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def unmapped_page_ids(
    retrieved: Sequence[dict[str, Any]], page_to_doc: dict[str, str]
) -> list[str]:
    """召回结果里不在 page_map 中的 page id。非空说明库里有本次子集之外的页。"""
    return sorted(
        {
            s["sourcePageId"]
            for s in retrieved
            if s.get("sourcePageId") and s["sourcePageId"] not in page_to_doc
        }
    )


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """前 k 个里命中的 gold 占全部 gold 的比例。"""
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("recall_at_k requires at least one gold document")
    return len(gold_set & set(ranked[:k])) / len(gold_set)


def hit_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """前 k 个里至少命中一个 gold 就算 1。"""
    return 1.0 if set(gold) & set(ranked[:k]) else 0.0


def mrr(ranked: Sequence[str], gold: Sequence[str]) -> float:
    """首个 gold 的倒数排名。"""
    gold_set = set(gold)
    for position, doc_id in enumerate(ranked, 1):
        if doc_id in gold_set:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """二元相关性下的 nDCG。理想排序是全部 gold 占据最前的 min(len(gold), k) 位。"""
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("ndcg_at_k requires at least one gold document")
    dcg = sum(1.0 / log2(rank + 1) for rank, d in enumerate(ranked[:k], 1) if d in gold_set)
    ideal = sum(1.0 / log2(rank + 1) for rank in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal if ideal else 0.0


def full_coverage(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """前 k 个里**凑齐全部** gold 才算 1。

    多跳题少一跳就答不对。平均 Recall 0.5 既可能是「一半题全齐」，
    也可能是「每题都差一篇」，这两种情况的性质完全不同，
    所以这个指标比 Recall 均值更贴近多跳的实际需求。
    """
    gold_set = set(gold)
    return 1.0 if gold_set and gold_set <= set(ranked[:k]) else 0.0


def evaluate_sample(
    ranked: Sequence[str], gold: Sequence[str], ks: Sequence[int] = DEFAULT_KS
) -> dict[str, float]:
    """单条样本的全部检索指标。"""
    metrics: dict[str, float] = {"mrr": mrr(ranked, gold)}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(ranked, gold, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked, gold, k)
        metrics[f"hit@{k}"] = hit_at_k(ranked, gold, k)
        metrics[f"full_coverage@{k}"] = full_coverage(ranked, gold, k)
    return metrics


def aggregate(per_sample: Sequence[dict[str, float]]) -> dict[str, float]:
    """逐样本指标按算术平均汇总。"""
    if not per_sample:
        return {}
    keys = per_sample[0].keys()
    return {key: sum(row[key] for row in per_sample) / len(per_sample) for key in keys}
