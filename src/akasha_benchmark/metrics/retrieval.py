"""检索指标：Recall@k、nDCG@k、MRR、Hit@k，输入是排序后的 page id 列表。

用 ``retrievedSources`` 算，不用 ``citations``（后者已被裁剪过）。
相关性是二元的，所以 nDCG 的理想排序是把全部 gold 排在最前。
"""

from __future__ import annotations

from collections.abc import Sequence
from math import log2
from typing import Any

DEFAULT_KS: tuple[int, ...] = (2, 5, 10, 20)


def ranked_doc_ids(retrieved: Sequence[dict[str, Any]], page_to_doc: dict[str, str]) -> list[str]:
    """按排序返回去重后的 doc_id。反查不到的 page 用占位键占住排名位。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for source in retrieved:
        page_id = source.get("sourcePageId")
        if not page_id:
            continue
        doc_id = page_to_doc.get(page_id)
        key = doc_id if doc_id is not None else f"__unmapped__:{page_id}"
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def unmapped_page_ids(
    retrieved: Sequence[dict[str, Any]], page_to_doc: dict[str, str]
) -> list[str]:
    """召回结果里不在 page_map 中的 page id。"""
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
    """前 k 个里凑齐全部 gold 才算 1。"""
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
