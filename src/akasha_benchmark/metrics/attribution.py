"""引用归因：precision / recall、裁剪损失、证据可验证率。

``citations`` 是 ``retrievedSources`` 收窄成「答案真的引了」且「有证据支撑」
的子集，两者的差集即被检索到但未露出的文档。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _page_ids(entries: Sequence[dict[str, Any]]) -> list[str]:
    """保序去重地取出 sourcePageId。"""
    seen: dict[str, None] = {}
    for entry in entries:
        page_id = entry.get("sourcePageId")
        if page_id:
            seen.setdefault(page_id, None)
    return list(seen)


def to_doc_ids(entries: Sequence[dict[str, Any]], page_to_doc: dict[str, str]) -> list[str]:
    """page id 反查成 doc_id，查不到的直接跳过。"""
    return [page_to_doc[p] for p in _page_ids(entries) if p in page_to_doc]


def evaluate_sample(
    citations: Sequence[dict[str, Any]],
    retrieved: Sequence[dict[str, Any]],
    gold: Sequence[str],
    page_to_doc: dict[str, str],
) -> dict[str, float]:
    """单条样本的归因指标。"""
    gold_set = set(gold)
    cited = to_doc_ids(citations, page_to_doc)
    retrieved_docs = to_doc_ids(retrieved, page_to_doc)

    cited_set, retrieved_set = set(cited), set(retrieved_docs)
    uncited = retrieved_set - cited_set

    return {
        "citation_precision": len(cited_set & gold_set) / len(cited_set) if cited_set else 0.0,
        "citation_recall": len(cited_set & gold_set) / len(gold_set) if gold_set else 0.0,
        "uncited_count": float(len(uncited)),
        "uncited_gold_count": float(len(uncited & gold_set)),
    }
