"""引用归因：precision / recall、裁剪损失、证据可验证率。

``citations`` 是最终面向答案的集合，由 ``resolveAnswerCitations`` 从
``retrievedSources`` 收窄成「答案真的引了」且「有证据支撑」的部分
（``ai-knowledge-chat.service.ts:527-534``）。

两个集合的差集才是有意思的量：落在差集里的文档是被检索到了但没露出来，
其中若有 gold，说明是引用过滤太严，而不是检索没找到。
这两种问题要调的地方完全不同。
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
    citation_evidence: Sequence[dict[str, Any]],
    gold: Sequence[str],
    page_to_doc: dict[str, str],
) -> dict[str, float]:
    """单条样本的归因指标。"""
    gold_set = set(gold)
    cited = to_doc_ids(citations, page_to_doc)
    retrieved_docs = to_doc_ids(retrieved, page_to_doc)

    cited_set, retrieved_set = set(cited), set(retrieved_docs)
    truncated = retrieved_set - cited_set

    evidence_backed = sum(1 for e in citation_evidence if e.get("excerpts"))
    evidence_total = len(citation_evidence)

    return {
        "citation_precision": len(cited_set & gold_set) / len(cited_set) if cited_set else 0.0,
        "citation_recall": len(cited_set & gold_set) / len(gold_set) if gold_set else 0.0,
        "citation_count": float(len(cited_set)),
        "retrieved_count": float(len(retrieved_set)),
        # 被检索到但没进答案引用的文档数。
        "truncation_loss": float(len(truncated)),
        # 其中本来是 gold 的：检索找到了，是引用过滤把它丢了。
        "truncated_gold": float(len(truncated & gold_set)),
        # excerpts 非空的引用占比，即「这条引用能不能被核验」。
        "evidence_verifiable_rate": (
            evidence_backed / evidence_total if evidence_total else 0.0
        ),
        "evidence_entries": float(evidence_total),
    }


def aggregate(per_sample: Sequence[dict[str, float]]) -> dict[str, float]:
    if not per_sample:
        return {}
    return {
        key: sum(row[key] for row in per_sample) / len(per_sample)
        for key in per_sample[0]
    }
