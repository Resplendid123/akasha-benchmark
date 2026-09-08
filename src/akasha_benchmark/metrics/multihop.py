"""多跳专项：图扩展的净价值，以及随跳数的衰减。

这部分是通用 RAG 基准测不出来的。Akasha 的编译器把实体物化成独立 artifact
并跨文档合并，所以两个 hop 可能被编译器直接连成一条 graph edge，
而不是靠两次独立检索各自找到。``snippets[].retrievalReasons`` 暴露了每个
snippet 是哪个信号产出的，图扩展的净贡献因此可以直接量化。

snippet 对应哪些页，走 ``sourceWindows[].sourcePageId``
（``KnowledgeSourceWindow extends KnowledgeCitation``）—— 这是把「检索原因」
和「是否命中 gold」关联起来的唯一路径。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

GRAPH_NEIGHBOR = "graph-neighbor"
# 已知的信号取值，取自 knowledge-retrieval.service.ts。
KNOWN_REASONS = ("semantic", "lexical", "exact-title", GRAPH_NEIGHBOR, "sidecar-prefiltered")


def snippet_doc_ids(snippet: dict[str, Any], page_to_doc: dict[str, str]) -> set[str]:
    """一个 snippet 背后对应的 doc_id 集合。"""
    return {
        page_to_doc[window["sourcePageId"]]
        for window in snippet.get("sourceWindows") or []
        if window.get("sourcePageId") in page_to_doc
    }


def evaluate_sample(
    snippets: Sequence[dict[str, Any]], gold: Sequence[str], page_to_doc: dict[str, str]
) -> dict[str, Any]:
    """按信号统计 snippet 条数，以及其中有多少条命中了 gold。"""
    gold_set = set(gold)
    reason_counts: Counter[str] = Counter()
    reason_gold_counts: Counter[str] = Counter()
    reason_gold_docs: dict[str, set[str]] = {}

    graph_snippets = 0
    graph_gold_snippets = 0
    gold_only_from_graph: set[str] = set()
    gold_from_other: set[str] = set()

    for snippet in snippets:
        reasons = snippet.get("retrievalReasons") or []
        docs = snippet_doc_ids(snippet, page_to_doc)
        hits = docs & gold_set

        # 一个 snippet 可能同时挂多个原因，按 set 去重后各自记一次。
        for reason in set(reasons):
            reason_counts[reason] += 1
            if hits:
                reason_gold_counts[reason] += 1
                reason_gold_docs.setdefault(reason, set()).update(hits)

        if GRAPH_NEIGHBOR in reasons:
            graph_snippets += 1
            if hits:
                graph_gold_snippets += 1
            gold_only_from_graph |= hits
        else:
            gold_from_other |= hits

    total = len(snippets)
    # 只能靠图扩展才拿到的 gold —— 这是 graph edge 的净增量价值，
    # 即语义/词法召回本来就能找到的部分之外，图额外贡献了什么。
    graph_exclusive_gold = gold_only_from_graph - gold_from_other

    return {
        "snippet_count": total,
        "graph_neighbor_snippets": graph_snippets,
        "graph_neighbor_share": graph_snippets / total if total else 0.0,
        "graph_neighbor_gold_snippets": graph_gold_snippets,
        "graph_neighbor_precision": graph_gold_snippets / graph_snippets if graph_snippets else 0.0,
        "graph_exclusive_gold_count": len(graph_exclusive_gold),
        "graph_exclusive_gold_share": (
            len(graph_exclusive_gold) / len(gold_set) if gold_set else 0.0
        ),
        "reason_counts": dict(reason_counts),
        "reason_gold_counts": dict(reason_gold_counts),
        "reason_gold_doc_counts": {r: len(d) for r, d in sorted(reason_gold_docs.items())},
    }


def aggregate(per_sample: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """数值项取均值；按信号的计数项累加后再算命中率。"""
    if not per_sample:
        return {}
    numeric_keys = [
        "snippet_count",
        "graph_neighbor_snippets",
        "graph_neighbor_share",
        "graph_neighbor_gold_snippets",
        "graph_neighbor_precision",
        "graph_exclusive_gold_count",
        "graph_exclusive_gold_share",
    ]
    result: dict[str, Any] = {
        key: sum(row[key] for row in per_sample) / len(per_sample) for key in numeric_keys
    }

    reason_totals: Counter[str] = Counter()
    reason_gold_totals: Counter[str] = Counter()
    for row in per_sample:
        reason_totals.update(row["reason_counts"])
        reason_gold_totals.update(row["reason_gold_counts"])

    result["reason_totals"] = dict(sorted(reason_totals.items()))
    result["reason_gold_totals"] = dict(sorted(reason_gold_totals.items()))
    # 各信号「产出的 snippet 里有多少命中 gold」，即信号自身的质量。
    result["reason_gold_rate"] = {
        reason: reason_gold_totals[reason] / count
        for reason, count in sorted(reason_totals.items())
        if count
    }
    return result


def stratify(
    rows: Sequence[dict[str, Any]], key: str
) -> dict[str, list[dict[str, Any]]]:
    """按某个 metadata 字段给逐样本结果分组，用于出随跳数的衰减曲线。"""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = str((row.get("metadata") or {}).get(key, "unknown"))
        buckets.setdefault(value, []).append(row)
    return dict(sorted(buckets.items()))
