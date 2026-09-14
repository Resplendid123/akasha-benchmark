"""多跳专项：图扩展的净价值。

``snippets[].retrievalReasons`` 给出每个 snippet 由哪个信号产出，
``sourceWindows[].sourcePageId`` 给出它对应哪些页。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

GRAPH_NEIGHBOR = "graph-neighbor"


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
    # 只能靠图扩展才拿到的 gold，即图边的净增量。
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
