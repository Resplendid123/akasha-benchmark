"""hotpotqa 与 2wikimultihopqa 共用的 ``supporting_facts`` 解析。

gold evidence 是 ``[title, 句子下标]`` 对，同一 title 会重复出现，
所以 gold 去重成 title 集合，否则 recall 的分母会算大。
"""

from __future__ import annotations

from typing import Any

from .corpus import CorpusIndex


def gold_titles_from_supporting_facts(
    row: dict[str, Any], row_index: int, dataset: str
) -> tuple[str, ...]:
    """返回去重后的 gold title，顺序按首次出现。"""
    facts = row.get("supporting_facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError(f"{dataset}: row {row_index} has empty or non-list supporting_facts")

    seen: dict[str, None] = {}
    for fact in facts:
        if not isinstance(fact, (list, tuple)) or len(fact) != 2:
            raise ValueError(
                f"{dataset}: row {row_index} supporting_facts entry is not "
                f"[title, sentence_index]: {fact!r}"
            )
        title = fact[0]
        if not isinstance(title, str) or not title:
            raise ValueError(f"{dataset}: row {row_index} supporting_facts title is not a string")
        seen.setdefault(title, None)
    return tuple(seen)


def resolve_gold_doc_ids(
    titles: tuple[str, ...], corpus: CorpusIndex, row_index: int, dataset: str
) -> tuple[str, ...]:
    """gold title 映射成 doc_id。"""
    resolved: list[str] = []
    for title in titles:
        try:
            resolved.append(corpus.id_for_title(title))
        except KeyError as exc:
            raise ValueError(f"{dataset}: row {row_index} gold unresolvable: {exc}") from None
    return tuple(resolved)
