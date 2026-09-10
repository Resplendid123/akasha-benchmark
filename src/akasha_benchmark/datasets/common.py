"""hotpotqa 与 2wikimultihopqa 共用的 ``supporting_facts`` 解析。

两者的 gold evidence 都是 ``[title, 句子下标]`` 对，所以同一个 title 会按
支撑句的数量重复出现。因此 gold **必须**去重成 title 集合：
hotpotqa 有 350 行、2wiki 有 1 行的 gold title 列表内部有重复，
不去重就会把 recall 的分母算大。

两者的句子拼接符不同（hotpotqa 用 ``""``、2wiki 用 ``" "``），这里保留成参数。
各阶段全程按 id 匹配，用不到拼接；但原文基线要从 ``context``
还原文档，所以这个知识留在这里，免得到时候重新去踩一遍。
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
    """gold title 映射成 doc_id。在锁定快照上实测命中率 100%。"""
    resolved: list[str] = []
    for title in titles:
        try:
            resolved.append(corpus.id_for_title(title))
        except KeyError as exc:
            raise ValueError(f"{dataset}: row {row_index} gold unresolvable: {exc}") from None
    return tuple(resolved)


def join_context_sentences(row: dict[str, Any], joiner: str) -> dict[str, str]:
    """把 ``context`` 还原成 ``title -> 文档正文``。供原文基线用。"""
    return {title: joiner.join(sentences) for title, sentences in row["context"]}
