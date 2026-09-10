"""corpus 加载、doc_id 赋予，以及适配器解析 gold 要用的反查表。

四组 corpus 文件的身份字段并不统一：hotpotqa 和 narrativeqa 带 ``idx``，
2wiki 和 musique 只有 ``{title, text}``。这个差异只在本模块收口，
规则见 :data:`CORPUS_ID_RULES`。

**语料一律不去重。** musique 有 647 个 title 重复、涉及 2465 行，
但它们是同名文档的不同段落，去重会直接丢掉 gold。
去重前后的条数都写进 manifest，这个选择随时可查。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..io_utils import load_json
from .models import CORPUS_ID_RULES, CorpusDoc


class CorpusIndex:
    """corpus 全部行，加上解析 gold 需要的两张反查表。"""

    def __init__(self, dataset: str, docs: list[CorpusDoc]) -> None:
        self.dataset = dataset
        self.docs = docs

        self.by_id: dict[str, CorpusDoc] = {}
        for doc in docs:
            if doc.doc_id in self.by_id:
                raise ValueError(
                    f"{dataset}: duplicate doc_id {doc.doc_id!r}; "
                    f"corpus identity rule {CORPUS_ID_RULES[dataset]!r} is not unique"
                )
            self.by_id[doc.doc_id] = doc

        # title -> [doc_id]（可能一对多）；(title, text) -> doc_id（唯一）。
        self.title_to_ids: dict[str, list[str]] = defaultdict(list)
        self.pair_to_id: dict[tuple[str, str], str] = {}
        collisions: list[tuple[str, str]] = []
        for doc in docs:
            self.title_to_ids[doc.title].append(doc.doc_id)
            key = (doc.title, doc.text)
            if key in self.pair_to_id:
                collisions.append(key)
            else:
                self.pair_to_id[key] = doc.doc_id

        # PLAN.md 0.1：(title, text) 仍重复的话，就没有任何键能定位到行了，
        # 此时必须报错，不能静默挑一个。
        if collisions:
            title, text = collisions[0]
            raise ValueError(
                f"{dataset}: {len(collisions)} duplicate (title, text) pairs, "
                f"e.g. title={title!r} text[:80]={text[:80]!r}. "
                "No unique corpus key remains; refusing to guess."
            )

    def __len__(self) -> int:
        return len(self.docs)

    @property
    def unique_title_count(self) -> int:
        return len(self.title_to_ids)

    def id_for_title(self, title: str) -> str:
        """按 title 定位唯一 doc_id。有歧义或找不到都抛异常。"""
        ids = self.title_to_ids.get(title)
        if not ids:
            raise KeyError(f"{self.dataset}: title not in corpus: {title!r}")
        if len(ids) > 1:
            raise KeyError(
                f"{self.dataset}: title maps to {len(ids)} corpus rows: {title!r}; "
                "use id_for_pair to disambiguate"
            )
        return ids[0]

    def id_for_pair(self, title: str, text: str) -> str:
        """按 (title, text) 定位 doc_id。musique 必须走这条路。"""
        try:
            return self.pair_to_id[(title, text)]
        except KeyError:
            raise KeyError(
                f"{self.dataset}: (title, text) not in corpus: title={title!r} "
                f"text[:80]={text[:80]!r}"
            ) from None

    def dedup_stats(self) -> dict[str, int]:
        """给 manifest 用的统计。只报告，不执行去重。"""
        return {
            "rows": len(self.docs),
            "unique_titles": len(self.title_to_ids),
            "unique_title_text_pairs": len(self.pair_to_id),
            "rows_in_duplicate_title_groups": sum(
                len(ids) for ids in self.title_to_ids.values() if len(ids) > 1
            ),
        }


def assign_doc_id(dataset: str, row: dict[str, Any], row_index: int) -> str:
    """按该数据集声明的规则赋 doc_id。"""
    rule = CORPUS_ID_RULES[dataset]
    if rule == "native_idx":
        if "idx" not in row:
            raise ValueError(
                f"{dataset}: corpus row {row_index} has no 'idx' but the identity "
                f"rule is {rule!r}; upstream data shape changed"
            )
        return str(row["idx"])
    if rule == "row_index":
        # 之所以用行号当身份，前提就是这份 corpus 没有 idx。
        # 它突然有了，说明数据换版了，行号身份不再可信。
        if "idx" in row:
            raise ValueError(
                f"{dataset}: corpus row {row_index} unexpectedly has an 'idx' field. "
                f"The identity rule is {rule!r} because this corpus had none. "
                "Re-check the data before trusting row numbers as identity."
            )
        return str(row_index)
    raise ValueError(f"{dataset}: unknown corpus identity rule {rule!r}")


def load_corpus(dataset: str, path: Path) -> CorpusIndex:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON array, got {type(rows).__name__}")

    docs: list[CorpusDoc] = []
    for row_index, row in enumerate(rows):
        missing = {"title", "text"} - row.keys()
        if missing:
            raise ValueError(f"{path}: row {row_index} missing {sorted(missing)}")
        docs.append(
            CorpusDoc(
                doc_id=assign_doc_id(dataset, row, row_index),
                title=row["title"],
                text=row["text"],
            )
        )
    return CorpusIndex(dataset, docs)
