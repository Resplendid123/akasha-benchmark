"""corpus 加载、doc_id 赋予，以及适配器解析 gold 要用的反查表。

各组 corpus 的身份字段不统一，规则收在 :data:`CORPUS_ID_RULES`。
语料一律不去重：musique 的重复 title 是同名文档的不同段落，去重会丢 gold。
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

        # (title, text) 仍重复时没有任何键能定位到行，报错而不是挑一个。
        if collisions:
            title, text = collisions[0]
            raise ValueError(
                f"{dataset}: {len(collisions)} duplicate (title, text) pairs, "
                f"e.g. title={title!r} text[:80]={text[:80]!r}. "
                "No unique corpus key remains; refusing to guess."
            )

    def __len__(self) -> int:
        return len(self.docs)

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


def assign_doc_id(dataset: str, row: dict[str, Any], row_index: int) -> str:
    """按该数据集声明的身份字段赋 doc_id；``row_idx`` 表示用行号。"""
    field = CORPUS_ID_RULES[dataset]
    if field == "row_idx":
        # 用行号当身份的前提是这份 corpus 没有身份字段；有了说明数据换版。
        present = sorted({"idx", "id"} & row.keys())
        if present:
            raise ValueError(
                f"{dataset}: corpus row {row_index} unexpectedly has {present} "
                "while its identity rule is 'row_idx'; re-check the data version"
            )
        return str(row_index)
    if field not in row:
        raise ValueError(
            f"{dataset}: corpus row {row_index} has no {field!r} but that is its "
            "declared identity field; upstream data shape changed"
        )
    return str(row[field])


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
