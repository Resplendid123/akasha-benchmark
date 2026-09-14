"""适配器契约。

一个数据集一个类，各自声明名字、别名、文件名与 provides。
分派只按数据集名字，不按「row 里有没有某个字段」猜 —— 后者在数据换版时
会静默走错分支，症状是一个看着合理的指标而不是报错。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .corpus import CorpusIndex
from .models import CanonicalSample, DataDependency


class DatasetAdapter(ABC):
    """把某个数据集的原始行转成 :class:`CanonicalSample`。"""

    name: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]] = ()
    qa_filename: ClassVar[str]
    corpus_filename: ClassVar[str]
    # 这个数据集拥有哪些标注，决定哪些指标算得出来。
    provides: ClassVar[frozenset[DataDependency]]
    version: ClassVar[str] = "1"

    def has(self, dependency: DataDependency) -> bool:
        return dependency in self.provides

    def identity_rules(self) -> dict[str, str]:
        """这个数据集怎么给样本与语料行赋身份，取自 :mod:`.models` 的两张规则表。"""
        from .models import CORPUS_ID_RULES, SAMPLE_ID_RULES

        return {
            "sample_id": SAMPLE_ID_RULES[self.name],
            "corpus_doc_id": CORPUS_ID_RULES[self.name],
        }

    @abstractmethod
    def parse_row(
        self, row: dict[str, Any], row_index: int, corpus: CorpusIndex
    ) -> CanonicalSample:
        """转换一行原始 QA，遇到意外结构就报错。``corpus`` 供解析 gold 反查。"""

    def expected_qa_rows(self) -> int | None:
        """锁定快照上的行数，归一化时比对。``None`` 表示不校验。"""
        return None
