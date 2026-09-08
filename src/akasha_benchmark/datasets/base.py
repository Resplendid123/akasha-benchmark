"""适配器契约。

一个数据集一个类，各自声明自己的名字、别名、文件名和 capability。
分派**只**按数据集名字，绝不按「row 里有没有某个字段」来猜 ——
按字段存在性分派的话，数据集换个版本就会静默走错分支，
而且症状是一个看着挺合理的指标，不是一个报错。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .corpus import CorpusIndex
from .models import Capability, CanonicalSample


class DatasetAdapter(ABC):
    """把某个数据集的原始行转成 :class:`CanonicalSample`。"""

    name: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]] = ()
    qa_filename: ClassVar[str]
    corpus_filename: ClassVar[str]
    capabilities: ClassVar[frozenset[Capability]]
    version: ClassVar[str] = "1"

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    @abstractmethod
    def parse_row(
        self, row: dict[str, Any], row_index: int, corpus: CorpusIndex
    ) -> CanonicalSample:
        """转换一行原始 QA。遇到任何意外结构都报错。"""

    def expected_qa_rows(self) -> int | None:
        """在锁定的数据快照上实测的行数，校验时比对。"""
        return None
