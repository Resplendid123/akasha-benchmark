"""数据集适配器注册表。"""

from __future__ import annotations

from .base import DatasetAdapter
from .hotpotqa import HotpotQAAdapter
from .itfaq import ITFaqAdapter
from .musique import MusiqueAdapter
from .narrativeqa import NarrativeQAAdapter
from .two_wiki import TwoWikiMultihopQAAdapter

ADAPTER_CLASSES: tuple[type[DatasetAdapter], ...] = (
    HotpotQAAdapter,
    TwoWikiMultihopQAAdapter,
    MusiqueAdapter,
    NarrativeQAAdapter,
    ITFaqAdapter,
)

_LOOKUP = {cls.name: cls for cls in ADAPTER_CLASSES}
DATASET_NAMES: tuple[str, ...] = tuple(_LOOKUP)


def get_adapter(name: str) -> DatasetAdapter:
    """按规范数据集名取一个新的适配器实例。"""
    try:
        return _LOOKUP[name]()
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}. Known: {', '.join(sorted(_LOOKUP))}"
        ) from None


def all_adapters() -> tuple[DatasetAdapter, ...]:
    return tuple(cls() for cls in ADAPTER_CLASSES)
