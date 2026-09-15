"""各适配器的名称与别名注册表。"""

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

DATASET_NAMES: tuple[str, ...] = tuple(cls.name for cls in ADAPTER_CLASSES)


def _build_lookup() -> dict[str, type[DatasetAdapter]]:
    """名字和别名统一转小写建表；两个适配器抢同一个键就直接报错。"""
    lookup: dict[str, type[DatasetAdapter]] = {}
    for cls in ADAPTER_CLASSES:
        for key in (cls.name, *cls.aliases):
            normalized = key.lower()
            if normalized in lookup:
                raise RuntimeError(
                    f"dataset key {normalized!r} claimed by both "
                    f"{lookup[normalized].__name__} and {cls.__name__}"
                )
            lookup[normalized] = cls
    return lookup


_LOOKUP = _build_lookup()


def get_adapter(name: str) -> DatasetAdapter:
    """按数据集名或别名取一个新的适配器实例。"""
    try:
        return _LOOKUP[name.strip().lower()]()
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}. Known: {', '.join(sorted(_LOOKUP))}"
        ) from None


def all_adapters() -> tuple[DatasetAdapter, ...]:
    return tuple(cls() for cls in ADAPTER_CLASSES)
