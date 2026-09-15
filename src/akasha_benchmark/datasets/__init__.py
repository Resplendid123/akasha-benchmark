"""数据集适配器，以及各阶段共用的规范化模型。"""

from .base import DatasetAdapter
from .corpus import CorpusIndex, load_corpus
from .models import (
    CORPUS_ID_RULES,
    SAMPLE_ID_RULES,
    CanonicalSample,
    CorpusDoc,
    DataDependency,
    DependencyError,
    SubsetStrategy,
    make_sample_id,
)
from .registry import ADAPTER_CLASSES, DATASET_NAMES, all_adapters, get_adapter
from .resolver import ResolvedDataset, resolve

__all__ = [
    "ADAPTER_CLASSES",
    "CORPUS_ID_RULES",
    "DATASET_NAMES",
    "SAMPLE_ID_RULES",
    "CanonicalSample",
    "CorpusDoc",
    "CorpusIndex",
    "DataDependency",
    "DatasetAdapter",
    "DependencyError",
    "ResolvedDataset",
    "SubsetStrategy",
    "all_adapters",
    "get_adapter",
    "load_corpus",
    "make_sample_id",
    "resolve",
]
