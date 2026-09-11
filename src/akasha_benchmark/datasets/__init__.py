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
    make_sample_id,
)
from .registry import ADAPTER_CLASSES, DATASET_NAMES, all_adapters, get_adapter
from .resolver import ResolvedDataset, normalized_dir, repo_relative, resolve, subset_dir

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
    "all_adapters",
    "get_adapter",
    "load_corpus",
    "make_sample_id",
    "normalized_dir",
    "repo_relative",
    "resolve",
    "subset_dir",
]
