"""数据集名 -> (适配器, 文件路径)。

目录布局收在这里一处，文件缺失时给的是一句清楚的提示，
而不是循环深处冒出来的 FileNotFoundError。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import DatasetAdapter
from .registry import get_adapter

# 本文件位于 src/akasha_benchmark/datasets/，往上三层是仓库根。
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"


@dataclass(frozen=True)
class ResolvedDataset:
    adapter: DatasetAdapter
    qa_path: Path
    corpus_path: Path

    @property
    def name(self) -> str:
        return self.adapter.name


def resolve(name: str, dataset_dir: Path | None = None) -> ResolvedDataset:
    root = dataset_dir or DEFAULT_DATASET_DIR
    adapter = get_adapter(name)
    qa_path = root / adapter.qa_filename
    corpus_path = root / adapter.corpus_filename

    missing = [p.name for p in (qa_path, corpus_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{adapter.name}: 缺少原始文件 {missing}，请在数据集页先下载。"
        )
    return ResolvedDataset(adapter=adapter, qa_path=qa_path, corpus_path=corpus_path)
