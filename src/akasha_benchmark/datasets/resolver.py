"""数据集名 -> (适配器, 文件路径)。

目录布局收在这里一处，好处是脚本只接 ``--dataset hotpotqa`` 而不是裸路径，
而且文件缺失时给的是一句清楚的提示，不是循环深处冒出来的 FileNotFoundError。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import DatasetAdapter
from .registry import get_adapter

# 本文件位于 src/akasha_benchmark/datasets/，往上三层是仓库根。
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_DATA_DIR = REPO_ROOT / "data"


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

    missing = [str(p) for p in (qa_path, corpus_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{adapter.name}: missing {missing}. "
            "Run `uv run python scripts/download_datasets.py` first."
        )
    return ResolvedDataset(adapter=adapter, qa_path=qa_path, corpus_path=corpus_path)


def repo_relative(path: Path) -> str:
    """manifest 里记录路径用的形式：仓库内的写成相对路径，分隔符统一为 ``/``。

    绝对路径会把某一台机器的目录结构焊进产物里 —— 换机器、换平台，
    或者别人拿到这份 manifest，记下来的路径都是错的，也没法逐字段比对两次产出。
    仓库外的路径（例如测试用的临时目录）没有有意义的相对表示，保持绝对形式。
    """
    resolved = Path(path).resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return resolved.as_posix()


def normalized_dir(dataset: str, data_dir: Path | None = None) -> Path:
    """归一化产出目录。"""
    return (data_dir or DEFAULT_DATA_DIR) / "normalized" / dataset


def subset_dir(run_id: str, dataset: str, data_dir: Path | None = None) -> Path:
    """子集产出目录，按 run_id 隔离。"""
    return (data_dir or DEFAULT_DATA_DIR) / "subsets" / run_id / dataset
