"""共用夹具：一个建好表的临时库，以及一个装了假数据集的库。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from akasha_benchmark.store import connect, data_store, init_db

# 一份最小的假数据集：2 条 QA、4 篇语料，够跑通抽样与指标。
QA_ROWS = [
    {
        "_id": "q1",
        "question": "Who won the Grammy and the Emmy?",
        "answer": "Rita Moreno",
        "type": "bridge",
        "supporting_facts": [["Rita Moreno", 0], ["EGOT", 0]],
        "context": [["Rita Moreno", ["a"]], ["EGOT", ["b"]]],
    },
    {
        "_id": "q2",
        "question": "Which city hosts the festival?",
        "answer": "Venice",
        "type": "comparison",
        "supporting_facts": [["Venice", 0]],
        "context": [["Venice", ["c"]]],
    },
]

CORPUS_ROWS = [
    {"idx": 0, "title": "Rita Moreno", "text": "Rita Moreno won a Grammy and an Emmy award."},
    {"idx": 1, "title": "EGOT", "text": "EGOT means Emmy, Grammy, Oscar and Tony."},
    {"idx": 2, "title": "Venice", "text": "Venice hosts the film festival each year."},
    {"idx": 3, "title": "Distractor", "text": "Unrelated filler content for negatives."},
]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "bench.db"
    init_db(path)
    return path


@pytest.fixture
def db(db_path: Path):
    connection = connect(db_path)
    yield connection
    connection.close()


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """hotpotqa 形状的原始文件，行数与适配器预期不符，所以测试要绕开行数校验。"""
    directory = tmp_path / "dataset"
    directory.mkdir()
    (directory / "hotpotqa.json").write_text(json.dumps(QA_ROWS), encoding="utf-8")
    (directory / "hotpotqa_corpus.json").write_text(json.dumps(CORPUS_ROWS), encoding="utf-8")
    return directory


@pytest.fixture
def normalized(db, dataset_dir: Path, monkeypatch):
    """把假数据集归一化进库，返回连接。"""
    from akasha_benchmark.datasets.hotpotqa import HotpotQAAdapter
    from akasha_benchmark.stages import normalize

    # 真实适配器按锁定快照校验行数；假数据只有 2 行。
    monkeypatch.setattr(HotpotQAAdapter, "expected_qa_rows", lambda self: None)
    normalize.normalize_dataset(db, "hotpotqa", None, dataset_dir)
    assert data_store.get_dataset(db, "hotpotqa") is not None
    return db
