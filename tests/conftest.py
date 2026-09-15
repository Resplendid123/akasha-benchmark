"""共用夹具：一个建好表的临时库，以及一个装了假数据集的库。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from akasha_benchmark.store import (
    compile_store,
    connect,
    data_store,
    eval_store,
    init_db,
    query_store,
)

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


# itfaq 形状：QA 只有 {id, question, answer}，语料 {id, title, text} 且正文自带 H1。
ITFAQ_QA_ROWS = [
    {"id": "qa_001", "question": "怎么申请手机？", "answer": "走 IT 设备申请流程。"},
    {"id": "qa_002", "question": "显卡坏了怎么换？", "answer": "到 IT 现场登记后更换。"},
    {"id": "qa_003", "question": "邮箱客户端登不上？", "answer": "改用客户端专用密码。"},
]

ITFAQ_CORPUS_ROWS = [
    {"id": "doc_001", "title": "设备申请说明", "text": "# 设备申请说明\n\n填写申请单即可。"},
    {"id": "doc_002", "title": "邮箱配置说明", "text": "# 邮箱配置说明\n\nIMAP 用 993 端口。"},
]


@pytest.fixture
def itfaq_dataset_dir(tmp_path: Path) -> Path:
    """itfaq 形状的原始文件。行数与适配器预期不符，所以测试要绕开行数校验。"""
    directory = tmp_path / "itfaq"
    directory.mkdir()
    (directory / "itfaq.json").write_text(
        json.dumps(ITFAQ_QA_ROWS, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "itfaq_corpus.json").write_text(
        json.dumps(ITFAQ_CORPUS_ROWS, ensure_ascii=False), encoding="utf-8"
    )
    return directory


@pytest.fixture
def itfaq_normalized(db, itfaq_dataset_dir: Path, monkeypatch):
    """把假 itfaq 归一化进库，返回连接。"""
    from akasha_benchmark.datasets.itfaq import ITFaqAdapter
    from akasha_benchmark.stages import normalize

    monkeypatch.setattr(ITFaqAdapter, "expected_qa_rows", lambda self: None)
    normalize.normalize_dataset(db, "itfaq", None, itfaq_dataset_dir)
    return db


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


@pytest.fixture
def compile_id(db):
    return compile_store.create_compile_run(
        db, run_id="r", datasets=["d"], seed=1, qa_limit=2, negatives_ratio=1.0
    )


@pytest.fixture
def query_id(db, compile_id):
    return query_store.create_query_run(
        db, name="q", compile_id=compile_id, score_threshold=None, concurrency=1, model_configs={}
    )


@pytest.fixture
def eval_id(db, query_id):
    return eval_store.create_eval_run(
        db, name="e", query_id=query_id, ks=[2], metrics=["faithfulness"], judge_provider_id=None
    )
