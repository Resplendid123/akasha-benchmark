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

    directory = tmp_path / "dataset"
    directory.mkdir()
    (directory / "hotpotqa.json").write_text(json.dumps(QA_ROWS), encoding="utf-8")
    (directory / "hotpotqa_corpus.json").write_text(json.dumps(CORPUS_ROWS), encoding="utf-8")
    return directory


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

    directory = tmp_path / "itfaq"
    directory.mkdir()
    (directory / "itfaq.json").write_text(
        json.dumps(ITFAQ_QA_ROWS, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "itfaq_corpus.json").write_text(
        json.dumps(ITFAQ_CORPUS_ROWS, ensure_ascii=False), encoding="utf-8"
    )
    return directory


def _normalize(db, name: str, directory: Path, monkeypatch):
    from akasha_benchmark.datasets.registry import get_adapter
    from akasha_benchmark.stages import normalize

    monkeypatch.setattr(type(get_adapter(name)), "expected_qa_rows", lambda self: None)
    normalize.normalize_dataset(db, name, None, directory)
    assert data_store.get_dataset(db, name) is not None
    return db


@pytest.fixture
def itfaq_normalized(db, itfaq_dataset_dir: Path, monkeypatch):
    return _normalize(db, "itfaq", itfaq_dataset_dir, monkeypatch)


@pytest.fixture
def normalized(db, dataset_dir: Path, monkeypatch):
    return _normalize(db, "hotpotqa", dataset_dir, monkeypatch)


def make_compile_run(connection, run_id="r", datasets=("d",), **overrides) -> int:
    params = {"seed": 1, "qa_limit": 2, "negatives_ratio": 1.0} | overrides
    return compile_store.create_compile_run(
        connection, run_id=run_id, datasets=list(datasets), **params
    )


@pytest.fixture
def compile_id(db):
    return make_compile_run(db)


@pytest.fixture
def query_id(db, compile_id):
    return query_store.create_query_run(
        db, name="q", compile_id=compile_id, concurrency=1, model_configs={}
    )


@pytest.fixture
def eval_id(db, query_id):
    return eval_store.create_eval_run(
        db, name="e", query_id=query_id, ks=[2], metrics=["faithfulness"], judge_provider_id=None
    )
