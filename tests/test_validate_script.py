"""校验脚本：它现在核对的是**库里的产物**，不是磁盘上的 jsonl。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from akasha_benchmark.store import connect, repo
from akasha_benchmark.store.migrate import migrate

# 脚本在 scripts/ 下，不是包的一部分，所以按路径加载。
import importlib.util
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "validate_datasets", REPO_ROOT / "scripts" / "validate_datasets.py"
)
assert _spec and _spec.loader
validate_datasets = importlib.util.module_from_spec(_spec)
sys.modules["validate_datasets"] = validate_datasets
_spec.loader.exec_module(validate_datasets)


def test_script_exposes_a_db_backed_signature():
    """签名必须是 ``(dataset, dataset_dir, connection)``。"""
    parameters = validate_datasets.validate.__code__.co_varnames[:3]
    assert parameters == ("dataset", "dataset_dir", "connection")


def test_main_refuses_a_missing_database(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """库不存在时给一句清楚的提示并返回 1。"""
    assert validate_datasets.main(["--db", str(tmp_path / "absent.db")]) == 1
    assert "does not exist" in capsys.readouterr().err


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "t.db"
    migrate(path, verbose=False)
    return path


def _seed(connection: sqlite3.Connection, *, qa_sha256: str) -> None:
    repo.upsert_dataset(
        connection,
        name="hotpotqa",
        adapter="HotpotQAAdapter",
        adapter_version="1",
        provides=["gold_docs", "reference_answers"],
        identity_rules={},
        qa_path="dataset/hotpotqa.json",
        qa_sha256=qa_sha256,
        qa_rows=1,
        corpus_path="dataset/hotpotqa_corpus.json",
        corpus_sha256="1" * 64,
        corpus_rows=1,
        dedup_stats={},
        gold_count_distribution={},
        unique_question_texts=1,
    )
    connection.commit()


# 下面几条要真的跑一遍 validate()，所以需要原始数据文件在位。
# 缺文件时 skip 而不是失败：那是「没下载数据」，不是「代码坏了」。
DATASET_DIR = REPO_ROOT / "dataset"
needs_dataset = pytest.mark.skipif(
    not (DATASET_DIR / "hotpotqa.json").is_file(),
    reason="需要 dataset/hotpotqa.json（通过数据集页面下载）",
)


@needs_dataset
def test_missing_dataset_row_points_at_the_normalize_stage(db: Path):
    """库里没有这一行时，提示必须指向 normalize 阶段。"""
    connection = connect(db, read_only=True)
    try:
        report = validate_datasets.validate("hotpotqa", DATASET_DIR, connection)
    finally:
        connection.close()

    assert not report.ok
    assert any("not in the database" in error for error in report.errors)
    assert any("akasha_benchmark.normalize" in error for error in report.errors)


@needs_dataset
def test_upstream_hash_drift_is_reported(db: Path):
    """库里记的原始文件 sha256 与磁盘不一致时必须报错。"""
    connection = connect(db)
    try:
        _seed(connection, qa_sha256="9" * 64)
        report = validate_datasets.validate("hotpotqa", DATASET_DIR, connection)
    finally:
        connection.close()

    assert not report.ok
    assert any("changed since normalization" in error for error in report.errors)


@needs_dataset
def test_a_real_normalize_then_validate_round_trip_passes(db: Path):
    """归一化进库再校验，必须通过 —— 这是 `make offline` 那条链的核心一环。"""
    from akasha_benchmark.normalize import normalize_dataset

    connection = connect(db)
    try:
        result = normalize_dataset(connection, "hotpotqa", DATASET_DIR)
        assert result["samples"] == 1000
        report = validate_datasets.validate("hotpotqa", DATASET_DIR, connection)
    finally:
        connection.close()

    assert report.ok, report.errors
    # 那几条实测统计要在 notes 里，它们是人核对用的。
    assert any("gold_count_distribution={2: 1000}" in note for note in report.notes)
    assert any("provides=gold_docs,reference_answers" in note for note in report.notes)


@needs_dataset
def test_validate_detects_samples_deleted_from_the_database(db: Path):
    """库里的样本被删掉一部分时必须报错，不能因为「剩下的都对」就放行。"""
    from akasha_benchmark.normalize import normalize_dataset

    connection = connect(db)
    try:
        normalize_dataset(connection, "hotpotqa", DATASET_DIR)
        connection.execute("DELETE FROM sample WHERE sample_id IN (SELECT sample_id FROM sample LIMIT 5)")
        connection.commit()
        report = validate_datasets.validate("hotpotqa", DATASET_DIR, connection)
    finally:
        connection.close()

    assert not report.ok
    assert any("995 samples, re-derived 1000" in error for error in report.errors)
