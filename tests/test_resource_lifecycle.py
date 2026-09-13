"""启动配置、路由连接和归一化事务的回归测试。"""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from akasha_benchmark import normalize
from akasha_benchmark.datasets.hotpotqa import HotpotQAAdapter
from akasha_benchmark.store import connect, repo
from akasha_benchmark.store.migrate import migrate
from akasha_platform.api._common import db, writable
from akasha_platform.main import create_app, main
from akasha_platform.settings import Settings


def test_reload_factory_uses_cli_database(tmp_path, monkeypatch):
    path = tmp_path / "selected.db"
    migrate(path, verbose=False)
    for key in ("HOST", "PORT", "DB", "AUTH_TOKEN"):
        monkeypatch.setenv(f"AKASHA_PLATFORM_{key}", "" if key == "AUTH_TOKEN" else {
            "HOST": "127.0.0.1", "PORT": "8848", "DB": str(tmp_path / "other.db"),
        }[key])

    def run(app, **kwargs):
        assert kwargs["factory"] and kwargs["reload"]
        assert create_app().state.settings.db_path == path

    monkeypatch.setattr("uvicorn.run", run)
    assert main(["--db", str(path), "--reload"]) == 0


def test_request_connections_close_and_failed_writes_roll_back(tmp_path):
    path = tmp_path / "test.db"
    migrate(path, verbose=False)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=Settings(db_path=path))))
    with writable(request) as connection:
        connection.execute("CREATE TABLE probe (value TEXT)")
        connection.execute("INSERT INTO probe VALUES ('kept')")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with pytest.raises(RuntimeError), writable(request) as connection:
        connection.execute("DELETE FROM probe")
        raise RuntimeError("abort")
    with db(request) as connection:
        assert connection.execute("SELECT value FROM probe").fetchone()[0] == "kept"
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_failed_corpus_replacement_preserves_previous_normalization(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    migrate(path, verbose=False)
    monkeypatch.setattr(HotpotQAAdapter, "expected_qa_rows", lambda self: 1)
    qa = tmp_path / "hotpotqa.json"
    row = {"_id": "one", "question": "original", "answer": "answer", "supporting_facts": [["Title", 0]]}
    qa.write_text(json.dumps([row]))
    (tmp_path / "hotpotqa_corpus.json").write_text(json.dumps([{"idx": 1, "title": "Title", "text": "body"}]))
    connection = connect(path)
    try:
        normalize.normalize_dataset(connection, "hotpotqa", tmp_path)
        previous = repo.get_dataset(connection, "hotpotqa")
        qa.write_text(json.dumps([{**row, "question": "changed"}]))

        def fail(connection, dataset, docs):
            connection.execute("DELETE FROM corpus_doc WHERE dataset = ?", (dataset,))
            raise RuntimeError("corpus write failed")

        monkeypatch.setattr(repo, "replace_corpus", fail)
        with pytest.raises(RuntimeError, match="corpus write failed"):
            normalize.normalize_dataset(connection, "hotpotqa", tmp_path)
        assert repo.get_dataset(connection, "hotpotqa") == previous
        assert repo.samples_of(connection, "hotpotqa")[0]["question"] == "original"
        assert len(repo.corpus_of(connection, "hotpotqa")) == 1
    finally:
        connection.close()
