from __future__ import annotations

import pytest

from akasha_benchmark.lineage import (
    CHUNKS_OF_ARTIFACTS,
    COMPILED_SOURCE_PAGES,
    SPACE_EMBEDDING_GENERATIONS,
    BadPageId,
    LineageReader,
    active_coverage,
)


def test_compiled_source_pages_are_queried_in_one_batch(monkeypatch):
    page_a = "00000000-0000-0000-0000-000000000001"
    page_b = "00000000-0000-0000-0000-000000000002"
    seen: dict = {}

    class Database:
        read_only = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return self

        def execute(self, sql, params):
            seen.update({"sql": sql, "params": params})

        def fetchall(self):
            return [(page_a,), (page_a,), (page_b,)]

    reader = LineageReader("postgresql://configured")
    monkeypatch.setattr(reader, "_connect", Database)

    result = reader.compiled_source_page_ids([page_a, page_a, "invalid", page_b])

    assert result == {page_a, page_b}
    assert seen["sql"] == COMPILED_SOURCE_PAGES
    assert [str(page) for page in seen["params"]["pages"]] == [page_a, page_b]


def test_compiled_chunks_exclude_stale_generations():
    """重建会插新代并把旧代标 stale；不过滤会让扩写比按多代重复计数。"""
    assert "kc.stale_at IS NULL" in CHUNKS_OF_ARTIFACTS


def test_space_embedding_generations_groups_by_model_and_liveness(monkeypatch):
    space = "00000000-0000-0000-0000-00000000000a"
    seen: dict = {}

    class Database:
        read_only = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return self

        def execute(self, sql, params):
            seen.update({"sql": sql, "params": params})

        @property
        def description(self):
            return [
                type("Column", (), {"name": name})
                for name in (
                    "embedding_model",
                    "embedding_dimensions",
                    "active",
                    "chunks",
                    "embedded",
                    "total_tokens",
                    "duration_ms_sum",
                    "duration_ms_mean",
                    "first_created_at",
                    "last_created_at",
                )
            ]

        def fetchall(self):
            return [
                ("e2", 1024, True, 4, 4, 800, 2400, 600.0, None, None),
                ("e1", 768, False, 4, 4, None, None, None, None, None),
            ]

    reader = LineageReader("postgresql://configured")
    monkeypatch.setattr(reader, "_connect", Database)

    report = reader.space_embedding_generations(space)

    assert seen["sql"] == SPACE_EMBEDDING_GENERATIONS
    assert seen["params"] == {"space": space}
    active, stale = report["generations"]
    assert (active["model"], active["active"], active["total_tokens"]) == ("e2", True, 800)
    assert (stale["model"], stale["active"], stale["total_tokens"]) == ("e1", False, None)


def test_space_embedding_generations_rejects_non_uuid():
    with pytest.raises(BadPageId, match="space id"):
        LineageReader("postgresql://configured").space_embedding_generations("space-1")


def test_active_coverage_ignores_stale_generations():
    generations = [
        {"model": "e2", "active": True, "chunks": 3},
        {"model": "e1", "active": True, "chunks": 1},
        {"model": "e1", "active": False, "chunks": 4},
    ]

    assert active_coverage(generations, "e2") == {
        "total": 4,
        "matched": 3,
        "ready": False,
        "models": ["e1", "e2"],
    }
    assert active_coverage([{"model": "e2", "active": True, "chunks": 4}], "e2")["ready"] is True
    assert active_coverage([], "e2")["ready"] is False
