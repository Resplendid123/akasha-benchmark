from __future__ import annotations

from akasha_benchmark.lineage import COMPILED_SOURCE_PAGES, LineageReader


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
