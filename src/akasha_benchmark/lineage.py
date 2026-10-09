from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID


ARTIFACTS_OF_SOURCE = """
SELECT DISTINCT kp.id, kp.title, kp.page_type
FROM knowledge_page_sources kps
JOIN knowledge_pages kp ON kp.id = kps.knowledge_page_id
WHERE kps.source_page_id = %(page)s
ORDER BY kp.title
"""


CHUNKS_OF_ARTIFACTS = """
SELECT kc.id, kc.knowledge_page_id, kp.title, kc.chunk_role,
       kc.retrieval_channel, kc.embedding_profile, kc.text
FROM knowledge_chunks kc
JOIN knowledge_pages kp ON kp.id = kc.knowledge_page_id
WHERE kc.knowledge_page_id = ANY(%(ids)s)
ORDER BY kp.title, kc.chunk_role
"""


SOURCE_CHUNKS = """
SELECT ksc.id, ksc.source_page_id, ksc.text
FROM knowledge_source_chunks ksc
WHERE ksc.source_page_id = %(page)s
ORDER BY ksc.id
"""

QUERY_AUDITS = """
SELECT DISTINCT ON (query_hash) query_hash, metadata
FROM knowledge_query_audit
WHERE query_hash = ANY(%(hashes)s)
  AND (%(since)s::timestamptz IS NULL OR created_at >= %(since)s::timestamptz)
ORDER BY query_hash, created_at DESC
"""


EDGES_OF_ARTIFACTS = """
SELECT e.id, e.relation, e.stale_at,
       e.from_knowledge_page_id, f.title AS from_title, f.canonical_key AS from_key,
       e.to_knowledge_page_id,   t.title AS to_title,   t.canonical_key AS to_key
FROM knowledge_graph_edges e
JOIN knowledge_pages f ON f.id = e.from_knowledge_page_id
JOIN knowledge_pages t ON t.id = e.to_knowledge_page_id
WHERE e.from_knowledge_page_id = ANY(%(ids)s)
   OR e.to_knowledge_page_id = ANY(%(ids)s)
ORDER BY e.relation
"""


COMPILED_SOURCE_PAGES = """
SELECT DISTINCT kps.source_page_id
FROM knowledge_page_sources kps
JOIN knowledge_pages kp ON kp.id = kps.knowledge_page_id
WHERE kps.source_page_id = ANY(%(pages)s)
  AND kp.stale_at IS NULL
  AND EXISTS (
      SELECT 1
      FROM knowledge_chunks kc
      WHERE kc.knowledge_page_id = kp.id
  )
"""


class LineageUnavailable(RuntimeError):
    pass


class BadPageId(ValueError):
    pass


def _rows(cursor) -> list[dict[str, Any]]:
    columns = [c.name for c in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


class LineageReader:
    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise LineageUnavailable(
                "no database_url configured. Fill database_url in the settings view to enable the "
                "lineage and diff views; every other view works without it."
            )
        self.database_url = database_url

    def _connect(self):
        try:
            import psycopg
        except ModuleNotFoundError as exc:
            raise LineageUnavailable(
                "psycopg is not installed; run `uv sync`"
            ) from exc

        return psycopg.connect(self.database_url, autocommit=False)

    def compiled_source_page_ids(self, page_ids: list[str]) -> set[str]:

        valid: list[UUID] = []
        for page_id in dict.fromkeys(page_ids):
            try:
                valid.append(UUID(page_id))
            except (ValueError, AttributeError, TypeError):
                continue
        if not valid:
            return set()

        try:
            with self._connect() as connection:
                connection.read_only = True
                with connection.cursor() as cursor:
                    cursor.execute(COMPILED_SOURCE_PAGES, {"pages": valid})
                    return {str(row[0]) for row in cursor.fetchall()}
        except LineageUnavailable:
            raise
        except Exception as exc:

            raise LineageUnavailable(
                f"PostgreSQL compilation count unavailable: {type(exc).__name__}"
            ) from exc

    def query_audits(
        self, questions: list[str], *, since: str | None = None
    ) -> dict[str, dict[str, Any]]:

        hashes = [
            "sha256:" + hashlib.sha256(question.encode("utf-8")).hexdigest()
            for question in dict.fromkeys(questions)
        ]
        if not hashes:
            return {}
        try:
            with self._connect() as connection:
                connection.read_only = True
                with connection.cursor() as cursor:
                    cursor.execute(QUERY_AUDITS, {"hashes": hashes, "since": since})
                    return {
                        str(row[0]): row[1] if isinstance(row[1], dict) else {}
                        for row in cursor.fetchall()
                    }
        except LineageUnavailable:
            raise
        except Exception as exc:
            raise LineageUnavailable(
                f"PostgreSQL query audit unavailable: {type(exc).__name__}"
            ) from exc

    @staticmethod
    def _check_page_id(page_id: str) -> str:

        try:
            UUID(page_id)
        except (ValueError, AttributeError, TypeError) as exc:
            raise BadPageId(f"{page_id!r} is not a valid page id (expected a UUID)") from exc
        return page_id

    def lineage(self, page_id: str, *, chunk_chars: int | None = 2000) -> dict[str, Any]:

        page_id = self._check_page_id(page_id)
        with self._connect() as connection:
            connection.read_only = True
            with connection.cursor() as cursor:
                cursor.execute(ARTIFACTS_OF_SOURCE, {"page": page_id})
                artifacts = _rows(cursor)
                ids = [a["id"] for a in artifacts]

                chunks: list[dict[str, Any]] = []
                edges: list[dict[str, Any]] = []
                if ids:
                    cursor.execute(CHUNKS_OF_ARTIFACTS, {"ids": ids})
                    chunks = _rows(cursor)
                    if chunk_chars is not None:
                        for chunk in chunks:
                            chunk["text"] = (chunk["text"] or "")[:chunk_chars]
                    cursor.execute(EDGES_OF_ARTIFACTS, {"ids": ids})
                    edges = _rows(cursor)

                cursor.execute(SOURCE_CHUNKS, {"page": page_id})
                source_chunks = _rows(cursor)
                if chunk_chars is not None:
                    for chunk in source_chunks:
                        chunk["text"] = (chunk["text"] or "")[:chunk_chars]

        return {
            "source_page_id": page_id,
            "artifacts": artifacts,
            "chunks": chunks,
            "source_chunks": source_chunks,
            "edges": edges,
            "counts": {
                "artifacts": len(artifacts),
                "chunks": len(chunks),
                "source_chunks": len(source_chunks),
                "edges": len(edges),
            },
        }
