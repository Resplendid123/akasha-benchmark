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
  AND kc.stale_at IS NULL
ORDER BY kp.title, kc.chunk_role
"""


SPACE_EMBEDDING_GENERATIONS = """
SELECT kc.embedding_model,
       kc.embedding_dimensions,
       kc.stale_at IS NULL AS active,
       COUNT(*)::bigint AS chunks,
       COUNT(kc.embedding)::bigint AS embedded,
       SUM(kc.embedding_total_tokens)::bigint AS total_tokens,
       SUM(kc.embedding_duration_ms)::bigint AS duration_ms_sum,
       AVG(kc.embedding_duration_ms)::double precision AS duration_ms_mean,
       MIN(kc.created_at) AS first_created_at,
       MAX(kc.created_at) AS last_created_at
FROM knowledge_chunks kc
WHERE kc.space_id = %(space)s
GROUP BY kc.embedding_model, kc.embedding_dimensions, kc.stale_at IS NULL
ORDER BY active DESC, last_created_at DESC
"""


SPACE_REBUILD_BATCHES = """
SELECT embedding_model,
       COUNT(*)::bigint AS batches,
       SUM(chunk_count)::bigint AS chunks,
       SUM(failed_count)::bigint AS failed,
       SUM(total_tokens)::bigint AS total_tokens,
       SUM(duration_ms)::bigint AS duration_ms_sum,
       MIN(created_at) AS first_batch_at,
       MAX(created_at) AS last_batch_at
FROM knowledge_embedding_rebuild_batches
WHERE space_id = %(space)s
  AND (%(since)s::timestamptz IS NULL OR created_at >= %(since)s::timestamptz)
GROUP BY embedding_model
ORDER BY last_batch_at DESC
"""


SERVER_NOW = "SELECT now()"


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


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _iso(value: Any) -> str | None:
    return None if value is None else getattr(value, "isoformat", lambda: str(value))()


def active_coverage(generations: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """活跃 chunk 里匹配目标模型的占比。重建原地改写同一批行，总数恒定。"""
    active = [entry for entry in generations if entry["active"]]
    total = sum(entry["chunks"] for entry in active)
    matched = sum(entry["chunks"] for entry in active if entry["model"] == model)
    return {
        "total": total,
        "matched": matched,
        "ready": bool(total) and matched == total,
        "models": sorted({str(entry["model"]) for entry in active if entry["model"]}),
    }


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

    def server_now(self) -> str:
        """取 PG 自己的时钟，用它当重建起点，免得本机与库有时差。"""
        try:
            with self._connect() as connection:
                connection.read_only = True
                with connection.cursor() as cursor:
                    cursor.execute(SERVER_NOW)
                    return _iso(cursor.fetchone()[0]) or ""
        except LineageUnavailable:
            raise
        except Exception as exc:
            raise LineageUnavailable(
                f"PostgreSQL clock unavailable: {type(exc).__name__}"
            ) from exc

    def space_rebuild_usage(
        self, space_id: str, *, since: str | None = None
    ) -> dict[str, Any]:
        """重建批次的真实用量：duration_ms 是整批并发 embed 的墙钟，可直接比并发。

        chunk 行上的 embedding_total_tokens/_duration_ms 只有编译路径写，重建不写，
        所以重建用量只能从这张批次表读。
        """
        space_id = self._check_id(space_id, "space id")
        try:
            with self._connect() as connection:
                connection.read_only = True
                with connection.cursor() as cursor:
                    cursor.execute(
                        SPACE_REBUILD_BATCHES, {"space": space_id, "since": since}
                    )
                    rows = _rows(cursor)
        except LineageUnavailable:
            raise
        except Exception as exc:
            raise LineageUnavailable(
                f"PostgreSQL rebuild batches unavailable: {type(exc).__name__}"
            ) from exc

        models = [
            {
                "model": row["embedding_model"],
                "batches": int(row["batches"] or 0),
                "chunks": int(row["chunks"] or 0),
                "failed": int(row["failed"] or 0),
                "total_tokens": _optional_int(row["total_tokens"]),
                "duration_ms_sum": _optional_int(row["duration_ms_sum"]),
                "first_batch_at": _iso(row["first_batch_at"]),
                "last_batch_at": _iso(row["last_batch_at"]),
            }
            for row in rows
        ]
        timestamps = [entry["last_batch_at"] for entry in models if entry["last_batch_at"]]
        return {
            "space_id": space_id,
            "models": models,
            "batches": sum(entry["batches"] for entry in models),
            "chunks": sum(entry["chunks"] for entry in models),
            "failed": sum(entry["failed"] for entry in models),
            "last_batch_at": max(timestamps, default=None),
        }

    def space_embedding_generations(self, space_id: str) -> dict[str, Any]:
        """按 space 聚合 chunk 的嵌入模型分布；用量两列只反映编译时，重建不写。"""
        space_id = self._check_id(space_id, "space id")
        try:
            with self._connect() as connection:
                connection.read_only = True
                with connection.cursor() as cursor:
                    cursor.execute(SPACE_EMBEDDING_GENERATIONS, {"space": space_id})
                    rows = _rows(cursor)
        except LineageUnavailable:
            raise
        except Exception as exc:
            raise LineageUnavailable(
                f"PostgreSQL embedding generations unavailable: {type(exc).__name__}"
            ) from exc

        generations = [
            {
                "model": row["embedding_model"],
                "dimensions": row["embedding_dimensions"],
                "active": bool(row["active"]),
                "chunks": int(row["chunks"] or 0),
                "embedded": int(row["embedded"] or 0),
                "total_tokens": _optional_int(row["total_tokens"]),
                "duration_ms_sum": _optional_int(row["duration_ms_sum"]),
                "duration_ms_mean": _optional_float(row["duration_ms_mean"]),
                "first_created_at": _iso(row["first_created_at"]),
                "last_created_at": _iso(row["last_created_at"]),
            }
            for row in rows
        ]
        return {"space_id": space_id, "generations": generations}

    @staticmethod
    def _check_id(value: str, label: str = "page id") -> str:

        try:
            UUID(value)
        except (ValueError, AttributeError, TypeError) as exc:
            raise BadPageId(f"{value!r} is not a valid {label} (expected a UUID)") from exc
        return value

    @classmethod
    def _check_page_id(cls, page_id: str) -> str:
        return cls._check_id(page_id)

    def lineage(self, page_id: str, *, chunk_chars: int | None = 2000) -> dict[str, Any]:

        page_id = self._check_page_id(page_id)
        with self._connect() as connection:
            connection.read_only = True
            with connection.cursor() as cursor:
                cursor.execute(ARTIFACTS_OF_SOURCE, {"page": page_id})
                artifacts = _rows(cursor)
                ids = [a["id"] for a in artifacts]

                chunks: list[dict[str, Any]] = []
                if ids:
                    cursor.execute(CHUNKS_OF_ARTIFACTS, {"ids": ids})
                    chunks = _rows(cursor)
                    if chunk_chars is not None:
                        for chunk in chunks:
                            chunk["text"] = (chunk["text"] or "")[:chunk_chars]

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
            "counts": {
                "artifacts": len(artifacts),
                "chunks": len(chunks),
                "source_chunks": len(source_chunks),
            },
        }
