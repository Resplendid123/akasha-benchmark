"""通过只读 PostgreSQL 查询追踪原文、编译产物、检索块和图关系。"""

from __future__ import annotations

from typing import Any
from uuid import UUID

# artifact：一个原始 page 贡献给了哪些编译产物。
# page_type 取 entity / source_summary；canonical_key 是实体合并的键。
ARTIFACTS_OF_SOURCE = """
SELECT DISTINCT kp.id, kp.title, kp.page_type
FROM knowledge_page_sources kps
JOIN knowledge_pages kp ON kp.id = kps.knowledge_page_id
WHERE kps.source_page_id = %(page)s
ORDER BY kp.title
"""

# 参与召回的文本。换 embedding 后旧 chunk 的 embedding_profile 对不上，召回不到。
CHUNKS_OF_ARTIFACTS = """
SELECT kc.id, kc.knowledge_page_id, kp.title, kc.chunk_role,
       kc.retrieval_channel, kc.embedding_profile, kc.text
FROM knowledge_chunks kc
JOIN knowledge_pages kp ON kp.id = kc.knowledge_page_id
WHERE kc.knowledge_page_id = ANY(%(ids)s)
ORDER BY kp.title, kc.chunk_role
"""

# 原文。**不参与召回**，只在引用解析时提供证据窗口。
SOURCE_CHUNKS = """
SELECT ksc.id, ksc.source_page_id, ksc.text
FROM knowledge_source_chunks ksc
WHERE ksc.source_page_id = %(page)s
ORDER BY ksc.id
"""

# 图边。relation 是自由生成的，取值不成枚举，所以只如实列出、不按类型遍历。
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

# 一篇源页面只有在仍有效的编译产物至少生成一个可检索 chunk 时才算编译成功。
# DISTINCT 避免同一源页面生成多个 artifact / chunk 后被重复计数。
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
    """没配只读数据库，或 psycopg 没装。"""


class BadPageId(ValueError):
    """page_id 不是合法的 UUID。让路由回 400，不把 Postgres 的原始错误文本透出去。"""


def _rows(cursor) -> list[dict[str, Any]]:
    columns = [c.name for c in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


class LineageReader:
    """只读地走血缘链路。每次调用开一个只读事务。"""

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
        except ModuleNotFoundError as exc:  # pragma: no cover - 依赖已在 pyproject 里
            raise LineageUnavailable(
                "psycopg is not installed; run `uv sync`"
            ) from exc
        # read_only 由连接层保证，不靠「只写了 SELECT」这种约定。
        return psycopg.connect(self.database_url, autocommit=False)

    def compiled_source_page_ids(self, page_ids: list[str]) -> set[str]:
        """批量返回真正生成了可检索产物的源页面 ID。"""
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
            # 不把 DSN 或 PG 原始错误透给接口；列表页把它显示为「未知」。
            raise LineageUnavailable(
                f"PostgreSQL compilation count unavailable: {type(exc).__name__}"
            ) from exc

    @staticmethod
    def _check_page_id(page_id: str) -> str:
        """在打到数据库之前校验 UUID，避免把 Postgres 的错误文本连参数值一起透出。"""
        try:
            UUID(page_id)
        except (ValueError, AttributeError, TypeError) as exc:
            raise BadPageId(f"{page_id!r} is not a valid page id (expected a UUID)") from exc
        return page_id

    def lineage(self, page_id: str, *, chunk_chars: int = 2000) -> dict[str, Any]:
        """一条完整链路：artifact -> chunk -> 图边 -> 原文。跨五张表六跳。"""
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
                    for chunk in chunks:
                        chunk["text"] = (chunk["text"] or "")[:chunk_chars]
                    cursor.execute(EDGES_OF_ARTIFACTS, {"ids": ids})
                    edges = _rows(cursor)

                cursor.execute(SOURCE_CHUNKS, {"page": page_id})
                source_chunks = _rows(cursor)
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
