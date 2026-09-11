"""血缘：从语料文档一路走到编译产物、chunk、图边与原文。**只读 Postgres。**

这条链路是 PLAN.md §12.9 在 run001 的真实库上逐跳走通的，表名列名均已核对 ——
不是照 Akasha 文档抄的。两处与先前假设不符，已在那里纠正：

* ``knowledge_chunks`` 的外键是 ``knowledge_page_id``，**不是** ``source_page_id``
* ``knowledge_page_sources`` **直接带** ``source_page_id``，不必经
  ``knowledge_sources`` 中转

```
pages.id                      ← page_map 里的 page_id（导入接口返回）
  ↓ knowledge_page_sources.source_page_id
knowledge_pages.id            ← 编译产出的 artifact，不是原始 page
  ↓ knowledge_chunks.knowledge_page_id          参与召回的文本
  ↓ knowledge_graph_edges.from/to_knowledge_page_id   图边
knowledge_source_chunks.source_page_id          原文，不参与召回
```

**方向单向**：这个库我们从不写。所有连接都以只读事务打开。
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

# artifact：一个原始 page 贡献给了哪些编译产物。
# page_type 取值 entity / source_summary；canonical_key 是实体合并的键。
ARTIFACTS_OF_SOURCE = """
SELECT DISTINCT kp.id, kp.title, kp.page_type, kp.compile_scope,
       kp.canonical_key, kp.stale_at
FROM knowledge_page_sources kps
JOIN knowledge_pages kp ON kp.id = kps.knowledge_page_id
WHERE kps.source_page_id = %(page)s
ORDER BY kp.title
"""

# 参与召回的文本。embedding_profile 是那条静默失效的线索：换 embedding 后
# 旧 chunk 的 profile 对不上，这些 chunk 永远召回不到。
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

# 图边。relation 是自由生成的（555 条边散在 377 种取值上，createdBy 与
# created_by 并存），所以图遍历无法按关系类型做 —— 这里只如实列出。
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

# snippets[].id 是裸 UUID 不带类型前缀，且 snippet 里没有 kind，所以光看响应
# 分不出这条来自原文块还是编译产物。用 knowledge_chunks.id 反查补上（§12.10）。
CHUNK_KINDS = """
SELECT kc.id, kc.chunk_role, kc.retrieval_channel, kp.page_type, kp.title
FROM knowledge_chunks kc
JOIN knowledge_pages kp ON kp.id = kc.knowledge_page_id
WHERE kc.id = ANY(%(ids)s)
"""


class LineageUnavailable(RuntimeError):
    """没配只读数据库，或 psycopg 没装。"""


class BadPageId(ValueError):
    """page_id 不是合法的 UUID。

    单独一个异常类型，是为了让路由回 400 而不是 500 —— 后者会把 Postgres 的
    原始错误文本（含参数值）透到响应里，而那是个信息泄露面。
    """


def _rows(cursor) -> list[dict[str, Any]]:
    columns = [c.name for c in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


class LineageReader:
    """只读地走血缘链路。每次调用开一个只读事务。"""

    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise LineageUnavailable(
                "no database_url configured. Set AKASHA_DATABASE_URL to enable the "
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
        # read_only 由连接层保证，不靠「我们只写了 SELECT」这种约定。
        return psycopg.connect(self.database_url, autocommit=False)

    @staticmethod
    def _check_page_id(page_id: str) -> str:
        """page_id 必须是 UUID。

        在打到数据库**之前**挡住，否则 Postgres 会抛
        ``invalid input syntax for type uuid``，而那条错误文本里带着参数值,
        直接透给调用方就是一个信息泄露面。
        """
        try:
            UUID(page_id)
        except (ValueError, AttributeError, TypeError) as exc:
            raise BadPageId(f"{page_id!r} is not a valid page id (expected a UUID)") from exc
        return page_id

    def artifacts(self, page_id: str) -> list[dict[str, Any]]:
        """一个原始 page 编成了哪些 artifact。"""
        page_id = self._check_page_id(page_id)
        with self._connect() as connection:
            connection.read_only = True
            with connection.cursor() as cursor:
                cursor.execute(ARTIFACTS_OF_SOURCE, {"page": page_id})
                return _rows(cursor)

    def lineage(self, page_id: str, *, chunk_chars: int = 2000) -> dict[str, Any]:
        """一条完整链路：artifact -> chunk -> 图边 -> 原文。

        这是「必须一屏走完」的那个视图（§12.9）—— 跨五张表六跳，
        否则归因就得像那次一样手写 SQL。
        """
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
            # 参与召回的文本。
            "chunks": chunks,
            # 原文，不参与召回 —— 这个区别是 §0.3 那条架构论断的全部依据。
            "source_chunks": source_chunks,
            "edges": edges,
            "counts": {
                "artifacts": len(artifacts),
                "chunks": len(chunks),
                "source_chunks": len(source_chunks),
                "edges": len(edges),
            },
        }

    def chunk_kinds(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """给 snippet 的裸 UUID 补上 page_type 与 chunk_role。"""
        if not chunk_ids:
            return {}
        chunk_ids = [self._check_page_id(cid) for cid in chunk_ids]
        with self._connect() as connection:
            connection.read_only = True
            with connection.cursor() as cursor:
                cursor.execute(CHUNK_KINDS, {"ids": chunk_ids})
                return {row["id"]: row for row in _rows(cursor)}
