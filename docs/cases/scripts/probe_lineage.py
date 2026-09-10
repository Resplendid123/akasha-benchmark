"""走完一条样本的血缘链路：原文 page -> artifact -> chunk -> 图边 -> 原文块。

案例见 ../recall-miss-compiled-away.md。默认查的是那一案的两篇 gold：
6365（Dee Does Broadway，漏召回）与 6369（Cyndi Lauper，召回第一名）。

链路（列名均已核对）：
    pages.id -> knowledge_page_sources.source_page_id
             -> knowledge_pages.id
             -> knowledge_chunks.knowledge_page_id
             -> knowledge_graph_edges.from/to_knowledge_page_id
    knowledge_source_chunks.source_page_id  = 原文，不参与召回

只读 SQL。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import psycopg

from akasha_benchmark.config import load_config

PAGE_6365 = "01a089d1-fdc7-7e19-9d4b-05109dc38745"  # Dee Does Broadway
PAGE_6369 = "01a089d2-07d5-72d5-bf7e-a38f23fe2c51"  # Cyndi Lauper

ARTIFACTS_OF_SOURCE = """
SELECT DISTINCT kp.id, kp.title, kp.page_type, kp.compile_scope, kp.canonical_key, kp.stale_at
FROM knowledge_page_sources kps
JOIN knowledge_pages kp ON kp.id = kps.knowledge_page_id
WHERE kps.source_page_id = %(page)s
ORDER BY kp.title
"""


def artifacts(cur, page: str) -> list[tuple]:
    cur.execute(ARTIFACTS_OF_SOURCE, {"page": page})
    return cur.fetchall()


def main() -> int:
    config = load_config()
    with psycopg.connect(config.database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='knowledge_page_sources' ORDER BY ordinal_position"
        )
        print("knowledge_page_sources 列:", [r[0] for r in cur.fetchall()])

        for label, page in (("6365 Dee Does Broadway", PAGE_6365), ("6369 Cyndi Lauper", PAGE_6369)):
            print()
            print("=" * 72)
            print(f"{label} 贡献给了哪些 artifact")
            try:
                rows = artifacts(cur, page)
            except psycopg.Error as exc:
                conn.rollback()
                print("  查询失败:", exc)
                continue
            print(f"  共 {len(rows)} 个 artifact")
            for kid, title, ptype, scope, ckey, stale in rows:
                print(f"    {title[:48]:48} type={ptype} scope={scope} stale={stale}")
                print(f"      canonical_key={ckey}  id={kid}")

        print()
        print("=" * 72)
        print("6365 的 artifact 里，chunk 正文提到 Lauper 吗")
        cur.execute(
            f"""
            WITH a AS ({ARTIFACTS_OF_SOURCE})
            SELECT a.title, kc.chunk_role, kc.retrieval_channel,
                   position('Lauper' in kc.text) > 0, left(kc.text, 400)
            FROM a JOIN knowledge_chunks kc ON kc.knowledge_page_id = a.id
            ORDER BY a.title, kc.chunk_role
            """,
            {"page": PAGE_6365},
        )
        for title, role, channel, has, text in cur.fetchall():
            print(f"  [{title[:36]}] role={role} channel={channel} Lauper={has}")
            print(f"    {text}")

        print()
        print("=" * 72)
        print("图边：6365 的 artifact 与 6369 的 artifact 之间")
        cur.execute(
            f"""
            WITH src AS ({ARTIFACTS_OF_SOURCE}),
                 dst AS (
                   SELECT DISTINCT kp.id, kp.title
                   FROM knowledge_page_sources kps
                   JOIN knowledge_pages kp ON kp.id = kps.knowledge_page_id
                   WHERE kps.source_page_id = %(other)s
                 )
            SELECT s.title, e.relation, d.title, e.stale_at
            FROM knowledge_graph_edges e
            JOIN src s ON s.id = e.from_knowledge_page_id
            JOIN dst d ON d.id = e.to_knowledge_page_id
            UNION ALL
            SELECT d.title, e.relation, s.title, e.stale_at
            FROM knowledge_graph_edges e
            JOIN dst d ON d.id = e.from_knowledge_page_id
            JOIN src s ON s.id = e.to_knowledge_page_id
            """,
            {"page": PAGE_6365, "other": PAGE_6369},
        )
        rows = cur.fetchall()
        print(f"  两组 artifact 之间的边：{len(rows)} 条")
        for f, rel, t, stale in rows:
            print(f"    {f[:34]:34} --[{rel}]--> {t[:34]:34} stale={stale}")

        print()
        print("=" * 72)
        print("6365 的 artifact 的全部边（任意方向）")
        cur.execute(
            f"""
            WITH a AS ({ARTIFACTS_OF_SOURCE})
            SELECT a.title, e.relation, kp2.title, 'out' AS dir, e.stale_at
            FROM a JOIN knowledge_graph_edges e ON e.from_knowledge_page_id = a.id
            JOIN knowledge_pages kp2 ON kp2.id = e.to_knowledge_page_id
            UNION ALL
            SELECT a.title, e.relation, kp2.title, 'in', e.stale_at
            FROM a JOIN knowledge_graph_edges e ON e.to_knowledge_page_id = a.id
            JOIN knowledge_pages kp2 ON kp2.id = e.from_knowledge_page_id
            ORDER BY 4, 2
            """,
            {"page": PAGE_6365},
        )
        rows = cur.fetchall()
        print(f"  共 {len(rows)} 条")
        for t1, rel, t2, direction, stale in rows:
            arrow = "-->" if direction == "out" else "<--"
            print(f"    {t1[:30]:30} {arrow}[{rel}] {t2[:34]:34} stale={stale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
