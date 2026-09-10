"""编译器的实体/关系抽取是「有上限」还是「有选择」？压缩还是扩写？

跨全部已入库文档看分布，而不是只看一条样本 —— 用来判断某个案例是个例
还是系统性行为。案例见 ../recall-miss-compiled-away.md。

只读 SQL。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import psycopg

from akasha_benchmark.config import load_config


def main() -> int:
    config = load_config()
    with psycopg.connect(config.database_url) as conn, conn.cursor() as cur:
        print("=" * 72)
        print("1. 每篇原始 page 编出多少 artifact（按 page_type）")
        cur.execute(
            """
            SELECT kp.page_type, count(*) AS artifacts,
                   count(DISTINCT kps.source_page_id) AS pages
            FROM knowledge_page_sources kps
            JOIN knowledge_pages kp ON kp.id = kps.knowledge_page_id
            GROUP BY kp.page_type ORDER BY 2 DESC
            """
        )
        for ptype, n, pages in cur.fetchall():
            print(f"  {ptype:16} artifact={n:5}  覆盖 {pages} 篇原始 page")

        print()
        print("2. entity artifact 数 / 每篇原始 page 的分布")
        cur.execute(
            """
            SELECT c, count(*) FROM (
              SELECT kps.source_page_id, count(*) AS c
              FROM knowledge_page_sources kps
              JOIN knowledge_pages kp ON kp.id = kps.knowledge_page_id
              WHERE kp.page_type = 'entity'
              GROUP BY 1
            ) t GROUP BY c ORDER BY c
            """
        )
        rows = cur.fetchall()
        total = sum(n for _, n in rows)
        print(f"  （{total} 篇有 entity artifact）")
        for c, n in rows:
            bar = "#" * min(n // 2, 60)
            print(f"    {c:3} 个 entity : {n:4} 篇  {bar}")

        print()
        print("3. 图边的 relation 取值分布")
        cur.execute(
            "SELECT relation, count(*) FROM knowledge_graph_edges "
            "GROUP BY relation ORDER BY 2 DESC LIMIT 30"
        )
        for rel, n in cur.fetchall():
            print(f"  {rel:28} {n}")

        print()
        print("4. 出边数 / 每个 entity artifact 的分布")
        cur.execute(
            """
            SELECT c, count(*) FROM (
              SELECT kp.id, count(e.id) AS c
              FROM knowledge_pages kp
              LEFT JOIN knowledge_graph_edges e ON e.from_knowledge_page_id = kp.id
              WHERE kp.page_type = 'entity'
              GROUP BY kp.id
            ) t GROUP BY c ORDER BY c
            """
        )
        for c, n in cur.fetchall():
            print(f"    {c:3} 条出边 : {n:4} 个 entity")

        print()
        print("=" * 72)
        print("5. 跨文档实体合并有没有在工作")
        cur.execute(
            """
            SELECT kp.title, kp.canonical_key, count(DISTINCT kps.source_page_id) AS srcs
            FROM knowledge_pages kp
            JOIN knowledge_page_sources kps ON kps.knowledge_page_id = kp.id
            WHERE kp.page_type = 'entity'
            GROUP BY kp.id, kp.title, kp.canonical_key
            HAVING count(DISTINCT kps.source_page_id) > 1
            ORDER BY 3 DESC LIMIT 12
            """
        )
        rows = cur.fetchall()
        print(f"  被多篇原文贡献的 entity artifact：{len(rows)} 个（取前 12）")
        for title, ckey, srcs in rows:
            print(f"    {title[:44]:44} {srcs} 篇  key={ckey}")

        print()
        print("6. cyndi_lauper 这个 artifact 被哪些原文贡献 / 有哪些边")
        cur.execute(
            """
            SELECT kp.id FROM knowledge_pages kp
            WHERE kp.canonical_key = 'cyndi_lauper'
            """
        )
        row = cur.fetchone()
        if row is None:
            print("  没找到")
        else:
            kid = row[0]
            cur.execute(
                """
                SELECT p.title FROM knowledge_page_sources kps
                JOIN pages p ON p.id = kps.source_page_id
                WHERE kps.knowledge_page_id = %(kid)s
                """,
                {"kid": kid},
            )
            print("  贡献它的原文:", [r[0] for r in cur.fetchall()])
            cur.execute(
                """
                SELECT e.relation, kp2.title, 'out' FROM knowledge_graph_edges e
                JOIN knowledge_pages kp2 ON kp2.id = e.to_knowledge_page_id
                WHERE e.from_knowledge_page_id = %(kid)s
                UNION ALL
                SELECT e.relation, kp2.title, 'in' FROM knowledge_graph_edges e
                JOIN knowledge_pages kp2 ON kp2.id = e.from_knowledge_page_id
                WHERE e.to_knowledge_page_id = %(kid)s
                """,
                {"kid": kid},
            )
            for rel, title, d in cur.fetchall():
                arrow = "-->" if d == "out" else "<--"
                print(f"    {arrow}[{rel}] {title[:50]}")

        print()
        print("=" * 72)
        print("7. 原文长度 vs 编译产物长度（压缩率）")
        cur.execute(
            """
            SELECT sc.source_page_id,
                   sum(length(sc.text)) AS src_len
            FROM knowledge_source_chunks sc GROUP BY 1
            """
        )
        src = {str(p): n for p, n in cur.fetchall()}
        cur.execute(
            """
            SELECT kps.source_page_id, sum(length(kc.text)) AS art_len
            FROM knowledge_page_sources kps
            JOIN knowledge_chunks kc ON kc.knowledge_page_id = kps.knowledge_page_id
            GROUP BY 1
            """
        )
        art = {str(p): n for p, n in cur.fetchall()}
        ratios = []
        for page, s in src.items():
            a = art.get(page)
            if a and s:
                ratios.append(a / s)
        if ratios:
            ratios.sort()
            n = len(ratios)
            print(f"  样本 {n} 篇；编译后总字符数 / 原文字符数")
            print(f"    最小 {ratios[0]:.2f}  p25 {ratios[n//4]:.2f}  "
                  f"中位 {ratios[n//2]:.2f}  p75 {ratios[3*n//4]:.2f}  最大 {ratios[-1]:.2f}")
            print(f"    < 1.0（净压缩）的比例: {sum(1 for r in ratios if r < 1.0)/n:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
