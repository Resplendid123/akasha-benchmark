"""评测补充：从 Akasha 的 ``knowledge_query_audit`` 做三段归因，并**抄进库存档**。

``retrievalDiagnostics`` 被 controller 从 HTTP 响应里解构排除了
（``llm-wiki.controller.ts:174``），只写进 ``knowledge_query_audit.metadata``。
所以想知道召回是**在哪一段**丢的，只有这一条路：

| 分段     | 判据                                        |
| -------- | ------------------------------------------- |
| 召回上限 | ``candidateChunkCount`` 对比 gold 命中      |
| 排序损失 | ``rankedCandidateCount`` 对比候选数         |
| 授权损失 | ``filteredChunkCount``                      |

没有这个拆分，你只知道 Recall@10 = 0.6，却不知道该调哪个旋钮。

**抄进 SQLite 是必须的**（决策 4）：那是 Akasha 的运行时表，会随容器重建消失,
而这批归因无法重算 —— 重跑 query 会得到新的审计行，但那是另一次运行的了。

方向单向：PG 只读，SQLite 可写。连接键是 ``sha256:<hex>``，**带前缀**。

    uv run python -m akasha_benchmark.audit_join --query-label run001-query
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from . import run_args
from .config import load_config
from .datasets import DATASET_NAMES
from .io_utils import atomic_write_json, utc_now
from .store import connect, repo

QUERY = """
SELECT query_hash, retrieval_mode, metadata, created_at
FROM knowledge_query_audit
WHERE workspace_id = %(workspace_id)s
  AND created_at BETWEEN %(start)s AND %(end)s
ORDER BY created_at
"""


def query_hash(query: str) -> str:
    """精确复现服务端的 ``hashQuery``。

    ``llm-wiki.controller.ts:1170`` 返回的是 ``sha256:<hex>``，**带前缀**。
    裸的十六进制值一行都匹配不上，所以这个前缀不是可选的。
    """
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def stage_attribution(metadata: dict[str, Any], gold_hit: bool) -> dict[str, Any]:
    """把一条审计记录拆成三段损失。

    这些是**计数**，帮助定位损失发生在哪一段，不是逐 gold 的因果归因 ——
    报告不能把计数差解释为已证明的逐文档归因（§8.5）。``gold_hit`` 来自离线
    ``hit@10``，不是候选集里的 gold 命中。
    """
    candidates = metadata.get("candidateChunkCount") or 0
    ranked = metadata.get("rankedCandidateCount") or 0
    filtered = metadata.get("filteredChunkCount") or 0
    return {
        "candidate_chunk_count": candidates,
        "ranked_candidate_count": ranked,
        "filtered_chunk_count": filtered,
        "access_policy_fallback_used": bool(metadata.get("accessPolicyFallbackUsed")),
        # 候选集本身是空的：后面任何环节都救不回来，问题在召回上限。
        "recall_ceiling_miss": candidates == 0,
        # 进了候选集，但没通过 RRF / 阈值。
        "ranking_loss": max(candidates - ranked, 0),
        # 排序通过了，被第三道授权闸门丢掉。
        "authorization_loss": filtered,
        "gold_hit": gold_hit,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """一组归因行的汇总统计。"""
    if not rows:
        return {}
    count = len(rows)
    return {
        "count": count,
        "recall_ceiling_miss_rate": sum(r["recall_ceiling_miss"] for r in rows) / count,
        "mean_candidate_chunks": sum(r["candidate_chunk_count"] for r in rows) / count,
        "mean_ranking_loss": sum(r["ranking_loss"] for r in rows) / count,
        "mean_authorization_loss": sum(r["authorization_loss"] for r in rows) / count,
        "access_policy_fallback_rate": (
            sum(r["access_policy_fallback_used"] for r in rows) / count
        ),
        "gold_hit_rate": sum(r["gold_hit"] for r in rows) / count,
    }


def run(
    query_label: str,
    datasets: list[str],
    db_path: Path | None,
    eval_label: str | None = None,
    export_dir: Path | None = None,
) -> int:
    """执行审计表 join，写进 ``audit_record``。缺依赖或缺配置时给出明确提示并返回 1。"""
    try:
        import psycopg
    except ModuleNotFoundError:
        print(
            "ERROR psycopg is not installed. Run `uv add 'psycopg[binary]'` to enable "
            "the audit join.",
            file=sys.stderr,
        )
        return 1

    connection = connect(db_path)
    try:
        query_layer = repo.query_layer_by_label(connection, query_label)
        if query_layer is None:
            print(f"ERROR no query layer labelled {query_label!r}", file=sys.stderr)
            return 1
        query_layer_id = int(query_layer["id"])
        index_layer_id = int(query_layer["index_layer_id"])

        config = load_config(connection)
        if not config.database_url:
            print(
                "ERROR no database_url configured. Fill it in the settings view to enable "
                "the audit join; every other metric works without it.",
                file=sys.stderr,
            )
            return 1

        # workspace 取**这一层入库时记下的那个**，不是任何配置项 ——
        # 审计表按 workspace 分区，用别的值查会一行都连不上（而不是报错）。
        index_layer = repo.get_index_layer(connection, index_layer_id) or {}
        workspace_id = (index_layer.get("workspace_id") or "").strip()
        if not workspace_id:
            print(
                f"ERROR index layer #{index_layer_id} has no recorded workspace_id. "
                "It is written by ingest from users/me, so a layer imported by reindex "
                "may lack it — the audit join needs it to scope its query.",
                file=sys.stderr,
            )
            return 1

        # 时间窗从响应行的 min/max 现算，覆盖累积的全部会话 —— 存快照会在续跑时
        # 被本次统计覆盖，审计于是漏掉早期请求（§10.1）。
        window = repo.request_window(connection, query_layer_id)
        if window is None:
            print(f"ERROR query layer {query_label!r} has no responses", file=sys.stderr)
            return 1

        # 收集 question 文本，并统计哪些文本重复、没法安全 join。
        questions: dict[str, str] = {}
        text_counts: Counter[str] = Counter()
        for dataset in datasets:
            for sample in repo.subset_samples(connection, index_layer_id, dataset):
                questions[sample["sample_id"]] = sample["question"]
                text_counts[sample["question"]] += 1
        ambiguous = {q for q, count in text_counts.items() if count > 1}

        # 只读连接：方向是单向的，这个库我们从不写。
        with psycopg.connect(config.database_url) as pg:
            with pg.cursor() as cursor:
                cursor.execute(
                    QUERY,
                    {
                        "workspace_id": workspace_id,
                        "start": window[0],
                        "end": window[1],
                    },
                )
                rows = cursor.fetchall()

        by_hash: dict[str, list[dict[str, Any]]] = {}
        for stored_hash, retrieval_mode, metadata, created_at in rows:
            by_hash.setdefault(stored_hash, []).append(
                {
                    "retrieval_mode": retrieval_mode,
                    "metadata": metadata or {},
                    "created_at": str(created_at),
                }
            )

        # gold_hit 取自离线评测的 hit@10。没有评测层就留空，不猜。
        gold_hits: dict[str, bool] = {}
        eval_layer = (
            repo.eval_layer_by_label(connection, eval_label)
            if eval_label
            else next(iter(repo.list_eval_layers(connection, query_layer_id)), None)
        )
        if eval_layer:
            for row in repo.sample_evals(connection, int(eval_layer["id"])):
                metrics = repo.sample_metrics_of(
                    connection, int(eval_layer["id"]), row["sample_id"]
                )
                gold_hits[row["sample_id"]] = bool(metrics.get("hit@10"))

        joined: list[dict[str, Any]] = []
        excluded_ambiguous = 0
        unmatched = 0
        for sample_id, question in questions.items():
            if question in ambiguous:
                # 重复文本让那一行有歧义，直接排除，不去随便连一个。
                excluded_ambiguous += 1
                continue
            audit_rows = by_hash.get(query_hash(question))
            if not audit_rows:
                unmatched += 1
                continue
            # 同一条 query 若重试过会有多行，取窗口内最后一次。
            audit = audit_rows[-1]
            repo.record_audit(
                connection,
                query_layer_id,
                sample_id=sample_id,
                query_hash=query_hash(question),
                retrieval_mode=audit["retrieval_mode"],
                metadata=audit["metadata"],
                audit_created_at=audit["created_at"],
            )
            joined.append(
                {
                    "sample_id": sample_id,
                    "retrieval_mode": audit["retrieval_mode"],
                    **stage_attribution(audit["metadata"], gold_hits.get(sample_id, False)),
                }
            )
        connection.commit()

        by_mode: dict[str, list[dict[str, Any]]] = {}
        for row in joined:
            by_mode.setdefault(row["retrieval_mode"] or "unknown", []).append(row)

        report = {
            "stage": "audit_join",
            "query_label": query_label,
            "query_layer_id": query_layer_id,
            "generated_at": utc_now(),
            "window": {"start": window[0], "end": window[1]},
            "audit_rows_in_window": len(rows),
            "joined": len(joined),
            "excluded_ambiguous_question_text": excluded_ambiguous,
            "unmatched_samples": unmatched,
            "gold_hit_source": (
                f"eval layer #{eval_layer['id']} hit@10" if eval_layer else "unavailable"
            ),
            "overall": _summarize(joined),
            # high_completeness 和 high_completeness_fallback 是两种不同的召回口径,
            # 混在一起平均会把问题盖掉，所以必须按 retrievalMode 切开报告。
            "by_retrieval_mode": {mode: _summarize(rows) for mode, rows in sorted(by_mode.items())},
        }
        if export_dir is not None:
            atomic_write_json(export_dir / "audit_join.json", report)

        print(
            f"joined={len(joined)} unmatched={unmatched} "
            f"excluded_ambiguous={excluded_ambiguous} modes={sorted(by_mode)}"
        )
        print(f"window {window[0]} .. {window[1]} ({len(rows)} audit rows)")
        for mode, summary in report["by_retrieval_mode"].items():
            print(
                f"  {mode:<32} n={summary['count']:<4} "
                f"ceiling_miss={summary['recall_ceiling_miss_rate']:.3f} "
                f"rank_loss={summary['mean_ranking_loss']:.2f} "
                f"authz_loss={summary['mean_authorization_loss']:.2f}"
            )
        return 0
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-label", default=None, help="查询层标签")
    parser.add_argument("--eval-label", default=None, help="取 gold_hit 用的评测层，默认最新")
    parser.add_argument("--dataset", action="append", choices=list(DATASET_NAMES))
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--export-dir", type=Path, default=None, help="额外写一份 json（可选）")
    run_args.add_argument(parser)

    try:
        args = run_args.apply(parser.parse_args(argv), stage="audit")
        run_args.require(args.query_label, "query_label", "audit")
        return run(
            args.query_label,
            args.dataset or list(DATASET_NAMES),
            args.db,
            args.eval_label,
            args.export_dir,
        )
    except (RuntimeError, ValueError, FileNotFoundError, sqlite3.Error) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
