"""评测补充：从 ``knowledge_query_audit`` 做三段归因。

``retrievalDiagnostics`` 被 controller 从 HTTP 响应里解构排除了
（``llm-wiki.controller.ts:174``），只写进 ``knowledge_query_audit.metadata``。
所以想知道召回是**在哪一段**丢的，只有这一条路：

| 分段     | 判据                                        |
| -------- | ------------------------------------------- |
| 召回上限 | ``candidateChunkCount`` 对比 gold 命中      |
| 排序损失 | ``rankedCandidateCount`` 对比候选数         |
| 授权损失 | ``filteredChunkCount``                      |

没有这个拆分，你只知道 Recall@10 = 0.6，却不知道该调哪个旋钮。

连接键是 ``sha256:<hex>``，**带前缀**。重复的 question 文本会让那一行有歧义，
所以这些行直接排除，不去随便连一个 —— 在锁定快照上是 musique 1 行、其余 0 行。

本模块可选：需要 ``psycopg`` 和 ``database_url``。评测的其余指标不依赖它。

    uv run python -m akasha_benchmark.audit_join --run-id run001
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .config import load_config
from .datasets import DATASET_NAMES, CanonicalSample, subset_dir
from .evaluate import reports_dir
from .io_utils import atomic_write_json, load_json, read_jsonl, utc_now
from .run_queries import responses_dir

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


def _stage_attribution(metadata: dict[str, Any], gold_hit: bool) -> dict[str, Any]:
    """把一条审计记录拆成三段损失。"""
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


def run(run_id: str, datasets: list[str], config_path: Path | None, data_dir: Path | None) -> int:
    """执行审计表 join，产出 audit_join.json。缺依赖或缺配置时给出明确提示并返回 1。"""
    config = load_config(config_path)
    if not config.database_url:
        print(
            "ERROR no database_url configured. Set AKASHA_DATABASE_URL to enable the "
            "audit join; every other metric works without it.",
            file=sys.stderr,
        )
        return 1
    if not config.workspace_id:
        print("ERROR workspace_id is required for the audit join", file=sys.stderr)
        return 1

    try:
        import psycopg
    except ModuleNotFoundError:
        print(
            "ERROR psycopg is not installed. Run `uv add 'psycopg[binary]'` to enable "
            "the audit join.",
            file=sys.stderr,
        )
        return 1

    per_sample_path = reports_dir(run_id, data_dir) / "per_sample.jsonl"
    if not per_sample_path.is_file():
        print(f"ERROR {per_sample_path} missing; run the evaluate stage first", file=sys.stderr)
        return 1
    evaluated = {row["sample_id"]: row for row in read_jsonl(per_sample_path)}

    # 收集每条样本的 question 文本，同时统计哪些文本重复、没法安全 join。
    questions: dict[str, str] = {}
    text_counts: Counter[str] = Counter()
    for dataset in datasets:
        path = subset_dir(run_id, dataset, data_dir) / "samples.jsonl"
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            sample = CanonicalSample.model_validate(row)
            questions[sample.sample_id] = sample.question
            text_counts[sample.question] += 1

    ambiguous = {q for q, count in text_counts.items() if count > 1}

    window_path = responses_dir(run_id, data_dir) / "manifest.json"
    if not window_path.is_file():
        print(f"ERROR {window_path} missing; run the query stage first", file=sys.stderr)
        return 1
    window = load_json(window_path)

    with psycopg.connect(config.database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                QUERY,
                {
                    "workspace_id": config.workspace_id,
                    # 用查询阶段自己的运行时间窗卡住范围，
                    # 否则上一次运行的审计行会混进这次的 join。
                    "start": window["started_at"],
                    "end": window["generated_at"],
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

    joined: list[dict[str, Any]] = []
    excluded_ambiguous = 0
    unmatched = 0
    for sample_id, question in questions.items():
        if question in ambiguous:
            excluded_ambiguous += 1
            continue
        audit_rows = by_hash.get(query_hash(question))
        if not audit_rows:
            unmatched += 1
            continue
        # 同一条 query 若重试过会有多行，取最后一次。
        audit = audit_rows[-1]
        evaluation = evaluated.get(sample_id) or {}
        gold_hit = bool((evaluation.get("retrieval") or {}).get("hit@10"))
        joined.append(
            {
                "sample_id": sample_id,
                "retrieval_mode": audit["retrieval_mode"],
                **_stage_attribution(audit["metadata"], gold_hit),
            }
        )

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in joined:
        by_mode.setdefault(row["retrieval_mode"] or "unknown", []).append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
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

    report = {
        "stage": "audit_join",
        "run_id": run_id,
        "generated_at": utc_now(),
        "audit_rows_in_window": len(rows),
        "joined": len(joined),
        "excluded_ambiguous_question_text": excluded_ambiguous,
        "unmatched_samples": unmatched,
        "overall": summarize(joined),
        # high_completeness 和 high_completeness_fallback 是两种不同的召回口径，
        # 混在一起平均会把问题盖掉，所以必须按 retrievalMode 切开报告。
        "by_retrieval_mode": {mode: summarize(rows) for mode, rows in sorted(by_mode.items())},
    }
    atomic_write_json(reports_dir(run_id, data_dir) / "audit_join.json", report)

    print(
        f"joined={len(joined)} unmatched={unmatched} "
        f"excluded_ambiguous={excluded_ambiguous} modes={sorted(by_mode)}"
    )
    for mode, summary in report["by_retrieval_mode"].items():
        print(
            f"  {mode:<32} n={summary['count']:<4} "
            f"ceiling_miss={summary['recall_ceiling_miss_rate']:.3f} "
            f"rank_loss={summary['mean_ranking_loss']:.2f} "
            f"authz_loss={summary['mean_authorization_loss']:.2f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", action="append", choices=list(DATASET_NAMES))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    return run(args.run_id, args.dataset or list(DATASET_NAMES), args.config, args.data_dir)


if __name__ == "__main__":
    sys.exit(main())
