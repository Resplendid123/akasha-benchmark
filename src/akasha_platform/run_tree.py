"""编译、查询、评测与归因运行记录的批量读取模型。"""

from __future__ import annotations

import sqlite3
from typing import Any

from akasha_benchmark.metrics import registry
from akasha_benchmark.store import (
    attribution_store,
    compile_store,
    eval_store,
    loads,
    query_store,
)

_MISSING = object()


def build_compile_tree(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """批量读取完整运行树，查询数量不随运行记录数增长。"""
    compile_rows = compile_store.list_compile_runs(connection)
    query_rows = query_store.list_query_runs(connection)
    eval_rows = eval_store.list_eval_runs(connection)
    attribution_rows = attribution_store.list_attribution_runs(connection)
    compile_stats = _grouped(
        [
            dict(row)
            for row in connection.execute(
                """
                SELECT cd.compile_id, cd.dataset, COUNT(*) AS docs,
                       SUM(cd.page_id IS NOT NULL) AS imported, SUM(cd.is_gold) AS gold
                FROM compile_doc cd
                GROUP BY cd.compile_id, cd.dataset
                """
            )
        ],
        "compile_id",
    )
    compile_samples = {
        (int(row["compile_id"]), row["dataset"]): int(row["samples"])
        for row in connection.execute(
            """
            SELECT cs.compile_id, s.dataset, COUNT(*) AS samples
            FROM compile_sample cs JOIN sample s ON s.sample_id = cs.sample_id
            GROUP BY cs.compile_id, s.dataset
            """
        )
    }
    query_stats = _grouped(
        [
            dict(row)
            for row in connection.execute(
                """
                SELECT qr.query_id, qr.dataset, COUNT(*) AS responses,
                       SUM(NOT (qr.http_status BETWEEN 200 AND 299)) AS failures,
                       AVG(qr.latency_ms) AS latency_mean, MAX(qr.latency_ms) AS latency_max
                FROM query_response qr
                GROUP BY qr.query_id, qr.dataset
                """
            )
        ],
        "query_id",
    )
    query_sample_counts = {
        int(row["query_id"]): int(row["samples"])
        for row in connection.execute(
            "SELECT query_id, COUNT(*) AS samples FROM query_sample GROUP BY query_id"
        )
    }
    eval_samples = _grouped(
        [dict(row) for row in connection.execute("SELECT eval_id, sample_id, http_status FROM sample_eval")],
        "eval_id",
    )
    verdicts = {
        (int(row["eval_id"]), row["sample_id"], row["metric"]): row["failure_kind"]
        for row in connection.execute(
            "SELECT eval_id, sample_id, metric, failure_kind FROM judge_verdict"
        )
    }
    attribution_counts = {
        int(row["attribution_id"]): dict(row)
        for row in connection.execute(
                """
                SELECT ar.id AS attribution_id,
                       COALESCE(samples.samples, 0) AS samples,
                       COALESCE(results.succeeded, 0) AS succeeded
                FROM attribution_run ar
                LEFT JOIN (
                    SELECT eval_id, COUNT(*) AS samples
                    FROM sample_eval GROUP BY eval_id
                ) samples ON samples.eval_id = ar.eval_id
                LEFT JOIN (
                SELECT attribution_id, COUNT(*) AS succeeded
                FROM attribution_result GROUP BY attribution_id
            ) results ON results.attribution_id = ar.id
            """
        )
    }
    provider_labels = {
        int(row["id"]): row["label"]
        for row in connection.execute("SELECT id, label FROM model_provider")
    }

    queries_by_compile = _grouped(query_rows, "compile_id")
    evals_by_query = _grouped(eval_rows, "query_id")
    attributions_by_eval = _grouped(attribution_rows, "eval_id")
    return [
        _compile_view(
            row,
            compile_stats=compile_stats,
            compile_samples=compile_samples,
            queries=queries_by_compile.get(int(row["id"]), []),
            evals_by_query=evals_by_query,
            attributions_by_eval=attributions_by_eval,
            query_stats=query_stats,
            query_sample_counts=query_sample_counts,
            eval_samples=eval_samples,
            verdicts=verdicts,
            attribution_counts=attribution_counts,
            provider_labels=provider_labels,
        )
        for row in compile_rows
    ]


def _compile_view(
    row: dict[str, Any],
    *,
    compile_stats: dict[int, list[dict[str, Any]]],
    compile_samples: dict[tuple[int, str], int],
    queries: list[dict[str, Any]],
    evals_by_query: dict[int, list[dict[str, Any]]],
    attributions_by_eval: dict[int, list[dict[str, Any]]],
    query_stats: dict[int, list[dict[str, Any]]],
    query_sample_counts: dict[int, int],
    eval_samples: dict[int, list[dict[str, Any]]],
    verdicts: dict[tuple[int, str, str], str | None],
    attribution_counts: dict[int, dict[str, Any]],
    provider_labels: dict[int, str],
) -> dict[str, Any]:
    compile_id = int(row["id"])
    stats = {
        item["dataset"]: {key: value for key, value in item.items() if key != "compile_id"}
        for item in compile_stats.get(compile_id, [])
    }
    for (sample_compile_id, dataset), count in compile_samples.items():
        if sample_compile_id == compile_id:
            stats.setdefault(dataset, {"dataset": dataset})["samples"] = count
    total = sum(int(item.get("docs") or 0) for item in stats.values())
    missing = total - sum(int(item.get("imported") or 0) for item in stats.values())
    return {
        **_public_run(row),
        "config_group": _selection_label(
            loads(row.get("model_selection_json"), {}), row.get("config_group")
        ),
        "model_selection": loads(row.get("model_selection_json"), {}),
        "datasets": loads(row["datasets_json"], []),
        "stats": stats,
        "quality": loads(row["quality_json"]),
        "pace": loads(row["pace_json"]),
        "readiness": compile_store.compile_readiness(row, total=total, missing=missing),
        "compiled_pages": _compiled_pages(row, stats),
        "compiled_pages_error": None,
        "queries": [
            _query_view(
                query,
                evals=evals_by_query.get(int(query["id"]), []),
                attributions_by_eval=attributions_by_eval,
                query_stats=query_stats,
                query_sample_counts=query_sample_counts,
                eval_samples=eval_samples,
                verdicts=verdicts,
                attribution_counts=attribution_counts,
                provider_labels=provider_labels,
            )
            for query in queries
        ],
    }


def _query_view(
    row: dict[str, Any],
    *,
    evals: list[dict[str, Any]],
    attributions_by_eval: dict[int, list[dict[str, Any]]],
    query_stats: dict[int, list[dict[str, Any]]],
    query_sample_counts: dict[int, int],
    eval_samples: dict[int, list[dict[str, Any]]],
    verdicts: dict[tuple[int, str, str], str | None],
    attribution_counts: dict[int, dict[str, Any]],
    provider_labels: dict[int, str],
) -> dict[str, Any]:
    query_id = int(row["id"])
    stats = {
        item["dataset"]: {key: value for key, value in item.items() if key != "query_id"}
        for item in query_stats.get(query_id, [])
    }
    response_count = sum(int(item["responses"] or 0) for item in stats.values())
    return {
        **_public_run(row),
        "config_group": _selection_label(
            loads(row.get("model_selection_json"), {}), row.get("config_group")
        ),
        "model_selection": loads(row.get("model_selection_json"), {}),
        "sample_count": query_sample_counts.get(query_id, 0) or response_count,
        "success_count": sum(
            int(item["responses"] or 0) - int(item["failures"] or 0)
            for item in stats.values()
        ),
        "stats": stats,
        "evals": [
            _eval_view(
                evaluation,
                samples=eval_samples.get(int(evaluation["id"]), []),
                attributions=attributions_by_eval.get(int(evaluation["id"]), []),
                verdicts=verdicts,
                attribution_counts=attribution_counts,
                provider_labels=provider_labels,
            )
            for evaluation in evals
        ],
    }


def _eval_view(
    row: dict[str, Any],
    *,
    samples: list[dict[str, Any]],
    attributions: list[dict[str, Any]],
    verdicts: dict[tuple[int, str, str], str | None],
    attribution_counts: dict[int, dict[str, Any]],
    provider_labels: dict[int, str],
) -> dict[str, Any]:
    eval_id = int(row["id"])
    metrics = [
        name for name in loads(row["metrics_json"], []) if name in registry.METRIC_REGISTRY
    ]
    judge_metrics = {
        name for name in metrics if registry.get_metric(name).kind == registry.KIND_JUDGE
    }
    succeeded = sum(
        1
        for sample in samples
        if 200 <= int(sample["http_status"] or 0) < 300
        and all(
            verdicts.get((eval_id, sample["sample_id"], metric), _MISSING) is None
            for metric in judge_metrics
        )
    )
    return {
        **_public_run(row),
        "config_group": (
            provider_labels.get(int(row["judge_provider_id"]))
            if row.get("judge_provider_id") is not None
            else "确定性指标"
        ),
        "sample_count": len(samples),
        "success_count": succeeded,
        "ks": loads(row["ks_json"], []),
        "metrics": metrics,
        "attributions": [
            {
                **_public_run(attribution),
                "config_group": (
                    provider_labels.get(int(attribution["provider_id"]))
                    if attribution.get("provider_id") is not None
                    else "规则归因"
                ),
                "sample_count": int(
                    attribution_counts.get(int(attribution["id"]), {}).get("samples") or 0
                ),
                "success_count": int(
                    attribution_counts.get(int(attribution["id"]), {}).get("succeeded") or 0
                ),
            }
            for attribution in attributions
        ],
    }


def _grouped(rows: list[dict[str, Any]], key: str) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row[key]), []).append(row)
    return grouped


def _compiled_pages(run: dict[str, Any], stats: dict[str, Any]) -> int | None:
    quality = loads(run["quality_json"]) or {}
    succeeded = (quality.get("progress") or {}).get("succeeded")
    if isinstance(succeeded, int) and not isinstance(succeeded, bool):
        return succeeded
    if quality.get("passed") is True:
        return sum(int(item.get("imported") or 0) for item in stats.values())
    return None


def _public_run(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.endswith("_json")}


def _selection_label(selection: dict[str, Any] | None, fallback: str | None) -> str | None:
    selection = selection or {}
    labels = [
        str(item.get("label") or item.get("model") or "")
        for item in selection.values()
        if isinstance(item, dict)
    ]
    labels = [label for label in labels if label]
    return " + ".join(labels) if labels else fallback
