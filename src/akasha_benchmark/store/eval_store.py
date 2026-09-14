"""评测层存取。"""

from __future__ import annotations

import sqlite3
from typing import Any

from .db import dumps, loads, utc_now
from .run_store import STATUS_RUNNING


def create_eval_run(
    connection: sqlite3.Connection,
    *,
    name: str,
    query_id: int,
    ks: list[int],
    metrics: list[str],
    judge_provider_id: int | None,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO eval_run
            (name, query_id, ks_json, metrics_json, judge_provider_id, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            query_id,
            dumps(ks),
            dumps(metrics),
            judge_provider_id,
            STATUS_RUNNING,
            utc_now(),
        ),
    )
    return int(cursor.lastrowid or 0)


def get_eval_run(connection: sqlite3.Connection, eval_id: int) -> dict[str, Any] | None:
    row = connection.execute("SELECT * FROM eval_run WHERE id = ?", (eval_id,)).fetchone()
    return dict(row) if row else None


def eval_run_by_name(connection: sqlite3.Connection, name: str) -> dict[str, Any] | None:
    row = connection.execute("SELECT * FROM eval_run WHERE name = ?", (name,)).fetchone()
    return dict(row) if row else None


def list_eval_runs(
    connection: sqlite3.Connection, query_id: int | None = None
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM eval_run"
    params: tuple[Any, ...] = ()
    if query_id is not None:
        sql += " WHERE query_id = ?"
        params = (query_id,)
    return [dict(r) for r in connection.execute(sql + " ORDER BY id DESC", params)]


def delete_eval_run(connection: sqlite3.Connection, eval_id: int) -> int:
    return connection.execute("DELETE FROM eval_run WHERE id = ?", (eval_id,)).rowcount


def clear_eval_results(connection: sqlite3.Connection, eval_id: int, dataset: str) -> None:
    """重算确定性结果；已有 Judge 分数由 verdict 重新写入指标表。"""
    for table in ("sample_metric", "sample_eval", "metric_summary", "dataset_eval"):
        connection.execute(
            f"DELETE FROM {table} WHERE eval_id = ? AND dataset = ?", (eval_id, dataset)
        )


def restore_judge_metrics(connection: sqlite3.Connection, eval_id: int, metric: str) -> None:
    connection.execute(
        """
        INSERT INTO sample_metric (eval_id, sample_id, dataset, metric, value)
        SELECT j.eval_id, j.sample_id, s.dataset, j.metric, j.score
        FROM judge_verdict j
        JOIN sample_eval s ON s.eval_id = j.eval_id AND s.sample_id = j.sample_id
        WHERE j.eval_id = ? AND j.metric = ? AND j.score IS NOT NULL
        ON CONFLICT(eval_id, sample_id, metric) DO UPDATE SET value = excluded.value
        """,
        (eval_id, metric),
    )


def record_sample_eval(
    connection: sqlite3.Connection,
    eval_id: int,
    *,
    sample_id: str,
    dataset: str,
    answer_mode: str | None,
    http_status: int,
    answer: str,
    detail: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    connection.execute(
        """
        INSERT INTO sample_eval
            (eval_id, sample_id, dataset, answer_mode, http_status, answer, detail_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(eval_id, sample_id) DO UPDATE SET
            answer_mode = excluded.answer_mode,
            http_status = excluded.http_status,
            answer = excluded.answer,
            detail_json = excluded.detail_json
        """,
        (eval_id, sample_id, dataset, answer_mode, http_status, answer, dumps(detail)),
    )
    connection.executemany(
        """
        INSERT INTO sample_metric (eval_id, sample_id, dataset, metric, value)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(eval_id, sample_id, metric) DO UPDATE SET value = excluded.value
        """,
        [(eval_id, sample_id, dataset, name, float(value)) for name, value in metrics.items()],
    )


def record_metric_summary(
    connection: sqlite3.Connection,
    eval_id: int,
    dataset: str,
    scope: str,
    metrics: dict[str, float],
    sample_count: int,
) -> None:
    connection.executemany(
        """
        INSERT INTO metric_summary (eval_id, dataset, scope, metric, value, sample_count)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(eval_id, dataset, scope, metric) DO UPDATE SET
            value = excluded.value, sample_count = excluded.sample_count
        """,
        [
            (eval_id, dataset, scope, name, float(value), sample_count)
            for name, value in metrics.items()
        ],
    )


def record_dataset_eval(
    connection: sqlite3.Connection,
    eval_id: int,
    dataset: str,
    *,
    responses_evaluated: int,
    http_failures: int,
    omitted_metrics: list[str],
    answer_modes: dict[str, float],
) -> None:
    connection.execute(
        """
        INSERT INTO dataset_eval
            (eval_id, dataset, responses_evaluated, http_failures,
             omitted_metrics_json, answer_modes_json)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(eval_id, dataset) DO UPDATE SET
            responses_evaluated = excluded.responses_evaluated,
            http_failures = excluded.http_failures,
            omitted_metrics_json = excluded.omitted_metrics_json,
            answer_modes_json = excluded.answer_modes_json
        """,
        (
            eval_id,
            dataset,
            responses_evaluated,
            http_failures,
            dumps(omitted_metrics),
            dumps(answer_modes),
        ),
    )


def dataset_evals(connection: sqlite3.Connection, eval_id: int) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row["dataset"],
            "responses_evaluated": row["responses_evaluated"],
            "http_failures": row["http_failures"],
            "omitted_metrics": loads(row["omitted_metrics_json"], []),
            "answer_modes": loads(row["answer_modes_json"], {}),
        }
        for row in connection.execute(
            "SELECT * FROM dataset_eval WHERE eval_id = ? ORDER BY dataset", (eval_id,)
        )
    ]


def metric_summaries(connection: sqlite3.Connection, eval_id: int) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in connection.execute(
            "SELECT * FROM metric_summary WHERE eval_id = ? ORDER BY dataset, scope, metric",
            (eval_id,),
        )
    ]


def sample_evals(
    connection: sqlite3.Connection,
    eval_id: int,
    *,
    dataset: str | None = None,
    answer_mode: str | None = None,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM sample_eval WHERE eval_id = ?"
    params: list[Any] = [eval_id]
    if dataset:
        sql += " AND dataset = ?"
        params.append(dataset)
    if answer_mode:
        sql += " AND answer_mode = ?"
        params.append(answer_mode)
    return [
        {
            **{k: v for k, v in dict(row).items() if k != "detail_json"},
            "detail": loads(row["detail_json"], {}),
        }
        for row in connection.execute(sql + " ORDER BY sample_id", params)
    ]


def sample_eval(
    connection: sqlite3.Connection, eval_id: int, sample_id: str
) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT * FROM sample_eval WHERE eval_id = ? AND sample_id = ?", (eval_id, sample_id)
    ).fetchone()
    if row is None:
        return None
    return {
        **{k: v for k, v in dict(row).items() if k != "detail_json"},
        "detail": loads(row["detail_json"], {}),
    }


def sample_metrics_of(
    connection: sqlite3.Connection, eval_id: int, sample_id: str
) -> dict[str, float]:
    return {
        row["metric"]: row["value"]
        for row in connection.execute(
            "SELECT metric, value FROM sample_metric WHERE eval_id = ? AND sample_id = ?",
            (eval_id, sample_id),
        )
    }


def samples_ranked_by(
    connection: sqlite3.Connection,
    eval_id: int,
    metric: str,
    *,
    dataset: str | None = None,
    ascending: bool = True,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """按某个指标排序的样本，即归因层的「最差 N 条」。"""
    sql = """
        SELECT sm.sample_id, sm.dataset, sm.value, se.answer_mode, se.answer
        FROM sample_metric sm
        JOIN sample_eval se ON se.eval_id = sm.eval_id AND se.sample_id = sm.sample_id
        WHERE sm.eval_id = ? AND sm.metric = ?
    """
    params: list[Any] = [eval_id, metric]
    if dataset:
        sql += " AND sm.dataset = ?"
        params.append(dataset)
    sql += f" ORDER BY sm.value {'ASC' if ascending else 'DESC'}, sm.sample_id LIMIT ?"
    params.append(limit)
    return [dict(r) for r in connection.execute(sql, params)]


def record_judge_verdict(
    connection: sqlite3.Connection,
    eval_id: int,
    *,
    sample_id: str,
    score: float | None,
    failure_kind: str | None,
    detail: dict[str, Any] | None,
    metric: str = "faithfulness",
    latency_ms: int | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO judge_verdict
            (eval_id, sample_id, metric, score, failure_kind, latency_ms, detail_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(eval_id, sample_id, metric) DO UPDATE SET
            score = excluded.score,
            failure_kind = excluded.failure_kind,
            latency_ms = excluded.latency_ms,
            detail_json = excluded.detail_json
        """,
        (
            eval_id,
            sample_id,
            metric,
            score,
            failure_kind,
            latency_ms,
            dumps(detail) if detail else None,
        ),
    )


def judged_sample_ids(
    connection: sqlite3.Connection, eval_id: int, metric: str | None = None
) -> set[str]:
    """已判过的样本。``metric`` 为空时不分指标；续跑要逐指标问。"""
    sql = "SELECT sample_id FROM judge_verdict WHERE eval_id = ?"
    params: list[Any] = [eval_id]
    if metric is not None:
        sql += " AND metric = ?"
        params.append(metric)
    return {row["sample_id"] for row in connection.execute(sql, params)}


def judge_verdicts(connection: sqlite3.Connection, eval_id: int) -> list[dict[str, Any]]:
    return [
        {
            **{k: v for k, v in dict(row).items() if k != "detail_json"},
            "detail": loads(row["detail_json"]),
        }
        for row in connection.execute(
            "SELECT * FROM judge_verdict WHERE eval_id = ? ORDER BY sample_id", (eval_id,)
        )
    ]


def judge_means_by_dataset(
    connection: sqlite3.Connection, eval_id: int, metric: str | None = None
) -> list[tuple[str, float, int]]:
    """各数据集的 judge 均值。跳过与失败的不进分母。"""
    sql = """
        SELECT se.dataset, AVG(jv.score) AS mean, COUNT(*) AS n
        FROM judge_verdict jv
        JOIN sample_eval se ON se.eval_id = jv.eval_id AND se.sample_id = jv.sample_id
        WHERE jv.eval_id = ? AND jv.score IS NOT NULL
    """
    params: list[Any] = [eval_id]
    if metric is not None:
        sql += " AND jv.metric = ?"
        params.append(metric)
    return [
        (row["dataset"], row["mean"], row["n"])
        for row in connection.execute(sql + " GROUP BY se.dataset ORDER BY se.dataset", params)
    ]


def judge_summary(
    connection: sqlite3.Connection, eval_id: int, metric: str | None = None
) -> dict[str, Any]:
    """judge 汇总。失败该条排除、不记 0，另给失败率说明均值覆盖了多少。"""
    scope = "" if metric is None else " AND metric = ?"
    extra: list[Any] = [] if metric is None else [metric]
    row = connection.execute(
        f"""
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN score IS NOT NULL THEN 1 ELSE 0 END) AS scored,
               SUM(CASE WHEN failure_kind IS NOT NULL THEN 1 ELSE 0 END) AS failed,
               AVG(score) AS mean
        FROM judge_verdict WHERE eval_id = ?{scope}
        """,
        (eval_id, *extra),
    ).fetchone()
    total = row["total"] or 0
    kinds = {
        r["failure_kind"]: r["n"]
        for r in connection.execute(
            "SELECT failure_kind, COUNT(*) AS n FROM judge_verdict "
            f"WHERE eval_id = ? AND failure_kind IS NOT NULL{scope} GROUP BY failure_kind",
            (eval_id, *extra),
        )
    }
    # 只算真发过调用的条目：跳过的没有 latency。
    latency = connection.execute(
        "SELECT AVG(latency_ms) AS mean FROM judge_verdict "
        f"WHERE eval_id = ? AND latency_ms IS NOT NULL{scope}",
        (eval_id, *extra),
    ).fetchone()
    return {
        "total": total,
        "scored": row["scored"] or 0,
        "failed": row["failed"] or 0,
        "mean": row["mean"],
        "latency_mean": latency["mean"],
        "failure_rate": (row["failed"] or 0) / total if total else 0.0,
        "failures_by_kind": kinds,
    }
