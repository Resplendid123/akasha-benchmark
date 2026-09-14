"""编译 / 查询 / 评测 / 归因四层的存取。

每层一张主表加若干产物表，产物表全部 ``ON DELETE CASCADE``，
所以删主表那一行就是这一层的清理。
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .db import dumps, loads, utc_now

STATUS_RUNNING = "running"
STATUS_PAUSED = "paused"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"


# ------------------------------------------------------------------ 编译层


def create_compile_run(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    datasets: list[str],
    seed: int,
    qa_limit: int,
    negatives_ratio: float,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO compile_run
            (run_id, datasets_json, seed, qa_limit, negatives_ratio, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, dumps(datasets), seed, qa_limit, negatives_ratio, STATUS_RUNNING, utc_now()),
    )
    return int(cursor.lastrowid or 0)


def update_compile_run(connection: sqlite3.Connection, compile_id: int, **fields: Any) -> None:
    allowed = {
        "space_id",
        "space_name",
        "workspace_id",
        "model_configs_json",
        "quality_json",
        "pace_json",
        "status",
        "finished_at",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"unknown compile_run fields: {sorted(unknown)}")
    if not fields:
        return
    assignments = ", ".join(f"{name} = ?" for name in fields)
    connection.execute(
        f"UPDATE compile_run SET {assignments} WHERE id = ?", (*fields.values(), compile_id)
    )


def get_compile_run(connection: sqlite3.Connection, compile_id: int) -> dict[str, Any] | None:
    row = connection.execute("SELECT * FROM compile_run WHERE id = ?", (compile_id,)).fetchone()
    return dict(row) if row else None


def compile_run_by_run_id(connection: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    row = connection.execute("SELECT * FROM compile_run WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def list_compile_runs(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in connection.execute("SELECT * FROM compile_run ORDER BY id DESC")]


def delete_compile_run(connection: sqlite3.Connection, compile_id: int) -> int:
    return connection.execute("DELETE FROM compile_run WHERE id = ?", (compile_id,)).rowcount


def replace_compile_subset(
    connection: sqlite3.Connection,
    compile_id: int,
    dataset: str,
    sample_ids: list[str],
    docs: list[dict[str, Any]],
) -> None:
    """写入一个数据集的子集。重抽样先删旧行，免得残留被一起导入。"""
    connection.execute(
        "DELETE FROM compile_sample WHERE compile_id = ? AND dataset = ?", (compile_id, dataset)
    )
    connection.execute(
        "DELETE FROM compile_doc WHERE compile_id = ? AND dataset = ?", (compile_id, dataset)
    )
    connection.executemany(
        "INSERT INTO compile_sample (compile_id, sample_id, dataset) VALUES (?, ?, ?)",
        [(compile_id, sample_id, dataset) for sample_id in sample_ids],
    )
    connection.executemany(
        "INSERT INTO compile_doc (compile_id, dataset, doc_id, is_gold) VALUES (?, ?, ?, ?)",
        [(compile_id, dataset, doc["doc_id"], int(doc["is_gold"])) for doc in docs],
    )


def compile_samples(
    connection: sqlite3.Connection, compile_id: int, dataset: str | None = None
) -> list[dict[str, Any]]:
    sql = """
        SELECT s.*, cs.dataset AS subset_dataset
        FROM compile_sample cs JOIN sample s ON s.sample_id = cs.sample_id
        WHERE cs.compile_id = ?
    """
    params: list[Any] = [compile_id]
    if dataset:
        sql += " AND cs.dataset = ?"
        params.append(dataset)
    return [
        {
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "dataset_sample_id": row["dataset_sample_id"],
            "question": row["question"],
            "answers": loads(row["answers_json"], []),
            "gold_doc_ids": loads(row["gold_doc_ids_json"], []),
            "metadata": loads(row["metadata_json"], {}),
        }
        for row in connection.execute(sql + " ORDER BY s.sample_id", params)
    ]


def compile_docs(
    connection: sqlite3.Connection,
    compile_id: int,
    dataset: str | None = None,
    *,
    pending_only: bool = False,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM compile_doc WHERE compile_id = ?"
    params: list[Any] = [compile_id]
    if dataset:
        sql += " AND dataset = ?"
        params.append(dataset)
    if pending_only:
        sql += " AND page_id IS NULL"
    return [dict(r) for r in connection.execute(sql + " ORDER BY dataset, doc_id", params)]


def record_page(
    connection: sqlite3.Connection,
    compile_id: int,
    dataset: str,
    doc_id: str,
    *,
    page_id: str | None,
    error: str | None,
) -> None:
    connection.execute(
        "UPDATE compile_doc SET page_id = ?, error = ? WHERE compile_id = ? AND dataset = ? AND doc_id = ?",
        (page_id, error, compile_id, dataset, doc_id),
    )


def page_to_doc(
    connection: sqlite3.Connection, compile_id: int, dataset: str
) -> dict[str, str]:
    """``page_id -> doc_id``，供评测把响应里的 sourcePageId 反查回语料文档。"""
    return {
        row["page_id"]: row["doc_id"]
        for row in connection.execute(
            "SELECT page_id, doc_id FROM compile_doc "
            "WHERE compile_id = ? AND dataset = ? AND page_id IS NOT NULL",
            (compile_id, dataset),
        )
    }


def compile_stats(connection: sqlite3.Connection, compile_id: int) -> dict[str, Any]:
    rows = connection.execute(
        """
        SELECT dataset,
               COUNT(*) AS docs,
               SUM(CASE WHEN page_id IS NOT NULL THEN 1 ELSE 0 END) AS imported,
               SUM(is_gold) AS gold
        FROM compile_doc WHERE compile_id = ? GROUP BY dataset ORDER BY dataset
        """,
        (compile_id,),
    )
    per_dataset = {row["dataset"]: dict(row) for row in rows}
    samples = connection.execute(
        "SELECT dataset, COUNT(*) AS n FROM compile_sample WHERE compile_id = ? GROUP BY dataset",
        (compile_id,),
    )
    for row in samples:
        per_dataset.setdefault(row["dataset"], {"dataset": row["dataset"]})["samples"] = row["n"]
    return per_dataset


def workspace_mismatch(
    connection: sqlite3.Connection, compile_id: int, resolved_workspace_id: str | None
) -> str | None:
    """这次编译的空间是否还在当前连接解析出的 workspace 里。不一致时返回原因。

    ``resolved_workspace_id`` 取 ``users/me`` 的响应，不是配置项。
    不一致时查询不报错，只会每条都召回不到。
    """
    run = get_compile_run(connection, compile_id)
    if run is None:
        return f"编译 #{compile_id} 不存在"
    recorded = run["workspace_id"]

    if not recorded or not resolved_workspace_id:
        return None
    if recorded == resolved_workspace_id:
        return None
    return (
        f"编译 {run['run_id']!r} 的空间 {run['space_id']} 属于 workspace {recorded}，"
        f"而当前连接解析出的是 {resolved_workspace_id}。这些 page_id 在这里解析不到，"
        "查询不会报错但每条都召回不到。请把连接指回原来的部署/账号，"
        "或清理这次编译重新编。"
    )


def compile_ready(connection: sqlite3.Connection, compile_id: int) -> dict[str, Any]:
    """能不能拿这次编译去查询，返回 ``{ready, reasons}``。"""
    run = get_compile_run(connection, compile_id)
    if run is None:
        return {"ready": False, "reasons": ["编译记录不存在"]}
    reasons: list[str] = []
    if run["status"] != STATUS_SUCCEEDED:
        reasons.append(f"编译状态为 {run['status']}，未成功结束")
    if not run["space_id"]:
        reasons.append("没有 Akasha 空间")
    missing = connection.execute(
        "SELECT COUNT(*) AS n FROM compile_doc WHERE compile_id = ? AND page_id IS NULL",
        (compile_id,),
    ).fetchone()["n"]
    if missing:
        reasons.append(f"{missing} 篇语料没有导入成功")
    quality = loads(run["quality_json"]) or {}
    if quality.get("passed") is not True:
        reasons.append("编译质量闸门未通过")
    return {"ready": not reasons, "reasons": reasons}


# ------------------------------------------------------------------ 查询层


def create_query_run(
    connection: sqlite3.Connection,
    *,
    name: str,
    compile_id: int,
    score_threshold: float | None,
    concurrency: int,
    model_configs: Any,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO query_run
            (name, compile_id, score_threshold, concurrency,
             model_configs_json, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            compile_id,
            score_threshold,
            concurrency,
            dumps(model_configs),
            STATUS_RUNNING,
            utc_now(),
        ),
    )
    return int(cursor.lastrowid or 0)


def get_query_run(connection: sqlite3.Connection, query_id: int) -> dict[str, Any] | None:
    row = connection.execute("SELECT * FROM query_run WHERE id = ?", (query_id,)).fetchone()
    return dict(row) if row else None


def query_run_by_name(connection: sqlite3.Connection, name: str) -> dict[str, Any] | None:
    row = connection.execute("SELECT * FROM query_run WHERE name = ?", (name,)).fetchone()
    return dict(row) if row else None


def list_query_runs(
    connection: sqlite3.Connection, compile_id: int | None = None
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM query_run"
    params: tuple[Any, ...] = ()
    if compile_id is not None:
        sql += " WHERE compile_id = ?"
        params = (compile_id,)
    return [dict(r) for r in connection.execute(sql + " ORDER BY id DESC", params)]


def set_query_status(
    connection: sqlite3.Connection, query_id: int, status: str, *, finished: bool = False
) -> None:
    connection.execute(
        "UPDATE query_run SET status = ?, finished_at = ? WHERE id = ?",
        (status, utc_now() if finished else None, query_id),
    )


def delete_query_run(connection: sqlite3.Connection, query_id: int) -> int:
    return connection.execute("DELETE FROM query_run WHERE id = ?", (query_id,)).rowcount


def freeze_query_samples(
    connection: sqlite3.Connection, query_id: int, samples: list[dict[str, Any]]
) -> None:
    """固化这一轮要问哪些样本，续跑据此算待办。"""
    connection.executemany(
        "INSERT OR IGNORE INTO query_sample (query_id, sample_id, dataset) VALUES (?, ?, ?)",
        [(query_id, s["sample_id"], s["dataset"]) for s in samples],
    )


def query_samples(connection: sqlite3.Connection, query_id: int) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in connection.execute(
            "SELECT * FROM query_sample WHERE query_id = ? ORDER BY sample_id", (query_id,)
        )
    ]


def pending_query_samples(connection: sqlite3.Connection, query_id: int) -> list[dict[str, Any]]:
    """待问的样本 = 固化选择 - 已落库响应，即续跑的依据。"""
    return [
        dict(r)
        for r in connection.execute(
            """
            SELECT qs.sample_id, qs.dataset, s.question
            FROM query_sample qs
            JOIN sample s ON s.sample_id = qs.sample_id
            LEFT JOIN query_response qr
                ON qr.query_id = qs.query_id AND qr.sample_id = qs.sample_id
            WHERE qs.query_id = ? AND qr.sample_id IS NULL
            ORDER BY qs.sample_id
            """,
            (query_id,),
        )
    ]


def record_response(
    connection: sqlite3.Connection,
    query_id: int,
    *,
    sample_id: str,
    dataset: str,
    question: str,
    http_status: int,
    latency_ms: int | None,
    error: str | None,
    response: Any,
) -> None:
    mode = response.get("answerMode") if isinstance(response, dict) else None
    connection.execute(
        """
        INSERT INTO query_response
            (query_id, sample_id, dataset, question, http_status,
             latency_ms, error, answer_mode, response_json, requested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(query_id, sample_id) DO UPDATE SET
            http_status = excluded.http_status,
            latency_ms = excluded.latency_ms,
            error = excluded.error,
            answer_mode = excluded.answer_mode,
            response_json = excluded.response_json,
            requested_at = excluded.requested_at
        """,
        (
            query_id,
            sample_id,
            dataset,
            question,
            http_status,
            latency_ms,
            error,
            mode,
            dumps(response) if response is not None else None,
            utc_now(),
        ),
    )


def responses_of(
    connection: sqlite3.Connection, query_id: int, dataset: str | None = None
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM query_response WHERE query_id = ?"
    params: list[Any] = [query_id]
    if dataset:
        sql += " AND dataset = ?"
        params.append(dataset)
    return [
        {**{k: v for k, v in dict(row).items() if k != "response_json"},
         "response": loads(row["response_json"])}
        for row in connection.execute(sql + " ORDER BY sample_id", params)
    ]


def response_of(
    connection: sqlite3.Connection, query_id: int, sample_id: str
) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT * FROM query_response WHERE query_id = ? AND sample_id = ?",
        (query_id, sample_id),
    ).fetchone()
    if row is None:
        return None
    return {
        **{k: v for k, v in dict(row).items() if k != "response_json"},
        "response": loads(row["response_json"]),
    }


def delete_failed_responses(connection: sqlite3.Connection, query_id: int) -> int:
    """删失败行让它们重试，成功的保留。"""
    return connection.execute(
        "DELETE FROM query_response WHERE query_id = ? "
        "AND (http_status < 200 OR http_status >= 300)",
        (query_id,),
    ).rowcount


def query_stats(connection: sqlite3.Connection, query_id: int) -> dict[str, Any]:
    rows = connection.execute(
        """
        SELECT dataset, COUNT(*) AS responses,
               SUM(CASE WHEN http_status BETWEEN 200 AND 299 THEN 0 ELSE 1 END) AS failures,
               AVG(latency_ms) AS latency_mean,
               MAX(latency_ms) AS latency_max
        FROM query_response WHERE query_id = ? GROUP BY dataset ORDER BY dataset
        """,
        (query_id,),
    )
    return {row["dataset"]: dict(row) for row in rows}


def response_datasets(connection: sqlite3.Connection, query_id: int) -> list[str]:
    return [
        row["dataset"]
        for row in connection.execute(
            "SELECT DISTINCT dataset FROM query_response WHERE query_id = ? ORDER BY dataset",
            (query_id,),
        )
    ]


# ------------------------------------------------------------------ 评测层


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


def set_eval_status(
    connection: sqlite3.Connection, eval_id: int, status: str, *, finished: bool = False
) -> None:
    connection.execute(
        "UPDATE eval_run SET status = ?, finished_at = ? WHERE id = ?",
        (status, utc_now() if finished else None, eval_id),
    )


def delete_eval_run(connection: sqlite3.Connection, eval_id: int) -> int:
    return connection.execute("DELETE FROM eval_run WHERE id = ?", (eval_id,)).rowcount


def clear_eval_results(connection: sqlite3.Connection, eval_id: int, dataset: str) -> None:
    """重算一个数据集前先清它的旧结果，免得汇总把两轮混在一起。"""
    for table in ("sample_metric", "sample_eval", "metric_summary", "dataset_eval"):
        connection.execute(
            f"DELETE FROM {table} WHERE eval_id = ? AND dataset = ?", (eval_id, dataset)
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
        [
            (eval_id, sample_id, dataset, name, float(value))
            for name, value in metrics.items()
        ],
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
        {**{k: v for k, v in dict(row).items() if k != "detail_json"},
         "detail": loads(row["detail_json"], {})}
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
        {**{k: v for k, v in dict(row).items() if k != "detail_json"},
         "detail": loads(row["detail_json"])}
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
        for row in connection.execute(
            sql + " GROUP BY se.dataset ORDER BY se.dataset", params
        )
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


# ------------------------------------------------------------------ 归因层


def create_attribution_run(
    connection: sqlite3.Connection,
    *,
    name: str,
    eval_id: int,
    metric: str,
    sample_limit: int,
    provider_id: int | None,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO attribution_run
            (name, eval_id, metric, sample_limit, provider_id, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (name, eval_id, metric, sample_limit, provider_id, STATUS_RUNNING, utc_now()),
    )
    return int(cursor.lastrowid or 0)


def get_attribution_run(
    connection: sqlite3.Connection, attribution_id: int
) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT * FROM attribution_run WHERE id = ?", (attribution_id,)
    ).fetchone()
    return dict(row) if row else None


def attribution_run_by_name(
    connection: sqlite3.Connection, name: str
) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT * FROM attribution_run WHERE name = ?", (name,)
    ).fetchone()
    return dict(row) if row else None


def list_attribution_runs(
    connection: sqlite3.Connection, eval_id: int | None = None
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM attribution_run"
    params: tuple[Any, ...] = ()
    if eval_id is not None:
        sql += " WHERE eval_id = ?"
        params = (eval_id,)
    return [dict(r) for r in connection.execute(sql + " ORDER BY id DESC", params)]


def set_attribution_status(
    connection: sqlite3.Connection, attribution_id: int, status: str, *, finished: bool = False
) -> None:
    connection.execute(
        "UPDATE attribution_run SET status = ?, finished_at = ? WHERE id = ?",
        (status, utc_now() if finished else None, attribution_id),
    )


def delete_attribution_run(connection: sqlite3.Connection, attribution_id: int) -> int:
    return connection.execute(
        "DELETE FROM attribution_run WHERE id = ?", (attribution_id,)
    ).rowcount


def record_attribution(
    connection: sqlite3.Connection,
    attribution_id: int,
    *,
    sample_id: str,
    dataset: str,
    root_cause: str,
    evidence: dict[str, Any],
    narrative: str | None,
    rule_based: bool,
    latency_ms: int | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO attribution_result
            (attribution_id, sample_id, dataset, root_cause, evidence_json,
             narrative, rule_based, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(attribution_id, sample_id) DO UPDATE SET
            root_cause = excluded.root_cause,
            evidence_json = excluded.evidence_json,
            narrative = excluded.narrative,
            rule_based = excluded.rule_based,
            latency_ms = excluded.latency_ms
        """,
        (
            attribution_id,
            sample_id,
            dataset,
            root_cause,
            dumps(evidence),
            narrative,
            int(rule_based),
            latency_ms,
        ),
    )


def attribution_results(
    connection: sqlite3.Connection, attribution_id: int
) -> list[dict[str, Any]]:
    return [
        {**{k: v for k, v in dict(row).items() if k != "evidence_json"},
         "evidence": loads(row["evidence_json"], {})}
        for row in connection.execute(
            "SELECT * FROM attribution_result WHERE attribution_id = ? ORDER BY sample_id",
            (attribution_id,),
        )
    ]


def attributed_sample_ids(connection: sqlite3.Connection, attribution_id: int) -> set[str]:
    return {
        row["sample_id"]
        for row in connection.execute(
            "SELECT sample_id FROM attribution_result WHERE attribution_id = ?",
            (attribution_id,),
        )
    }


def cause_counts(connection: sqlite3.Connection, attribution_id: int) -> dict[str, int]:
    return {
        row["root_cause"]: row["n"]
        for row in connection.execute(
            "SELECT root_cause, COUNT(*) AS n FROM attribution_result "
            "WHERE attribution_id = ? GROUP BY root_cause ORDER BY n DESC",
            (attribution_id,),
        )
    }
