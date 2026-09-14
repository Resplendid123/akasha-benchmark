"""查询层存取。"""

from __future__ import annotations

import sqlite3
from typing import Any

from .db import dumps, loads, utc_now
from .run_store import STATUS_RUNNING


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
        {
            **{k: v for k, v in dict(row).items() if k != "response_json"},
            "response": loads(row["response_json"]),
        }
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
