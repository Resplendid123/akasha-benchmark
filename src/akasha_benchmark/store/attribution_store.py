"""归因层存取。"""

from __future__ import annotations

import sqlite3
from typing import Any

from .db import dumps, loads, utc_now
from .run_store import STATUS_RUNNING, get_run


def create_attribution_run(
    connection: sqlite3.Connection,
    *,
    name: str,
    eval_id: int,
    report_provider_id: int | None,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO attribution_run
            (name, eval_id, report_provider_id, status, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, eval_id, report_provider_id, STATUS_RUNNING, utc_now()),
    )
    return int(cursor.lastrowid or 0)


def get_attribution_run(
    connection: sqlite3.Connection, attribution_id: int
) -> dict[str, Any] | None:
    return get_run(connection, "attribution", attribution_id)


def attribution_run_by_name(connection: sqlite3.Connection, name: str) -> dict[str, Any] | None:
    row = connection.execute("SELECT * FROM attribution_run WHERE name = ?", (name,)).fetchone()
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


def delete_attribution_run(connection: sqlite3.Connection, attribution_id: int) -> int:
    return connection.execute(
        "DELETE FROM attribution_run WHERE id = ?", (attribution_id,)
    ).rowcount


def record_report(
    connection: sqlite3.Connection,
    attribution_id: int,
    *,
    report: str | None,
    error: str | None,
    latency_ms: int | None,
) -> None:
    """保存整轮评测的模型分析；它属于运行，不属于任何单条样本。"""
    connection.execute(
        "UPDATE attribution_run SET report = ?, report_error = ?, report_latency_ms = ? "
        "WHERE id = ?",
        (report, error, latency_ms, attribution_id),
    )


def record_attribution(
    connection: sqlite3.Connection,
    attribution_id: int,
    *,
    sample_id: str,
    root_cause: str,
    evidence: dict[str, Any],
) -> None:
    connection.execute(
        """
        INSERT INTO attribution_result
            (attribution_id, sample_id, root_cause, evidence_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(attribution_id, sample_id) DO UPDATE SET
            root_cause = excluded.root_cause,
            evidence_json = excluded.evidence_json
        """,
        (
            attribution_id,
            sample_id,
            root_cause,
            dumps(evidence),
        ),
    )


def attribution_results(
    connection: sqlite3.Connection, attribution_id: int
) -> list[dict[str, Any]]:
    return [
        {
            **{k: v for k, v in dict(row).items() if k != "evidence_json"},
            "evidence": loads(row["evidence_json"], {}),
        }
        for row in connection.execute(
            """
            SELECT result.*, sample.dataset
            FROM attribution_result result
            JOIN attribution_run run ON run.id = result.attribution_id
            JOIN sample_eval sample
              ON sample.eval_id = run.eval_id AND sample.sample_id = result.sample_id
            WHERE result.attribution_id = ?
            ORDER BY result.sample_id
            """,
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
