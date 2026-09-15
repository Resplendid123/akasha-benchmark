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
        {
            **{k: v for k, v in dict(row).items() if k != "evidence_json"},
            "evidence": loads(row["evidence_json"], {}),
        }
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
