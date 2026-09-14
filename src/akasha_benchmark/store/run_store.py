"""运行记录的状态与上下游依赖。分层读写见各 *_store 模块。"""

from __future__ import annotations

import sqlite3
from typing import Any

from .db import utc_now

STATUS_RUNNING = "running"
STATUS_PAUSED = "paused"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"


RUN_TABLES = {
    "compile": "compile_run",
    "query": "query_run",
    "eval": "eval_run",
    "attribution": "attribution_run",
}
RUN_PARENTS = {
    "query": ("compile", "compile_id"),
    "eval": ("query", "query_id"),
    "attribution": ("eval", "eval_id"),
}
STAGE_INPUTS = {
    "query": ("compile", "compile_id"),
    "evaluate": ("query", "query_id"),
    "attribute": ("eval", "eval_id"),
}


def get_run(connection: sqlite3.Connection, kind: str, target_id: int) -> dict[str, Any] | None:
    row = connection.execute(
        f"SELECT * FROM {RUN_TABLES[kind]} WHERE id = ?", (target_id,)
    ).fetchone()
    return dict(row) if row else None


def set_run_status(connection: sqlite3.Connection, kind: str, target_id: int, status: str) -> None:
    connection.execute(
        f"UPDATE {RUN_TABLES[kind]} SET status = ?, finished_at = ? WHERE id = ?",
        (
            STATUS_RUNNING if status == "queued" else status,
            None if status in ("queued", STATUS_RUNNING) else utc_now(),
            target_id,
        ),
    )


def descendants(connection: sqlite3.Connection, kind: str, target_id: int) -> set[tuple[str, int]]:
    """包括当前记录及所有会被级联删除的下游运行。"""
    found = {(kind, target_id)}
    pending = [(kind, target_id)]
    while pending:
        parent_kind, parent_id = pending.pop()
        for child_kind, (parent, column) in RUN_PARENTS.items():
            if parent != parent_kind:
                continue
            for row in connection.execute(
                f"SELECT id FROM {RUN_TABLES[child_kind]} WHERE {column} = ?", (parent_id,)
            ):
                child = (child_kind, row["id"])
                found.add(child)
                pending.append(child)
    return found
