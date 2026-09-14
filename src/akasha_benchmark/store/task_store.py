"""任务与审计日志。

审计日志只追加：清理任务删 ``task`` 行，``audit_log`` 保留。
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .db import dumps, loads, utc_now

QUEUED = "queued"
RUNNING = "running"
PAUSED = "paused"
SUCCEEDED = "succeeded"
FAILED = "failed"

ACTIVE = (QUEUED, RUNNING)
TERMINAL = (SUCCEEDED, FAILED)


def create_task(
    connection: sqlite3.Connection, *, stage: str, params: dict[str, Any]
) -> int:
    cursor = connection.execute(
        "INSERT INTO task (stage, status, params_json, created_at) VALUES (?, ?, ?, ?)",
        (stage, QUEUED, dumps(params), utc_now()),
    )
    return int(cursor.lastrowid or 0)


def start_task(connection: sqlite3.Connection, task_id: int) -> None:
    connection.execute(
        "UPDATE task SET status = ?, started_at = COALESCE(started_at, ?), "
        "finished_at = NULL, error = NULL WHERE id = ?",
        (RUNNING, utc_now(), task_id),
    )


def finish_task(
    connection: sqlite3.Connection, task_id: int, *, status: str, error: str | None = None
) -> None:
    connection.execute(
        "UPDATE task SET status = ?, error = ?, finished_at = ? WHERE id = ?",
        (status, error, utc_now(), task_id),
    )


def pause_task(connection: sqlite3.Connection, task_id: int) -> None:
    connection.execute(
        "UPDATE task SET status = ?, finished_at = ? WHERE id = ?",
        (PAUSED, utc_now(), task_id),
    )


def set_task_target(
    connection: sqlite3.Connection, task_id: int, kind: str, target_id: int
) -> None:
    connection.execute(
        "UPDATE task SET target_kind = ?, target_id = ? WHERE id = ?",
        (kind, target_id, task_id),
    )


def update_progress(
    connection: sqlite3.Connection,
    task_id: int,
    *,
    done: int,
    total: int | None,
    note: str | None,
) -> None:
    connection.execute(
        "UPDATE task SET progress_done = ?, progress_total = ?, progress_note = ? WHERE id = ?",
        (done, total, note, task_id),
    )


def set_task_chain(
    connection: sqlite3.Connection, task_id: int, *, chain: list[dict[str, Any]], chain_id: int
) -> None:
    """把链上剩余阶段挂到任务上。"""
    connection.execute(
        "UPDATE task SET chain_id = ?, chain_json = ? WHERE id = ?",
        (chain_id, dumps(chain), task_id),
    )


def task_chain(connection: sqlite3.Connection, task_id: int) -> tuple[int | None, list[dict]]:
    """任务的链号与剩余阶段。"""
    row = connection.execute(
        "SELECT chain_id, chain_json FROM task WHERE id = ?", (task_id,)
    ).fetchone()
    if row is None:
        return None, []
    return row["chain_id"], loads(row["chain_json"], []) or []


# chain_json 是运行器的内部账本，不进接口响应。
_HIDDEN = ("params_json", "chain_json")


def _task(row: sqlite3.Row) -> dict[str, Any]:
    return {
        **{k: v for k, v in dict(row).items() if k not in _HIDDEN},
        "params": loads(row["params_json"], {}),
    }


def get_task(connection: sqlite3.Connection, task_id: int) -> dict[str, Any] | None:
    row = connection.execute("SELECT * FROM task WHERE id = ?", (task_id,)).fetchone()
    return _task(row) if row else None


def list_tasks(
    connection: sqlite3.Connection, *, status: str | None = None, limit: int = 100
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM task"
    params: list[Any] = []
    if status:
        sql += " WHERE status = ?"
        params.append(status)
    params.append(limit)
    return [_task(row) for row in connection.execute(sql + " ORDER BY id DESC LIMIT ?", params)]


def active_tasks(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    marks = ", ".join("?" for _ in ACTIVE)
    return [
        _task(row)
        for row in connection.execute(
            f"SELECT * FROM task WHERE status IN ({marks}) ORDER BY id", ACTIVE
        )
    ]


def delete_task(connection: sqlite3.Connection, task_id: int) -> int:
    """删任务记录。在跑的不许删，否则会留下一个没人认领的线程还在写库。"""
    row = connection.execute("SELECT status FROM task WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        return 0
    if row["status"] in ACTIVE:
        raise ValueError(f"任务 #{task_id} 正在运行，请先暂停")
    return connection.execute("DELETE FROM task WHERE id = ?", (task_id,)).rowcount


def delete_inactive_tasks(connection: sqlite3.Connection) -> int:
    marks = ", ".join("?" for _ in ACTIVE)
    return connection.execute(
        f"DELETE FROM task WHERE status NOT IN ({marks})", ACTIVE
    ).rowcount


def log(
    connection: sqlite3.Connection,
    *,
    task_id: int | None,
    stage: str,
    level: str,
    message: str,
) -> None:
    connection.execute(
        "INSERT INTO audit_log (task_id, stage, level, message, at) VALUES (?, ?, ?, ?, ?)",
        (task_id, stage, level, message[:4000], utc_now()),
    )


def task_logs(
    connection: sqlite3.Connection, task_id: int, *, after_id: int = 0, limit: int = 500
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM audit_log WHERE task_id = ? AND id > ? ORDER BY id LIMIT ?",
            (task_id, after_id, limit),
        )
    ]


def audit_logs(
    connection: sqlite3.Connection, *, stage: str | None = None, limit: int = 500
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM audit_log"
    params: list[Any] = []
    if stage:
        sql += " WHERE stage = ?"
        params.append(stage)
    params.append(limit)
    return [
        dict(row) for row in connection.execute(sql + " ORDER BY id DESC LIMIT ?", params)
    ]
