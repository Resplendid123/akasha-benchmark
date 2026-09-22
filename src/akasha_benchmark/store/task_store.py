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


def create_task(connection: sqlite3.Connection, *, stage: str, params: dict[str, Any]) -> int:
    cursor = connection.execute(
        "INSERT INTO task (stage, status, params_json, created_at) VALUES (?, ?, ?, ?)",
        (stage, QUEUED, dumps(params), utc_now()),
    )
    return int(cursor.lastrowid or 0)


def transition(
    connection: sqlite3.Connection, task_id: int, status: str, *, error: str | None = None
) -> None:
    """在同一事务内更新任务及其绑定产物的状态。调用方负责提交。"""
    from .run_store import RUN_TABLES, set_run_status

    connection.execute(
        "UPDATE task SET status = ?, error = ?, "
        "started_at = CASE WHEN ? = 'running' THEN COALESCE(started_at, ?) ELSE started_at END, "
        "finished_at = ? WHERE id = ?",
        (status, error, status, utc_now(), None if status in ACTIVE else utc_now(), task_id),
    )
    task = get_task(connection, task_id)
    if task and task["target_kind"] in RUN_TABLES:
        set_run_status(connection, task["target_kind"], task["target_id"], status)


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
    connection: sqlite3.Connection,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    return [
        _task(row)
        for row in connection.execute("SELECT * FROM task ORDER BY id DESC LIMIT ?", (limit,))
    ]


def count_inactive_tasks(connection: sqlite3.Connection) -> int:
    marks = ", ".join("?" for _ in ACTIVE)
    row = connection.execute(
        f"SELECT COUNT(*) AS n FROM task WHERE status NOT IN ({marks})", ACTIVE
    ).fetchone()
    return int(row["n"] if row else 0)


def task_tree(connection: sqlite3.Connection) -> dict[str, Any]:
    """按运行外键构建编译 → 查询 → 评测 → 归因任务树。

    已绑定产物的任务挂到对应运行节点；尚未创建产物的下游任务根据输入参数
    挂到父运行节点。下载、归一化和无法关联到现存运行的任务单独返回。
    """
    from .run_store import STAGE_INPUTS

    specs = (
        ("compile", "compile_run", "run_id", None, None),
        ("query", "query_run", "name", "compile", "compile_id"),
        ("eval", "eval_run", "name", "query", "query_id"),
        ("attribution", "attribution_run", "name", "eval", "eval_id"),
    )
    nodes: dict[tuple[str, int], dict[str, Any]] = {}
    roots: list[dict[str, Any]] = []
    for kind, table, name_column, parent_kind, parent_column in specs:
        for row in connection.execute(f"SELECT * FROM {table} ORDER BY id DESC"):
            record = dict(row)
            node = {
                "kind": kind,
                "id": int(record["id"]),
                "name": record[name_column],
                "status": record["status"],
                "created_at": record["created_at"],
                "finished_at": record["finished_at"],
                "tasks": [],
                "pending_tasks": [],
                "children": [],
            }
            nodes[(kind, node["id"])] = node
            if parent_kind is None:
                roots.append(node)
            else:
                parent = nodes.get((parent_kind, int(record[parent_column])))
                if parent is not None:
                    parent["children"].append(node)

    unlinked: list[dict[str, Any]] = []
    tasks = list_tasks(connection, limit=1_000_000)
    for task in tasks:
        target_kind = task.get("target_kind")
        target_id = task.get("target_id")
        target = (
            nodes.get((str(target_kind), int(target_id)))
            if target_kind is not None and target_id is not None
            else None
        )
        if target is not None:
            target["tasks"].append(task)
            continue

        input_spec = STAGE_INPUTS.get(str(task["stage"]))
        parent = None
        if target_kind is None and input_spec is not None:
            parent_kind, parameter = input_spec
            parent_id = (task.get("params") or {}).get(parameter)
            try:
                parent = nodes.get((parent_kind, int(parent_id)))
            except (TypeError, ValueError):
                parent = None
        if parent is not None:
            parent["pending_tasks"].append(task)
        else:
            unlinked.append(task)

    return {
        "total_tasks": len(tasks),
        "inactive_total": count_inactive_tasks(connection),
        "compiles": roots,
        "unlinked_tasks": unlinked,
    }


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
    return connection.execute(f"DELETE FROM task WHERE status NOT IN ({marks})", ACTIVE).rowcount


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
    return [dict(row) for row in connection.execute(sql + " ORDER BY id DESC LIMIT ?", params)]
