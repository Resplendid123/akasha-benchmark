"""编译层存取。"""

from __future__ import annotations

import sqlite3
from typing import Any

from .db import dumps, loads, utc_now
from .run_store import STATUS_RUNNING, STATUS_SUCCEEDED


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


def page_to_doc(connection: sqlite3.Connection, compile_id: int, dataset: str) -> dict[str, str]:
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
