"""SQLite 连接、schema 初始化与短事务。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = REPO_ROOT / "akasha_bench.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
RETIRED_METRICS = frozenset(
    {
        "citation_count",
        "retrieved_count",
        "evidence_entries",
        "graph_neighbor_share",
        "snippet_count",
        "evidence_verifiable_rate",
        "graph_neighbor_snippets",
        "graph_exclusive_gold_count",
    }
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def connect(path: Path | None = None, *, read_only: bool = False) -> sqlite3.Connection:
    """打开连接并设好 pragma。

    ``foreign_keys`` 必须逐连接开，否则 ``ON DELETE CASCADE`` 静默失效。
    WAL 让读写不互斥，任务在写库时前端仍能读进度。
    """
    target = path or DEFAULT_DB_PATH
    if read_only:
        if not target.is_file():
            raise FileNotFoundError(f"{target} does not exist")
        connection = sqlite3.connect(f"file:{target.as_posix()}?mode=ro", uri=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(target, check_same_thread=False)

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    if not read_only:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA busy_timeout = 10000")
    return connection


def init_db(path: Path | None = None) -> None:
    """按 schema.sql 初始化数据库，可重复调用。"""
    connection = connect(path)
    try:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        connection.execute("BEGIN IMMEDIATE")
        try:
            _ensure_model_configuration(connection)
            _ensure_attribution_scope(connection)
            _migrate_graph_neighbor_precision_to_documents(connection)
            _remove_retired_metrics(connection)
        except BaseException:
            connection.rollback()
            raise
        connection.commit()
    finally:
        connection.close()


def _ensure_model_configuration(connection: sqlite3.Connection) -> None:
    """补列并将旧配置组一次性拆为四类独立模型配置。"""
    additions = {
        "compile_run": {
            "compiler_model_id": "INTEGER",
            "embedding_model_id": "INTEGER",
            "image_model_id": "INTEGER",
            "model_selection_json": "TEXT",
        },
        "query_run": {
            "config_group": "TEXT",
            "answer_model_id": "INTEGER",
            "model_selection_json": "TEXT",
        },
        "eval_run": {"concurrency": "INTEGER NOT NULL DEFAULT 1"},
        "attribution_run": {"concurrency": "INTEGER NOT NULL DEFAULT 1"},
    }
    for table, columns in additions.items():
        existing = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
        for name, kind in columns.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")

    groups = [dict(row) for row in connection.execute("SELECT * FROM akasha_config_group")]
    for group in groups:
        configs = loads(group["configs_json"], {})
        for feature in ("compiler", "embedding", "answer", "image"):
            entry = dict(configs.get(feature) or {})
            model = str(entry.get("model") or "").strip()
            base_url = str(entry.get("baseUrl") or "").strip()
            if not model or not base_url:
                continue
            connection.execute(
                """
                INSERT OR IGNORE INTO akasha_model_provider
                    (feature, label, base_url, model, api_key, parameters_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feature,
                    group["label"],
                    base_url,
                    model,
                    str(entry.get("apiKey") or ""),
                    dumps(entry.get("parameters") or {}),
                    group["updated_at"],
                ),
            )

    for feature, column in (
        ("compiler", "compiler_model_id"),
        ("embedding", "embedding_model_id"),
        ("image", "image_model_id"),
    ):
        connection.execute(
            f"""
            UPDATE compile_run SET {column} = (
                SELECT id FROM akasha_model_provider p
                WHERE p.feature=? AND p.label=compile_run.config_group
            ) WHERE {column} IS NULL AND config_group IS NOT NULL
            """,
            (feature,),
        )
    connection.execute(
        """
        UPDATE query_run SET answer_model_id = (
            SELECT id FROM akasha_model_provider p
            WHERE p.feature='answer' AND p.label=query_run.config_group
        ) WHERE answer_model_id IS NULL AND config_group IS NOT NULL
        """
    )


def _ensure_attribution_scope(connection: sqlite3.Connection) -> None:
    """归因固定处理评测全量样本，不再保存排序指标与数量上限。"""
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(attribution_run)")}
    for column in ("metric", "sample_limit"):
        if column in columns:
            connection.execute(f"ALTER TABLE attribution_run DROP COLUMN {column}")


def _remove_retired_metrics(connection: sqlite3.Connection) -> None:
    """清理已删除指标的历史配置与结果，避免旧运行记录阻断启动。"""
    if not RETIRED_METRICS:
        return
    placeholders = ", ".join("?" for _ in RETIRED_METRICS)
    retired = tuple(sorted(RETIRED_METRICS))
    for row in connection.execute("SELECT id, metrics_json FROM eval_run"):
        metrics = loads(row["metrics_json"], [])
        kept = [name for name in metrics if name not in RETIRED_METRICS]
        if kept != metrics:
            connection.execute(
                "UPDATE eval_run SET metrics_json = ? WHERE id = ?",
                (dumps(kept), row["id"]),
            )
    connection.execute(
        f"DELETE FROM sample_metric WHERE metric IN ({placeholders})", retired
    )
    connection.execute(
        f"DELETE FROM metric_summary WHERE metric IN ({placeholders})", retired
    )
    for row in connection.execute(
        "SELECT eval_id, dataset, omitted_metrics_json FROM dataset_eval"
    ):
        omitted = loads(row["omitted_metrics_json"], [])
        kept = [name for name in omitted if name not in RETIRED_METRICS]
        if kept != omitted:
            connection.execute(
                "UPDATE dataset_eval SET omitted_metrics_json = ? "
                "WHERE eval_id = ? AND dataset = ?",
                (dumps(kept), row["eval_id"], row["dataset"]),
            )


def _migrate_graph_neighbor_precision_to_documents(
    connection: sqlite3.Connection,
) -> None:
    """旧值按 snippet 加权；新口径按 doc_id 去重，历史分数不可混用。"""
    name = "graph_neighbor_precision_document_scope"
    if connection.execute(
        "SELECT 1 FROM schema_migration WHERE name = ?", (name,)
    ).fetchone():
        return
    connection.execute(
        "DELETE FROM sample_metric WHERE metric = 'graph_neighbor_precision'"
    )
    connection.execute(
        "DELETE FROM metric_summary WHERE metric = 'graph_neighbor_precision'"
    )
    connection.execute(
        "INSERT INTO schema_migration (name, applied_at) VALUES (?, ?)",
        (name, utc_now()),
    )


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """成功提交，异常回滚。"""
    try:
        yield connection
    except BaseException:
        connection.rollback()
        raise
    connection.commit()
