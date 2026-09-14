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

# 旧 schema 的标志表。重构后表名全变，遇到它说明这是重构前的库。
LEGACY_TABLES = ("index_layer", "query_layer", "eval_layer")


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


def _is_legacy(path: Path) -> bool:
    if not path.is_file():
        return False
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        names = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        connection.close()
    return any(name in names for name in LEGACY_TABLES)


def _move_aside(path: Path) -> Path:
    """把旧库连同 -wal / -shm 改名留档，返回留档路径。"""
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    target = path.with_name(f"{path.name}.legacy-{stamp}")
    path.replace(target)
    for suffix in ("-wal", "-shm"):
        sidecar = path.with_name(path.name + suffix)
        if sidecar.is_file():
            sidecar.replace(target.with_name(target.name + suffix))
    return target


# 建表之后补的列：表名 -> ((列名, 类型), ...)
ADDED_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "task": (("chain_id", "INTEGER"), ("chain_json", "TEXT")),
    "judge_verdict": (("latency_ms", "INTEGER"),),
    "attribution_result": (("latency_ms", "INTEGER"),),
    "compile_run": (("pace_json", "TEXT"),),
}

# 主键变了的表：表名 -> 新主键。SQLite 改不了主键，只能重建。
REBUILT_KEYS: dict[str, tuple[str, ...]] = {
    "judge_verdict": ("eval_id", "sample_id", "metric"),
}


def _drop_outdated_tables(connection: sqlite3.Connection) -> list[str]:
    """主键与当前 schema 不一致的表，删掉让建表脚本重建。

    只对可重算的表这么做（:data:`REBUILT_KEYS` 里都是评测产物，重跑就能再得到）。
    """
    dropped = []
    for table, expected in REBUILT_KEYS.items():
        info = list(connection.execute(f"PRAGMA table_info({table})"))
        if not info:
            continue
        # table_info 的第 6 列是该列在主键里的序号，0 表示不在主键中。
        current = tuple(row[1] for row in sorted(info, key=lambda r: r[5]) if row[5])
        if current != expected:
            connection.execute(f"DROP TABLE {table}")
            dropped.append(table)
    return dropped


def _add_missing_columns(connection: sqlite3.Connection) -> None:
    """给已存在的表补后加的列。

    必须在建表脚本之前跑：脚本里的索引会引用新列。
    空的 ``table_info`` 表示这张表还不存在，交给建表脚本。
    """
    for table, columns in ADDED_COLUMNS.items():
        existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if not existing:
            continue
        for name, decl in columns:
            if name not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def init_db(path: Path | None = None) -> Path | None:
    """建表。遇到重构前的库先改名留档，返回留档路径。"""
    target = path or DEFAULT_DB_PATH
    archived = _move_aside(target) if _is_legacy(target) else None
    connection = connect(target)
    try:
        _add_missing_columns(connection)
        _drop_outdated_tables(connection)
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        from . import config_store

        config_store.normalize_hosts(connection)
        connection.commit()
    finally:
        connection.close()
    return archived


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """成功提交，异常回滚。"""
    try:
        yield connection
    except BaseException:
        connection.rollback()
        raise
    connection.commit()
