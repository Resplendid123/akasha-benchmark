"""迁移运行器：按文件名顺序跑 ``migrations/*.sql``，跑之前自动备份。

三条来自 PLAN.md §12.7 的硬性要求：

1. **迁移前自动备份**。20 行代码换掉一整类事故。
2. **迁移必须能在有数据的库上跑**，不能只在空库验证过。SQLite 的 ``ALTER TABLE``
   不能删列改类型，复杂改动走「建新表 → 拷数据 → 换名」。
3. 每个版本只跑一次，已跑过的记进 ``schema_migration`` 并校验 checksum ——
   迁移文件被改过就报错，否则两台机器的 schema 会悄悄分叉。

    uv run python -m akasha_benchmark.store.migrate
    uv run python -m akasha_benchmark.store.migrate --status
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from ..io_utils import sha256_text, utc_now
from .db import DEFAULT_DB_PATH, connect

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"


def discover(migrations_dir: Path | None = None) -> list[Path]:
    """按文件名排序返回迁移文件。命名形如 ``001_initial.sql``。"""
    directory = migrations_dir or MIGRATIONS_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"migrations directory not found: {directory}")
    return sorted(directory.glob("*.sql"))


def _ensure_bookkeeping(connection: sqlite3.Connection) -> None:
    """建 ``schema_migration`` 表。

    这张表是**运行器自己的账本**，不是某个迁移的产物。放在 001 里的话，
    任何一份新建的迁移目录都会在写账本时炸掉，而且报的是
    「no such table: schema_migration」—— 一句完全指错方向的错误。
    """
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migration (
            version    TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL,
            checksum   TEXT NOT NULL
        )
        """
    )
    connection.commit()


def _applied(connection: sqlite3.Connection) -> dict[str, str]:
    """已跑过的版本 -> checksum。"""
    return {
        row["version"]: row["checksum"]
        for row in connection.execute("SELECT version, checksum FROM schema_migration")
    }


def backup(db_path: Path, version: str) -> Path | None:
    """把当前库备份成 ``{db}.pre-{version}``。库还不存在时跳过。

    用 SQLite 的 backup API 而不是文件拷贝：WAL 模式下 ``.db`` 的最新内容
    有一部分还在 ``-wal`` 里，单独 ``cp`` 那个 ``.db`` 会得到一份少了尾部
    提交的副本 —— 那种备份在真要用的时候才发现不完整，正是备份最不该有的失效方式。
    """
    if not db_path.is_file():
        return None
    target = db_path.with_name(f"{db_path.name}.pre-{version}")
    source = sqlite3.connect(db_path)
    try:
        destination = sqlite3.connect(target)
        try:
            source.backup(destination)
        finally:
            destination.close()
    finally:
        source.close()
    return target


def migrate(
    db_path: Path | None = None, migrations_dir: Path | None = None, *, verbose: bool = True
) -> list[str]:
    """跑掉所有未应用的迁移，返回本次应用的版本列表。"""
    path = db_path or DEFAULT_DB_PATH
    files = discover(migrations_dir)
    # 必须在 connect() 之前判断：connect() 会把文件建出来，之后 backup() 就会
    # 给一个空库留下 .pre-001_initial —— 一份看起来像备份、实际什么都没有的文件。
    existed = path.is_file()

    connection = connect(path)
    try:
        _ensure_bookkeeping(connection)
        applied = _applied(connection)

        # 先校验已应用的迁移没被改过。改过就停：让 schema 与记录对不上地继续跑,
        # 后面每一个「字段不存在」的报错都会指向错误的方向。
        for file in files:
            version = file.stem
            if version not in applied:
                continue
            checksum = sha256_text(file.read_text(encoding="utf-8"))
            if checksum != applied[version]:
                raise RuntimeError(
                    f"migration {version} was modified after it ran "
                    f"(recorded {applied[version][:12]} != file {checksum[:12]}). "
                    "Add a new migration instead of editing an applied one."
                )

        pending = [f for f in files if f.stem not in applied]
        if not pending:
            if verbose:
                print(f"schema up to date at {path} ({len(applied)} migration(s) applied)")
            return []

        saved = backup(path, pending[0].stem) if existed else None
        if saved and verbose:
            print(f"backed up to {saved.name}")

        done: list[str] = []
        for file in pending:
            version = file.stem
            sql = file.read_text(encoding="utf-8")
            # executescript 会先隐式提交，所以 DDL 不放进 transaction()。
            connection.executescript(sql)
            connection.execute(
                "INSERT INTO schema_migration (version, applied_at, checksum) VALUES (?, ?, ?)",
                (version, utc_now(), sha256_text(sql)),
            )
            connection.commit()
            done.append(version)
            if verbose:
                print(f"applied {version}")
        return done
    finally:
        connection.close()


def status(db_path: Path | None = None, migrations_dir: Path | None = None) -> int:
    path = db_path or DEFAULT_DB_PATH
    if not path.is_file():
        print(f"no database at {path}")
        return 1
    # 只读连接建不了表，所以状态查询自己处理「账本还不存在」。
    connection = connect(path, read_only=True)
    try:
        has_ledger = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migration'"
        ).fetchone()
        applied = _applied(connection) if has_ledger else {}
    finally:
        connection.close()
    for file in discover(migrations_dir):
        mark = "applied" if file.stem in applied else "PENDING"
        print(f"{mark:>8}  {file.stem}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--migrations", type=Path, default=None)
    parser.add_argument("--status", action="store_true", help="只看状态，不执行")
    args = parser.parse_args(argv)

    if args.status:
        return status(args.db, args.migrations)
    try:
        migrate(args.db, args.migrations)
    except (RuntimeError, FileNotFoundError, sqlite3.Error) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
