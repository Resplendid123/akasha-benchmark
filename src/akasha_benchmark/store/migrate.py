"""Apply SQL migrations with checksum validation and a SQLite backup before changes."""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

from ..io_utils import sha256_text, utc_now
from .db import DEFAULT_DB_PATH, connect

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"

# Checksums identify databases already upgraded to the former six-file schema.
LEGACY_CHECKSUMS = {
    "001_initial": "b5563074335487afebfb828451b80bcc9c5fe6c3fd05a195b108c2633527d1bf",
    "002_platform": "5cb435e1024d16c11162a6d9b00f0cb026001d60ff024037f13e15a6d1340e0b",
    "003_connections": "2e0eac85c1366569cbc09528de03edd600656ff85f32561fc948f12b7b076fa3",
    "004_workspace_from_server": "f1b95af261b0e4a5e80823486e2a6997512f4c3a842bc632f52aa810529ccd2d",
    "005_single_connection": "e1e9b8d1a4edba4ddfaf358cf774ea392ac6357f78b558d2c75f905983bdd9ca",
    "006_drop_slug_prefix": "1307dd186b6c9899bb3e46f5f6dc8b72ce92fd6bb392ddf835936f167ae471a4",
}


def _schema(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        name: " ".join(re.sub(r"--[^\n]*", "", sql).split())
        for name, sql in connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
            "AND name != 'schema_migration'"
        )
    }


def _adopt_baseline(
    connection: sqlite3.Connection,
    path: Path,
    files: list[Path],
    applied: dict[str, str],
) -> bool:
    if not applied or applied.get("001_initial") != LEGACY_CHECKSUMS["001_initial"]:
        return False
    baseline = next((file for file in files if file.stem == "001_initial"), None)
    if baseline is None:
        return False
    sql = baseline.read_text(encoding="utf-8")
    if sha256_text(sql) == LEGACY_CHECKSUMS["001_initial"]:
        return False
    if applied != LEGACY_CHECKSUMS:
        raise RuntimeError(
            "Legacy database must first be upgraded through 006_drop_slug_prefix "
            "using the previous revision."
        )
    expected = sqlite3.connect(":memory:")
    try:
        expected.executescript(sql)
        if _schema(connection) != _schema(expected):
            raise RuntimeError(
                "Database schema differs from the final baseline; no changes made."
            )
    finally:
        expected.close()
    backup(path, "baseline")
    with connection:
        connection.execute("DELETE FROM schema_migration")
        connection.execute(
            "INSERT INTO schema_migration VALUES (?, ?, ?)",
            (baseline.stem, utc_now(), sha256_text(sql)),
        )
    return True


def discover(migrations_dir: Path | None = None) -> list[Path]:
    """按文件名排序返回迁移文件。命名形如 ``001_initial.sql``。"""
    directory = migrations_dir or MIGRATIONS_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"migrations directory not found: {directory}")
    return sorted(directory.glob("*.sql"))


def _ensure_bookkeeping(connection: sqlite3.Connection) -> None:
    """Create the migration ledger independently of application tables."""
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
    """Back up committed data, including WAL contents, using the SQLite backup API."""
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
    db_path: Path | None = None,
    migrations_dir: Path | None = None,
    *,
    verbose: bool = True,
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
        adopted = _adopt_baseline(connection, path, files, applied)
        if adopted:
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
                print(
                    f"schema up to date at {path} ({len(applied)} migration(s) applied)"
                )
            return ["001_initial"] if adopted else []

        saved = backup(path, pending[0].stem) if existed else None
        if saved and verbose:
            print(f"backed up to {saved.name}")

        done: list[str] = ["001_initial"] if adopted else []
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
        if applied == LEGACY_CHECKSUMS and file.stem == "001_initial":
            mark = "LEGACY"
        elif file.stem not in applied:
            mark = "PENDING"
        elif sha256_text(file.read_text(encoding="utf-8")) != applied[file.stem]:
            mark = "CHANGED"
        else:
            mark = "applied"
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
