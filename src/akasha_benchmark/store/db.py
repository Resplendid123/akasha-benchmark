"""SQLite 连接与短事务。

库是事实来源（PLAN.md §12 决策 2）。选 SQLite 而不是复用 Akasha 那个 Postgres,
主要理由是 ``normalize`` / ``subset`` / ``evaluate`` 必须继续完全不依赖 Akasha
在线（§12.2），而这三个阶段现在读写的就是这个库。

方向是单向的：**SQLite 可写权威，Akasha 的 PG 只读外来**（决策 3）。

一条贯穿全程的约束：**写事务必须短、逐批提交**。否则 Web 端在 ingest 的
15 小时里读不到进度，决策 1 要的观测就废了。:func:`batched` 就是为这件事存在的。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = REPO_ROOT / "akasha_bench.db"

# 每多少行提交一次。取 200 是因为 ingest 单篇约 40 秒、query 单条 10–14 秒,
# 真正的写入频率远低于这个数；批大小在这里的作用是给批量导入（reindex 1722 行）
# 兜一个上限，而不是给在线阶段限速。
DEFAULT_BATCH = 200


def connect(path: Path | None = None, *, read_only: bool = False) -> sqlite3.Connection:
    """打开连接并设好 pragma。

    ``foreign_keys`` 必须逐连接开 —— SQLite 默认是关的，不开的话
    ``ON DELETE CASCADE`` 和外键约束全部静默失效，而症状是几个月后发现
    一堆指向已删除层的孤儿行。

    WAL 让读写不互斥：Web 端要在 ingest 写库的同时读进度。
    """
    target = path or DEFAULT_DB_PATH
    if read_only:
        if not target.is_file():
            raise FileNotFoundError(f"{target} does not exist; run `make migrate` first")
        connection = sqlite3.connect(f"file:{target.as_posix()}?mode=ro", uri=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(target)

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    if not read_only:
        connection.execute("PRAGMA journal_mode = WAL")
        # NORMAL 而不是 FULL：这些产物都能从上游重算，拿一次断电风险换写入吞吐
        # 是值得的。annotation 与 judge_verdict 不能重算，但它们是交互式写入的
        # 单行，不在批量路径上。
        connection.execute("PRAGMA synchronous = NORMAL")
    return connection


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """一个显式事务。异常时回滚。

    Python 的 sqlite3 在 DDL 前会隐式提交，所以迁移不要走这里。
    """
    try:
        yield connection
    except BaseException:
        connection.rollback()
        raise
    connection.commit()


class Batcher:
    """逐批提交的计数器。

    用法是每写一行调一次 :meth:`tick`，它会在攒够 ``size`` 行时提交。
    这样长任务的进度对并发的读连接是可见的，而不是憋到最后一次性提交 ——
    后者会让 Web 端在整个 ingest 期间看到一个空库。
    """

    def __init__(self, connection: sqlite3.Connection, size: int = DEFAULT_BATCH) -> None:
        self.connection = connection
        self.size = size
        self._pending = 0

    def tick(self) -> None:
        self._pending += 1
        if self._pending >= self.size:
            self.flush()

    def flush(self) -> None:
        if self._pending:
            self.connection.commit()
            self._pending = 0


@contextmanager
def batched(connection: sqlite3.Connection, size: int = DEFAULT_BATCH) -> Iterator[Batcher]:
    """批量写入的上下文。退出时把剩余的行提交掉，异常时回滚未提交的部分。"""
    batcher = Batcher(connection, size)
    try:
        yield batcher
    except BaseException:
        connection.rollback()
        raise
    batcher.flush()


def query_all(connection: sqlite3.Connection, sql: str, params: Any = ()) -> list[sqlite3.Row]:
    return connection.execute(sql, params).fetchall()


def query_one(connection: sqlite3.Connection, sql: str, params: Any = ()) -> sqlite3.Row | None:
    return connection.execute(sql, params).fetchone()


def scalar(connection: sqlite3.Connection, sql: str, params: Any = ()) -> Any:
    row = connection.execute(sql, params).fetchone()
    return row[0] if row is not None else None
