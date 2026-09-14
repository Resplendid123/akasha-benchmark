"""路由共用的数据库上下文与依赖。"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from typing import Any

from fastapi import HTTPException, Request

from akasha_benchmark.config import AkashaConfig, load_config
from akasha_benchmark.store import connect, run_store, task_store

from ..settings import Settings
from ..tasks import TaskRunner


def settings_of(request: Request) -> Settings:
    return request.app.state.settings


def runner_of(request: Request) -> TaskRunner:
    return request.app.state.runner


@contextmanager
def db(request: Request) -> Iterator[sqlite3.Connection]:
    """只读连接，请求结束时关闭。"""
    with closing(connect(settings_of(request).db_path, read_only=True)) as connection:
        yield connection


@contextmanager
def writable(request: Request) -> Iterator[sqlite3.Connection]:
    """成功提交、异常回滚，始终关闭。"""
    with closing(connect(settings_of(request).db_path)) as connection, connection:
        yield connection


def config_of(request: Request) -> AkashaConfig:
    with db(request) as connection:
        return load_config(connection)


def public_run(row: dict[str, Any]) -> dict[str, Any]:
    """去掉 ``*_json`` 原始列。需要的那些由调用方解开后单独加回。"""
    return {k: v for k, v in row.items() if not k.endswith("_json")}


def reject_if_busy(connection: sqlite3.Connection, kind: str, target_id: int) -> None:
    """锁住写事务，检查级联影响范围及尚未绑定产物的任务。"""
    if not connection.in_transaction:
        connection.execute("BEGIN IMMEDIATE")
    affected = run_store.descendants(connection, kind, target_id)
    for task in task_store.active_tasks(connection):
        references = {(task["target_kind"], task["target_id"])}
        if source := run_store.STAGE_INPUTS.get(task["stage"]):
            parent, param = source
            references.add((parent, task["params"].get(param)))
        if references & affected:
            raise HTTPException(409, f"任务 #{task['id']} 正在使用这条记录或其下游，请先暂停再清理")
