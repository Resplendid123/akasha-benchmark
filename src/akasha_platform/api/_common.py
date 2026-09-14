"""路由共用的数据库上下文与依赖。"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager

from fastapi import Request

from akasha_benchmark.config import AkashaConfig, load_config
from akasha_benchmark.store import connect

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
