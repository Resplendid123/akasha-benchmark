"""路由共用的依赖与取数辅助。

一条贯穿的口径：**读走库，写只有「起任务」「改配置」「加标注」三类。**
阶段计算一律不在请求里跑 —— 15 小时的 ingest 不能挂在一个 HTTP 连接上。
"""

from __future__ import annotations

import sqlite3
from typing import Any

from akasha_benchmark.config import AkashaConfig, load_config
from akasha_benchmark.store import connect
from fastapi import HTTPException, Request

from ..lineage import LineageReader, LineageUnavailable
from ..settings import Settings


def settings_of(request: Request) -> Settings:
    return request.app.state.settings


def db(request: Request) -> sqlite3.Connection:
    """只读连接。读路径一律走它 —— 写要显式用 writable。"""
    return connect(settings_of(request).db_path, read_only=True)


def writable(request: Request) -> sqlite3.Connection:
    return connect(settings_of(request).db_path)


def config_of(request: Request) -> AkashaConfig:
    """那一份 Akasha 连接配置。

    库不存在时返回默认值，报错留给 ``require_credentials`` —— 那里的提示能指向
    缺的具体是哪一项。
    """
    path = settings_of(request).db_path
    if not path.is_file():
        return load_config(None)
    connection = connect(path, read_only=True)
    try:
        return load_config(connection)
    finally:
        connection.close()


def reader(request: Request) -> LineageReader:
    """只读 PG 的血缘读取器。没配 database_url 时给一句明确的 503。"""
    try:
        return LineageReader(config_of(request).database_url)
    except LineageUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


def strip_json(row: dict[str, Any]) -> dict[str, Any]:
    """去掉 ``*_json`` 原始列。调用方负责把需要的那些解开后单独加回来。"""
    return {k: v for k, v in row.items() if not k.endswith("_json")}
