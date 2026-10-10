from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from .model_configs import feature_of
from .store import attribution_store, compile_store, config_store, eval_store, query_store

_CHILD_LOOKUPS: dict[str, Callable[[sqlite3.Connection, str], dict[str, Any] | None]] = {
    "q": query_store.query_run_by_name,
    "e": eval_store.eval_run_by_name,
    "a": attribution_store.attribution_run_by_name,
}

_UNSAFE = re.compile(r"[^0-9a-z一-鿿]+")


def _today() -> str:
    return datetime.now(UTC).strftime("%m%d")


def _slug(value: str) -> str:
    return _UNSAFE.sub("", value.lower())


def model_slug(
    connection: sqlite3.Connection, configs: dict[str, Any], feature: str
) -> str:
    """远端配置里该 feature 的模型，认回本地配置标签；认不出用模型名。"""
    config = feature_of(configs, feature) or {}
    model = str(config.get("model") or "")
    if not model:
        return "unknown"
    label = config_store.remote_label_map(connection).get(
        (feature, model, str(config.get("baseUrl") or "")), model
    )
    return _slug(label) or _slug(model) or "unknown"


def _pick(candidate: Callable[[int], str], taken: Callable[[str], bool]) -> str:
    index = 1
    while taken(name := candidate(index)):
        index += 1
    return name


def _unique_compile(connection: sqlite3.Connection, stem: str) -> str:
    return _pick(
        lambda index: stem if index == 1 else f"{stem}-{index}",
        lambda name: compile_store.compile_run_by_run_id(connection, name) is not None,
    )


def compile_name(
    connection: sqlite3.Connection, *, datasets: list[str], model: str
) -> str:
    """数据集-编译模型-日期，如 musique-gpt56sol-1008；重名追加 -2。"""
    return _unique_compile(connection, f"{'+'.join(datasets)}-{model}-{_today()}")


def smoke_name(connection: sqlite3.Connection, dataset: str) -> str:
    """链路测试名，如 smoke-musique-1008。"""
    return _unique_compile(connection, f"smoke-{dataset}-{_today()}")


def child_name(
    connection: sqlite3.Connection, parent: str, tag: str, model: str = ""
) -> str:
    """父名加层级序号，如 musique-50-gpt56sol-1008-q1-gpt56sol；序号取未占用的最小值。"""
    lookup = _CHILD_LOOKUPS[tag]
    suffix = f"-{model}" if model else ""
    return _pick(
        lambda index: f"{parent}-{tag}{index}{suffix}",
        lambda name: lookup(connection, name) is not None,
    )


def space_slug(run_id: str) -> str:
    """远端 Akasha space 的 slug；Akasha 只收字母数字。"""
    return f"bench{''.join(c for c in run_id if c.isascii() and c.isalnum())}"[:64]
