"""从保存的配置解析模型端点。"""

import sqlite3

from ..store import config_store
from .client import JudgeConfigError, JudgeProvider


def resolve_provider(
    connection: sqlite3.Connection, provider_id: int | None, role: str
) -> JudgeProvider:
    """从库里取 provider 配置。没给 id 时取该角色最近更新的一条，凑不齐抛
    :class:`JudgeConfigError`。"""
    record = config_store.get_provider(connection, provider_id) if provider_id else None
    if provider_id is not None and record is None:
        raise JudgeConfigError(f"模型端点 #{provider_id} 已不存在")
    if record is None:
        candidates = config_store.list_providers(connection, role)
        record = candidates[0] if candidates else None
    if record is None:
        raise JudgeConfigError(f"没有配置 {role} 模型端点，请在配置页填写。")
    if record["role"] != role:
        raise JudgeConfigError(f"模型端点 {record['label']!r} 的角色不是 {role}")
    if not (record["api_key"] or "").strip():
        raise JudgeConfigError(f"模型端点 {record['label']!r} 没有 api key。")
    return JudgeProvider(
        provider_id=record["id"],
        base_url=record["base_url"],
        model=record["model"],
        api_key=record["api_key"],
    )
