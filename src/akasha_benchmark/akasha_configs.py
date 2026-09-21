"""Akasha 独立模型配置的选择与应用。"""

from __future__ import annotations

import sqlite3
from typing import Any, Protocol

from .store import config_store, loads


class ModelConfigClient(Protocol):
    def put_model_config(self, feature: str, payload: dict[str, Any]) -> Any: ...


def apply_config_group(
    connection: sqlite3.Connection, client: ModelConfigClient, group_id: int
) -> dict[str, Any]:
    """旧接口兼容：将旧组的四项配置应用到远端。"""
    group = config_store.get_config_group(connection, group_id)
    if group is None:
        raise ValueError(f"配置组 #{group_id} 不存在")
    configs = loads(group["configs_json"], {})
    applied: list[str] = []
    for feature in config_store.AKASHA_FEATURES:
        entry = dict(configs.get(feature) or {})
        client.put_model_config(feature, {"provider": "openai-compatible", **entry})
        applied.append(feature)
    config_store.set_selected_group(connection, group_id)
    return {"id": group_id, "label": group["label"], "applied": applied}


def apply_models(
    connection: sqlite3.Connection,
    client: ModelConfigClient,
    selections: dict[str, int],
) -> dict[str, dict[str, Any]]:
    applied: dict[str, dict[str, Any]] = {}
    for feature, model_id in selections.items():
        if feature not in config_store.AKASHA_FEATURES:
            raise ValueError(f"未知 Akasha 配置项 {feature!r}")
        record = config_store.get_akasha_model(connection, int(model_id))
        if record is None:
            raise ValueError(f"Akasha 模型配置 #{model_id} 不存在")
        if record["feature"] != feature:
            raise ValueError(
                f"模型配置 #{model_id} 属于 {record['feature']}，不能用于 {feature}"
            )
        payload = {
            "provider": "openai-compatible",
            "model": record["model"],
            "baseUrl": record["base_url"],
            "apiKey": record["api_key"],
            "parameters": loads(record["parameters_json"], {}),
        }
        client.put_model_config(feature, payload)
        applied[feature] = record
    return applied


def selection_snapshot(records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        feature: {
            "id": int(record["id"]),
            "label": record["label"],
            "model": record["model"],
            "baseUrl": record["base_url"],
        }
        for feature, record in records.items()
    }
