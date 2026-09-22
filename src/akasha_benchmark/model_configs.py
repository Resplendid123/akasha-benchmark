"""Akasha 四项模型配置的规范化与比对。

编译时固化一份快照，查询前与当前配置逐项比对；所有差异只用于告警和
标记可比性，不阻断查询。
"""

from __future__ import annotations

from typing import Any

FEATURES = ("compiler", "embedding", "answer", "image")

# 只有这几项影响结果。apiKeySet 属于部署状态，provider 只有一个合法取值。
_FIELDS = ("feature", "model", "baseUrl", "parameters")


def normalize(model_configs: Any) -> list[dict[str, Any]]:
    """整成稳定形状：只留 :data:`_FIELDS`，按 feature 排序。"""
    if isinstance(model_configs, dict):
        entries = model_configs.get("configs") or []
    elif isinstance(model_configs, list):
        entries = model_configs
    else:
        return []
    cleaned = [
        {field: entry.get(field) for field in _FIELDS}
        for entry in entries
        if isinstance(entry, dict)
    ]
    return sorted(cleaned, key=lambda e: str(e.get("feature")))


def feature_of(model_configs: Any, feature: str) -> dict[str, Any] | None:
    for entry in normalize(model_configs):
        if entry.get("feature") == feature:
            return entry
    return None


def matches(left: Any, right: Any, feature: str) -> bool:
    return feature_of(left, feature) == feature_of(right, feature)


def drift(current: Any, snapshot: Any) -> dict[str, bool]:
    """哪些项与快照不一致；差异用于告警，不阻断查询。"""
    return {feature: not matches(current, snapshot, feature) for feature in FEATURES}
