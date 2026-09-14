"""Akasha 四项模型配置的规范化与比对。

编译时固化一份快照，查询前拿现在的配置与它比 —— 换了 embedding 之后旧 chunk
的 ``embedding_profile`` 对不上，那些 chunk 永远召回不到，而评测会照常算出
一份看着合理的坏报告。这种失效不报错，所以必须显式比对。
"""

from __future__ import annotations

from typing import Any

FEATURES = ("compiler", "embedding", "answer", "image")

# 只有这几项影响结果。apiKeySet 属于部署状态，不算实验配置；
# provider 只有 openai-compatible 一个合法取值，比它等于没比。
_FIELDS = ("feature", "model", "baseUrl", "parameters")


def normalize(model_configs: Any) -> list[dict[str, Any]]:
    """整成稳定形状：只留影响结果的字段，按 feature 排序。"""
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
    """哪些项与快照不一致。``embedding`` 不一致必须拒绝执行。"""
    return {feature: not matches(current, snapshot, feature) for feature in FEATURES}
