"""实验配置的稳定哈希，排除标签、时间戳、速率和密钥。"""

from __future__ import annotations

import json
from typing import Any

from ..io_utils import sha256_text

# 模型配置里只有这几项影响结果。apiKeySet 是布尔量（不回传 key），
# 但它属于部署状态而不是实验配置，也不进哈希。
MODEL_FIELDS = ("feature", "provider", "model", "baseUrl", "parameters")


def canonical(value: Any) -> str:
    """规范化序列化：键排序、无多余空白、中文不转义。

    同一份配置无论字典插入顺序如何，序列化结果都一样 —— 否则「同配置」
    会因为字段顺序不同而算成两个层。
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_hash(payload: dict[str, Any]) -> str:
    """规范化后取 sha256，返回 16 位短哈希。

    截断到 16 位十六进制（64 位）：这是给人看和给 UI 比对用的，
    不是密码学承诺。碰撞概率在这个规模下可以忽略。
    """
    return sha256_text(canonical(payload))[:16]


def normalize_model_configs(model_configs: Any) -> list[dict[str, Any]]:
    """把 ``/model-configs`` 的响应整成稳定形状，只留影响结果的字段。

    服务端返回的是 ``{"configs": [...]}``，顺序不保证，所以按 feature 排序。
    """
    if isinstance(model_configs, dict):
        entries = model_configs.get("configs") or []
    elif isinstance(model_configs, list):
        entries = model_configs
    else:
        return []
    cleaned = [
        {field: entry.get(field) for field in MODEL_FIELDS}
        for entry in entries
        if isinstance(entry, dict)
    ]
    return sorted(cleaned, key=lambda e: str(e.get("feature")))


def model_config_for(model_configs: Any, feature: str) -> dict[str, Any] | None:
    for entry in normalize_model_configs(model_configs):
        if entry.get("feature") == feature:
            return entry
    return None


def index_layer_hash(*, subset_hash: str, model_configs: Any) -> str:
    """索引层身份 = **实际的文档集** + compiler + embedding。

    ``subset_hash`` 由 :func:`..store.repo.recompute_subset_hash` 按实际文档内容
    算出，不是由抽样配置算出。这一点是有意的：配置寻址的版本会说谎 ——
    随机源里只要有配置之外的东西，或者产物是 ``reindex`` 导进来的历史数据，
    「配置相同」就不再等于「文档相同」，而 UI 会照着哈希把两个不同的子集
    并列做对照。

    只吃 compiler 与 embedding 两项模型配置，因为只有它们改了才必须重编译
    。answer 模型改了不用重编译，所以它属于查询层。

    这个值只有在入库时才算得出来（那时才有模型配置），所以库里的
    ``index_layer.config_hash`` 在入库前是 NULL。
    """
    normalized = normalize_model_configs(model_configs)
    return config_hash(
        {
            "subset": subset_hash,
            "compiler": model_config_for(normalized, "compiler"),
            "embedding": model_config_for(normalized, "embedding"),
        }
    )


def query_layer_hash(
    *,
    index_config_hash: str,
    score_threshold: float | None,
    model_configs: Any,
) -> str:
    """查询层身份 = 挂哪个索引层 + answer 模型 + scoreThreshold。

    **并发度与请求间隔不进哈希**：它们只影响跑多快，不影响跑出什么。
    把它们算进去会让「并发 1 跑的」和「并发 4 跑的」变成两个不可比的层,
    而这两批响应其实是可比的。
    """
    return config_hash(
        {
            "index": index_config_hash,
            "score_threshold": score_threshold,
            "answer": model_config_for(model_configs, "answer"),
        }
    )


def eval_layer_hash(
    *,
    query_config_hash: str,
    ks: tuple[int, ...],
    metrics: list[str],
    judge: dict[str, Any] | None = None,
) -> str:
    """评测层身份 = 挂哪个查询层 + 指标组 + k（+ judge 配置）。"""
    return config_hash(
        {
            "query": query_config_hash,
            "ks": sorted(ks),
            "metrics": sorted(metrics),
            "judge": judge_hash(**judge) if judge else None,
        }
    )


def judge_hash(*, base_url: str, model: str, params: dict[str, Any] | None = None) -> str:
    """judge provider 的身份：**只有 base_url + model**（加可选的采样参数）。

    绝不吃 api_key。换密钥不改变实验语义，而把密钥掺进哈希会让它随每个
    引用这个哈希的地方一起扩散。
    """
    return config_hash({"base_url": base_url, "model": model, "params": params or {}})


def embedding_matches(left: Any, right: Any) -> bool:
    """两份模型配置的 embedding 是否一致。

    这是唯一必须**拒绝执行**的漂移：换模型后旧 chunk 的
    ``embedding_profile`` 对不上，那些 chunk 永远召回不到，而评测会照常算出
    一份「recall 低、拒答率高」的报告 —— 看起来像配置差，实际是索引与
    embedding 错配。这种失败不会报错，只会给出一个看着合理的坏结果。
    """
    return model_config_for(left, "embedding") == model_config_for(right, "embedding")


def compiler_matches(left: Any, right: Any) -> bool:
    """compiler 是否一致。不一致只警告：已编译的产物仍然自洽可用，
    只是与新配置编出来的不可比。"""
    return model_config_for(left, "compiler") == model_config_for(right, "compiler")
