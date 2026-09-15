"""链路测试：用极小样本把编译到归因四层各跑成一条真实任务。

是 compile/query/evaluate/attribute 四条普通任务，
由运行器按顺序推进，进度、日志、产物归属都和手动起的任务一样。

每一步收尾时校验一次契约（``VERIFY``），不成立就让那条任务失败并断链。
"""

from __future__ import annotations

import uuid
from typing import Any

from ..datasets import DATASET_NAMES
from ..store import (
    attribution_store,
    compile_store,
    data_store,
    eval_store,
    query_store,
)
from . import compile

# 只用有 gold 标注的三组，否则检索族指标全省略，测不到指标计算那一段。
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
MAX_SAMPLES = 3
DEFAULT_METRICS = ("recall", "hit", "em", "f1", "citation_recall")


def _check_query(connection, query_id: int) -> None:
    """查询响应要覆盖全部样本、都成功，且 knowledge 响应带着 retrievedSources。"""
    run = query_store.get_query_run(connection, query_id)
    if run is None:
        raise RuntimeError("查询记录丢失")
    responses = query_store.responses_of(connection, query_id)
    expected = {
        s["sample_id"] for s in compile_store.compile_samples(connection, int(run["compile_id"]))
    }
    if {r["sample_id"] for r in responses} != expected:
        raise RuntimeError("查询响应没有覆盖全部样本")
    failed = [r["sample_id"] for r in responses if not 200 <= r["http_status"] < 300]
    if failed:
        raise RuntimeError(f"{len(failed)} 条查询失败：{failed[:3]}")
    unmapped = [
        r["sample_id"]
        for r in responses
        if isinstance(r["response"], dict)
        and r["response"].get("answerMode") == "knowledge"
        and not (r["response"].get("retrievedSources") or [])
    ]
    if unmapped:
        raise RuntimeError(f"knowledge 响应没有 retrievedSources：{unmapped[:3]}")


def _check_evaluate(connection, eval_id: int) -> None:
    run = eval_store.get_eval_run(connection, eval_id)
    if run is None:
        raise RuntimeError("评测记录丢失")
    query_run = query_store.get_query_run(connection, int(run["query_id"]))
    if query_run is None:
        raise RuntimeError("评测指向的查询记录丢失")
    expected = {
        s["sample_id"]
        for s in compile_store.compile_samples(connection, int(query_run["compile_id"]))
    }
    if {r["sample_id"] for r in eval_store.sample_evals(connection, eval_id)} != expected:
        raise RuntimeError("评测没有覆盖全部样本")


def _check_attribute(connection, attribution_id: int) -> None:
    if not attribution_store.attribution_results(connection, attribution_id):
        raise RuntimeError("归因没有产出结论")


# 阶段名 -> 校验函数，签名是 (连接, 产物 id)。
VERIFY = {
    "query": _check_query,
    "evaluate": _check_evaluate,
    "attribute": _check_attribute,
}


def build(params: dict[str, Any], connection) -> list[dict[str, Any]]:
    """把一次链路测试摊平成四步。``link`` 是上一步产物 id 要填进的参数名。"""
    dataset = str(params.get("dataset") or DATASETS[0])
    if dataset not in DATASETS:
        raise ValueError(f"链路测试仅支持 {DATASETS}")
    if dataset not in DATASET_NAMES:
        raise ValueError(f"未知数据集：{dataset}")
    samples = int(params.get("samples") or 2)
    if not 1 <= samples <= MAX_SAMPLES:
        raise ValueError(f"样本数必须在 1–{MAX_SAMPLES} 之间")
    if data_store.get_dataset(connection, dataset) is None:
        raise ValueError(f"请先归一化 {dataset}")

    run_id = f"smoke{uuid.uuid4().hex[:8]}"
    metrics = list(params.get("metrics") or DEFAULT_METRICS)
    evaluate_params: dict[str, Any] = {
        "name": f"{run_id}-e",
        "metrics": metrics,
        "ks": [2, 5],
    }
    if params.get("judge_provider_id"):
        evaluate_params["judge_provider_id"] = params["judge_provider_id"]

    return [
        {
            "stage": "compile",
            "params": {
                "run_id": run_id,
                "datasets": [dataset],
                "qa_limit": samples,
                "negatives_ratio": 1.0,
                "seed": params.get("seed") or compile.default_seed(),
            },
        },
        {"stage": "query", "link": "compile_id", "params": {"name": f"{run_id}-q"}},
        {"stage": "evaluate", "link": "query_id", "params": evaluate_params},
        {
            "stage": "attribute",
            "link": "eval_id",
            "params": {
                "name": f"{run_id}-a",
                "metric": "recall@5",
                "sample_limit": samples,
                "use_model": bool(params.get("use_model", False)),
                "provider_id": params.get("provider_id"),
            },
        },
    ]
