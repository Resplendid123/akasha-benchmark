"""评测结果与归因结论。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from akasha_benchmark import attribution, textdiff
from akasha_benchmark.config import load_config
from akasha_benchmark.lineage import BadPageId, LineageReader, LineageUnavailable
from akasha_benchmark.store import (
    attribution_store,
    compile_store,
    eval_store,
    loads,
    query_store,
)

from ._common import db, public_run, reject_if_busy, writable

router = APIRouter(prefix="/api")


@router.get("/evals/{eval_id}")
def eval_detail(request: Request, eval_id: int) -> dict[str, Any]:
    """一次评测的全部汇总，按数据集与 scope 组织。"""
    with db(request) as connection:
        row = eval_store.get_eval_run(connection, eval_id)
        if row is None:
            raise HTTPException(404, f"评测 #{eval_id} 不存在")
        query_run = query_store.get_query_run(connection, int(row["query_id"]))

        scopes: dict[str, dict[str, dict[str, float]]] = {}
        for entry in eval_store.metric_summaries(connection, eval_id):
            scopes.setdefault(entry["dataset"], {}).setdefault(entry["scope"], {})[
                entry["metric"]
            ] = entry["value"]

        return {
            **public_run(row),
            "ks": loads(row["ks_json"], []),
            # 这一轮勾了哪些指标，报告页的列以它为准。
            "metrics": loads(row["metrics_json"], []),
            "query": public_run(query_run or {}),
            "datasets": [
                {**entry, "scopes": scopes.get(entry["dataset"], {})}
                for entry in eval_store.dataset_evals(connection, eval_id)
            ],
            "judge": eval_store.judge_summary(connection, eval_id),
        }


@router.get("/evals/{eval_id}/samples/{sample_id}")
def sample_detail(request: Request, eval_id: int, sample_id: str) -> dict[str, Any]:
    """单条样本：指标、明细、原始响应、judge 判决。"""
    with db(request) as connection:
        eval_run = eval_store.get_eval_run(connection, eval_id)
        if eval_run is None:
            raise HTTPException(404, f"评测 #{eval_id} 不存在")
        row = eval_store.sample_eval(connection, eval_id, sample_id)
        if row is None:
            raise HTTPException(404, f"评测 #{eval_id} 里没有样本 {sample_id!r}")
        query_id = int(eval_run["query_id"])
        query_run = query_store.get_query_run(connection, query_id) or {}
        compile_id = int(query_run.get("compile_id") or 0)
        response = query_store.response_of(connection, query_id, sample_id)
        page_to_doc = compile_store.page_to_doc(connection, compile_id, row["dataset"])
        doc_to_page = {doc: page for page, doc in page_to_doc.items()}
        # 一个样本可以有多条 judge 结论。
        verdicts = [
            v for v in eval_store.judge_verdicts(connection, eval_id) if v["sample_id"] == sample_id
        ]
        return {
            **row,
            "metrics": eval_store.sample_metrics_of(connection, eval_id, sample_id),
            "eval_id": eval_id,
            "query_id": query_id,
            "compile_id": compile_id,
            "response": (response or {}).get("response"),
            # doc_id -> page_id，链路视图的入口。
            "gold_pages": {
                doc: doc_to_page.get(doc) for doc in row["detail"].get("gold_doc_ids") or []
            },
            "judge_verdicts": verdicts,
        }


@router.delete("/evals/{eval_id}")
def delete_eval(request: Request, eval_id: int) -> dict[str, Any]:
    """清理一次评测及其下游的归因。"""
    with writable(request) as connection:
        if eval_store.get_eval_run(connection, eval_id) is None:
            raise HTTPException(404, f"评测 #{eval_id} 不存在")
        reject_if_busy(connection, "eval", eval_id)
        removed = eval_store.delete_eval_run(connection, eval_id)
    return {"deleted": removed}


# ------------------------------------------------------------------ 归因层


@router.get("/attributions/{attribution_id}")
def attribution_detail(request: Request, attribution_id: int) -> dict[str, Any]:
    """一次归因的全部结论，按根因分组计数。"""
    with db(request) as connection:
        row = attribution_store.get_attribution_run(connection, attribution_id)
        if row is None:
            raise HTTPException(404, f"归因 #{attribution_id} 不存在")
        results = attribution_store.attribution_results(connection, attribution_id)
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
    return {
        **public_run(row),
        "count_by_root_cause": {
            cause: sum(1 for r in results if r["root_cause"] == cause)
            for cause in sorted({r["root_cause"] for r in results})
        },
        # 规则归因不调模型，没有 latency，不进均值。
        "latency_mean": sum(latencies) / len(latencies) if latencies else None,
        "results": [{**r, "remedy": attribution.REMEDIES.get(r["root_cause"])} for r in results],
        "remedies": attribution.REMEDIES,
    }


@router.delete("/attributions/{attribution_id}")
def delete_attribution(request: Request, attribution_id: int) -> dict[str, Any]:
    with writable(request) as connection:
        if attribution_store.get_attribution_run(connection, attribution_id) is None:
            raise HTTPException(404, f"归因 #{attribution_id} 不存在")
        reject_if_busy(connection, "attribution", attribution_id)
        removed = attribution_store.delete_attribution_run(connection, attribution_id)
    return {"deleted": removed}


@router.get("/lineage/{page_id}")
def lineage(request: Request, page_id: str, question: str = "") -> dict[str, Any]:
    """一篇文档的原文 vs 编译产物并排。

    编译产物才是被检索的文本（``knowledge_chunks``），原文
    （``knowledge_source_chunks``）不参与召回。
    """
    with db(request) as connection:
        config = load_config(connection)
    try:
        chain = LineageReader(config.database_url).lineage(page_id)
    except LineageUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except BadPageId as exc:
        raise HTTPException(400, str(exc)) from exc
    return {**textdiff.build(chain, question), "base_url": config.base_url}
