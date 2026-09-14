"""评测结果与归因结论。

一条贯穿的口径：**样本列表按 answerMode 分组**。``no_match`` / ``general``
无条件返回空 ``retrievedSources``，它们的检索得分按定义为 0。混在一起排序,
「最差的 N 条」会被生成端拒答刷满，而那不是检索失败。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from akasha_benchmark import attribution, textdiff
from akasha_benchmark.config import load_config
from akasha_benchmark.lineage import BadPageId, LineageReader, LineageUnavailable
from akasha_benchmark.metrics import registry
from akasha_benchmark.store import loads, run_store

from ._common import db, writable
from .runs import _public, _reject_if_busy

router = APIRouter(prefix="/api")


@router.get("/evals/{eval_id}")
def eval_detail(request: Request, eval_id: int) -> dict[str, Any]:
    """一次评测的全部汇总，按数据集与 scope 组织。"""
    with db(request) as connection:
        row = run_store.get_eval_run(connection, eval_id)
        if row is None:
            raise HTTPException(404, f"评测 #{eval_id} 不存在")
        query_run = run_store.get_query_run(connection, int(row["query_id"]))

        scopes: dict[str, dict[str, dict[str, float]]] = {}
        for entry in run_store.metric_summaries(connection, eval_id):
            scopes.setdefault(entry["dataset"], {}).setdefault(entry["scope"], {})[
                entry["metric"]
            ] = entry["value"]

        return {
            **_public(row),
            "ks": loads(row["ks_json"], []),
            # 这一轮勾了哪些指标。报告页的列以它为准。
            "metrics": loads(row["metrics_json"], []),
            "query": _public(query_run or {}),
            "datasets": [
                {**entry, "scopes": scopes.get(entry["dataset"], {})}
                for entry in run_store.dataset_evals(connection, eval_id)
            ],
            "judge": run_store.judge_summary(connection, eval_id),
        }


@router.get("/evals/{eval_id}/samples")
def eval_samples(
    request: Request,
    eval_id: int,
    dataset: str | None = None,
    answer_mode: str | None = None,
) -> dict[str, Any]:
    """样本列表，**按 answerMode 分组返回**。分组是默认行为而不是筛选器。"""
    with db(request) as connection:
        if run_store.get_eval_run(connection, eval_id) is None:
            raise HTTPException(404, f"评测 #{eval_id} 不存在")
        rows = run_store.sample_evals(
            connection, eval_id, dataset=dataset, answer_mode=answer_mode
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["answer_mode"] or "missing", []).append(
                {
                    "sample_id": row["sample_id"],
                    "dataset": row["dataset"],
                    "http_status": row["http_status"],
                    "answer": (row["answer"] or "")[:400],
                    "question": row["detail"].get("question"),
                    "metrics": run_store.sample_metrics_of(
                        connection, eval_id, row["sample_id"]
                    ),
                }
            )
    return {
        "by_answer_mode": grouped,
        "counts": {mode: len(items) for mode, items in sorted(grouped.items())},
    }


@router.get("/evals/{eval_id}/worst")
def worst_samples(
    request: Request,
    eval_id: int,
    metric: str = "recall@5",
    dataset: str | None = None,
    limit: int = Query(20, le=200),
) -> dict[str, Any]:
    """按某个指标最差的样本。失败案例入口。"""
    try:
        definition = registry.get_metric(metric)
    except KeyError as exc:
        raise HTTPException(422, str(exc)) from exc

    with db(request) as connection:
        if run_store.get_eval_run(connection, eval_id) is None:
            raise HTTPException(404, f"评测 #{eval_id} 不存在")
        rows = run_store.samples_ranked_by(
            connection,
            eval_id,
            metric,
            dataset=dataset,
            ascending=definition.higher_is_better,
            limit=limit,
        )

    by_mode: dict[str, int] = {}
    for row in rows:
        key = row["answer_mode"] or "missing"
        by_mode[key] = by_mode.get(key, 0) + 1
    return {
        "metric": metric,
        "higher_is_better": definition.higher_is_better,
        "samples": rows,
        "count_by_answer_mode": by_mode,
    }


@router.get("/evals/{eval_id}/samples/{sample_id}")
def sample_detail(request: Request, eval_id: int, sample_id: str) -> dict[str, Any]:
    """单条样本：指标、明细、原始响应、judge 判决。"""
    with db(request) as connection:
        eval_run = run_store.get_eval_run(connection, eval_id)
        if eval_run is None:
            raise HTTPException(404, f"评测 #{eval_id} 不存在")
        row = run_store.sample_eval(connection, eval_id, sample_id)
        if row is None:
            raise HTTPException(404, f"评测 #{eval_id} 里没有样本 {sample_id!r}")
        query_id = int(eval_run["query_id"])
        query_run = run_store.get_query_run(connection, query_id) or {}
        compile_id = int(query_run.get("compile_id") or 0)
        response = run_store.response_of(connection, query_id, sample_id)
        page_to_doc = run_store.page_to_doc(connection, compile_id, row["dataset"])
        doc_to_page = {doc: page for page, doc in page_to_doc.items()}
        verdict = next(
            (v for v in run_store.judge_verdicts(connection, eval_id) if v["sample_id"] == sample_id),
            None,
        )
        return {
            **row,
            "metrics": run_store.sample_metrics_of(connection, eval_id, sample_id),
            "eval_id": eval_id,
            "query_id": query_id,
            "compile_id": compile_id,
            "response": (response or {}).get("response"),
            # doc_id -> page_id：链路视图的入口就是这些 page_id。
            "gold_pages": {
                doc: doc_to_page.get(doc) for doc in row["detail"].get("gold_doc_ids") or []
            },
            "judge_verdict": verdict,
        }


@router.delete("/evals/{eval_id}")
def delete_eval(request: Request, eval_id: int) -> dict[str, Any]:
    """清理一次评测及其下游的归因。"""
    with writable(request) as connection:
        if run_store.get_eval_run(connection, eval_id) is None:
            raise HTTPException(404, f"评测 #{eval_id} 不存在")
        _reject_if_busy(connection, "eval", eval_id)
        removed = run_store.delete_eval_run(connection, eval_id)
    return {"deleted": removed}


# ------------------------------------------------------------------ 归因层


@router.get("/attributions/{attribution_id}")
def attribution_detail(request: Request, attribution_id: int) -> dict[str, Any]:
    """一次归因的全部结论，按根因分组计数。"""
    with db(request) as connection:
        row = run_store.get_attribution_run(connection, attribution_id)
        if row is None:
            raise HTTPException(404, f"归因 #{attribution_id} 不存在")
        results = run_store.attribution_results(connection, attribution_id)
    return {
        **_public(row),
        "count_by_root_cause": {
            cause: sum(1 for r in results if r["root_cause"] == cause)
            for cause in sorted({r["root_cause"] for r in results})
        },
        "results": [
            {**r, "remedy": attribution.REMEDIES.get(r["root_cause"])} for r in results
        ],
        "remedies": attribution.REMEDIES,
    }


@router.delete("/attributions/{attribution_id}")
def delete_attribution(request: Request, attribution_id: int) -> dict[str, Any]:
    with writable(request) as connection:
        if run_store.get_attribution_run(connection, attribution_id) is None:
            raise HTTPException(404, f"归因 #{attribution_id} 不存在")
        _reject_if_busy(connection, "attribution", attribution_id)
        removed = run_store.delete_attribution_run(connection, attribution_id)
    return {"deleted": removed}


@router.get("/lineage/{page_id}")
def lineage(request: Request, page_id: str, question: str = "") -> dict[str, Any]:
    """一篇文档的原文 vs 编译产物并排。

    编译产物才是被检索的文本：向量与词法召回跑在 ``knowledge_chunks`` 上，
    原文在 ``knowledge_source_chunks`` 里，不参与召回。编译丢掉的实词
    如果正好是问题里的词，三条召回路径会同时断，而那不是调参能救的。
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
