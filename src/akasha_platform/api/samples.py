"""样本列表与单条明细。

**一条贯穿的口径：样本列表默认按 answerMode 切分**。

run001 上四条 ``recall@5 < 1.0`` 里三条是 ``answerMode: general``（生成端回落,
``retrievedSources`` 被无条件清空），只有一条是真的漏 gold —— 而那一条答案还是
对的。混在一起看会把 1 条检索问题读成 4 条，所以这不是可选筛选器。
"""

from __future__ import annotations

from typing import Any

from akasha_benchmark.metrics import registry
from akasha_benchmark.store import repo
from fastapi import APIRouter, HTTPException, Query, Request

from ._common import db, strip_json

router = APIRouter(prefix="/api")


@router.get("/layers/eval/{eval_layer_id}/samples")
def samples(
    request: Request,
    eval_layer_id: int,
    dataset: str | None = None,
    answer_mode: str | None = None,
    limit: int = Query(100, le=1000),
    offset: int = 0,
) -> dict[str, Any]:
    """样本列表，**按 answerMode 分组返回**。

    分组是默认行为而不是筛选器：``no_match`` / ``general`` 无条件返回空
    ``retrievedSources``，它们的检索得分按定义就是 0。混在一起排序，
    「最差的 N 条」会被生成端拒答刷满，而那不是检索失败。
    """
    with db(request) as connection:
        rows = repo.sample_evals(
            connection,
            eval_layer_id,
            dataset=dataset,
            answer_mode=answer_mode,
            limit=limit,
            offset=offset,
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            entry = {
                **{k: v for k, v in row.items() if k != "detail_json"},
                "metrics": repo.sample_metrics_of(connection, eval_layer_id, row["sample_id"]),
            }
            grouped.setdefault(row["answer_mode"] or "missing", []).append(entry)

        counts = {mode: len(items) for mode, items in sorted(grouped.items())}
    return {
        "by_answer_mode": grouped,
        "counts": counts,
        "note": (
            "no_match 与 general 无条件返回空 retrievedSources，其检索得分按定义为 0。"
            "把它们与 knowledge 混在一起读，会把生成端拒答误当成检索失败。"
        ),
    }


@router.get("/layers/eval/{eval_layer_id}/worst")
def worst_samples(
    request: Request,
    eval_layer_id: int,
    metric: str = "recall@5",
    dataset: str | None = None,
    limit: int = Query(20, le=200),
) -> dict[str, Any]:
    """按某个指标最差的样本。**失败案例入口。**

    返回里必须带 ``answer_mode``，并按它分开计数 —— 见 :func:`samples` 的说明。
    """
    try:
        definition = registry.get_metric(metric)
    except KeyError as exc:
        raise HTTPException(422, str(exc)) from exc

    with db(request) as connection:
        rows = repo.samples_ranked_by(
            connection,
            eval_layer_id,
            metric,
            dataset=dataset,
            ascending=definition.higher_is_better,
            limit=limit,
        )
        analyses = {
            a["sample_id"]: a for a in repo.badcase_analyses(connection, eval_layer_id)
        }

    by_mode: dict[str, int] = {}
    for row in rows:
        key = row["answer_mode"] or "missing"
        by_mode[key] = by_mode.get(key, 0) + 1

    return {
        "metric": metric,
        "higher_is_better": definition.higher_is_better,
        "samples": [
            {**row, "root_cause": (analyses.get(row["sample_id"]) or {}).get("root_cause")}
            for row in rows
        ],
        "count_by_answer_mode": by_mode,
        "note": (
            "先看 count_by_answer_mode：非 knowledge 的那些是生成端回落，"
            "它们的低分不是检索问题。"
        ),
    }


def sample_detail_of(connection, eval_layer_id: int, sample_id: str) -> dict[str, Any]:
    """单条样本的完整明细。归因层也调它，所以抽成函数而不是只有路由。"""
    layer = repo.get_eval_layer(connection, eval_layer_id)
    if layer is None:
        raise HTTPException(404, f"no eval layer #{eval_layer_id}")
    query_layer_id = int(layer["query_layer_id"])
    query_layer = repo.get_query_layer(connection, query_layer_id) or {}
    index_layer_id = int(query_layer["index_layer_id"])

    rows = [
        r for r in repo.sample_evals(connection, eval_layer_id) if r["sample_id"] == sample_id
    ]
    if not rows:
        raise HTTPException(404, f"no sample {sample_id!r} in eval layer #{eval_layer_id}")
    row = rows[0]

    response = repo.response_of(connection, query_layer_id, sample_id)
    subset = {
        s["sample_id"]: s
        for s in repo.subset_samples(connection, index_layer_id, row["dataset"])
    }
    sample = subset.get(sample_id, {})
    gold = list(sample.get("gold_doc_ids") or [])
    page_map = repo.page_to_doc(connection, index_layer_id, row["dataset"])
    doc_to_page = {doc: page for page, doc in page_map.items()}

    verdicts = [
        v for v in repo.judge_verdicts(connection, eval_layer_id) if v["sample_id"] == sample_id
    ]
    analyses = repo.badcase_analyses(connection, eval_layer_id, sample_id)

    return {
        **{k: v for k, v in row.items() if k != "detail_json"},
        "detail": repo.loads(row["detail_json"], {}),
        "metrics": repo.sample_metrics_of(connection, eval_layer_id, sample_id),
        "eval_layer_id": eval_layer_id,
        "query_layer_id": query_layer_id,
        "index_layer_id": index_layer_id,
        "question": sample.get("question"),
        "reference_answers": list(sample.get("answers") or []),
        "metadata": sample.get("metadata") or {},
        "gold_doc_ids": gold,
        # doc_id -> page_id，血缘视图的入口就是这些 page_id。
        "gold_pages": {doc: doc_to_page.get(doc) for doc in gold},
        "response": (response or {}).get("response"),
        "judge_verdicts": [
            {**strip_json(v_row), "reasoning": repo.loads(v_row["reasoning_json"])}
            for v_row in verdicts
        ],
        "annotations": repo.annotations_for(connection, "sample", sample_id),
        "badcase_analysis": analyses[0] if analyses else None,
        "audit": next(
            (
                a
                for a in repo.audit_records(connection, query_layer_id)
                if a["sample_id"] == sample_id
            ),
            None,
        ),
    }


@router.get("/layers/eval/{eval_layer_id}/samples/{sample_id}")
def sample_detail(request: Request, eval_layer_id: int, sample_id: str) -> dict[str, Any]:
    """单条样本：指标、明细、原始响应、judge 判决、归因与标注。"""
    with db(request) as connection:
        return sample_detail_of(connection, eval_layer_id, sample_id)
