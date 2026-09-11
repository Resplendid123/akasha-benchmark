"""归因层：完整链路 + 自动归因。

归因分两段（见 :mod:`..attribution`）：规则给分类，模型给因果叙述。
规则不需要任何模型配置就能出结果，所以这一层在零配置下也能用。

链路视图本身在 :mod:`.lineage` 里 —— 这个路由负责「把链路喂给判据」那一步。
"""

from __future__ import annotations

from typing import Any

from akasha_benchmark.store import repo
from fastapi import APIRouter, Body, HTTPException, Query, Request

from .. import attribution
from ..lineage import BadPageId, LineageUnavailable
from ._common import db, writable
from .lineage import sample_lineage
from .samples import sample_detail_of

router = APIRouter(prefix="/api")


@router.get("/badcase/{eval_layer_id}")
def list_analyses(request: Request, eval_layer_id: int) -> dict[str, Any]:
    """这一层已有的归因结论，按根因分组计数。"""
    with db(request) as connection:
        if repo.get_eval_layer(connection, eval_layer_id) is None:
            raise HTTPException(404, f"no eval layer #{eval_layer_id}")
        analyses = repo.badcase_analyses(connection, eval_layer_id)
        counts = repo.badcase_cause_counts(connection, eval_layer_id)

    return {
        "eval_layer_id": eval_layer_id,
        "count_by_root_cause": counts,
        "analyses": [
            {**a, "remedy": attribution.REMEDIES.get(a["root_cause"])} for a in analyses
        ],
        "causes": {
            cause: {"remedy": remedy}
            for cause, remedy in attribution.REMEDIES.items()
        },
    }


def _lineage_for(request: Request, eval_layer_id: int, sample_id: str) -> dict[str, Any] | None:
    """取一条样本的链路。没配只读库时返回 None 而不是报错。

    归因在链路不可用时仍然要出结果 —— 少了 ``compiled_away`` 那一条判据，
    但其余判据照旧成立。让整个归因因为缺一个可选配置而失败不划算。

    连接由 sample_lineage 自己按层解析，所以这里只需要判「能不能连上」。
    """
    try:
        return sample_lineage(request, eval_layer_id, sample_id)
    except HTTPException as exc:
        if exc.status_code in {503, 400}:
            return None
        raise
    except (LineageUnavailable, BadPageId):
        return None


@router.post("/badcase/{eval_layer_id}/{sample_id}")
def analyze_sample(
    request: Request,
    eval_layer_id: int,
    sample_id: str,
    payload: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    """归因一条样本。

    ``use_model=false`` 只跑规则。分析模型没配好时也自动退到规则，
    并把原因记进 ``evidence.model_error`` —— 不静默丢掉那次失败。
    """
    use_model = bool(payload.get("use_model", True))
    provider_label = payload.get("provider_label") or None

    lineage = _lineage_for(request, eval_layer_id, sample_id)
    with writable(request) as connection:
        sample = sample_detail_of(connection, eval_layer_id, sample_id)
        return attribution.analyze(
            connection,
            eval_layer_id,
            sample,
            lineage,
            provider_label=provider_label,
            use_model=use_model,
        )


@router.post("/badcase/{eval_layer_id}/batch/worst")
def analyze_worst(
    request: Request,
    eval_layer_id: int,
    metric: str = "recall@5",
    dataset: str | None = None,
    limit: int = Query(10, le=50),
    payload: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    """批量归因某个指标最差的 N 条。**归因层的默认入口。**

    上限压到 50：带模型的归因每条一次 LLM 调用，一个手滑的 500 会烧掉一笔钱。
    真要跑全量就多点几次 —— 那个摩擦是故意的。
    """
    from akasha_benchmark.metrics import registry

    try:
        definition = registry.get_metric(metric)
    except KeyError as exc:
        raise HTTPException(422, str(exc)) from exc

    use_model = bool(payload.get("use_model", True))
    provider_label = payload.get("provider_label") or None

    with db(request) as connection:
        rows = repo.samples_ranked_by(
            connection,
            eval_layer_id,
            metric,
            dataset=dataset,
            ascending=definition.higher_is_better,
            limit=limit,
        )

    results: list[dict[str, Any]] = []
    for row in rows:
        lineage = _lineage_for(request, eval_layer_id, row["sample_id"])
        with writable(request) as connection:
            sample = sample_detail_of(connection, eval_layer_id, row["sample_id"])
            results.append(
                attribution.analyze(
                    connection,
                    eval_layer_id,
                    sample,
                    lineage,
                    provider_label=provider_label,
                    use_model=use_model,
                )
            )

    counts: dict[str, int] = {}
    for result in results:
        counts[result["root_cause"]] = counts.get(result["root_cause"], 0) + 1

    return {
        "eval_layer_id": eval_layer_id,
        "metric": metric,
        "analyzed": len(results),
        "count_by_root_cause": counts,
        "results": results,
        "note": (
            "先看 count_by_root_cause 里 generation_fallback 占多少 —— "
            "那些的低分是生成端拒答，不是检索失败。"
        ),
    }
