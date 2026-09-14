"""归因层：针对某次 编译->查询->评测 链路，逐条推出根因。

规则判据在 :mod:`..attribution`。链路证据（原文 vs 编译产物的 diff）走只读
PostgreSQL，没配 database_url 时跳过那一段判据。
"""

from __future__ import annotations

from typing import Any

from .. import attribution, textdiff
from ..config import load_config
from ..judge.client import JudgeClient, JudgeConfigError, parse_json_object
from ..lineage import BadPageId, LineageReader, LineageUnavailable
from ..metrics import registry
from ..store import run_store
from ..task import TaskContext
from .evaluate import resolve_provider

DEFAULT_METRIC = "recall@5"
DEFAULT_LIMIT = 10
# 带模型的归因每条一次 LLM 调用，所以给条数设上限。
MAX_LIMIT = 50


def _lineage_of(
    reader: LineageReader | None,
    gold_doc_ids: list[str],
    page_by_doc: dict[str, str],
    question: str,
) -> list[dict[str, Any]] | None:
    """每篇 gold 的原文 vs 编译产物 diff。取不到链路时返回 None。"""
    if reader is None:
        return None
    entries: list[dict[str, Any]] = []
    for doc_id in gold_doc_ids:
        page_id = page_by_doc.get(doc_id)
        if not page_id:
            entries.append({"doc_id": doc_id, "page_id": None, "error": "未导入"})
            continue
        try:
            chain = reader.lineage(page_id)
        except LineageUnavailable:
            # 连不上只读库，判据退回不含 compiled_away 的那套。
            return None
        except BadPageId as exc:
            entries.append({"doc_id": doc_id, "page_id": page_id, "error": str(exc)})
            continue
        built = textdiff.build(chain, question)
        entries.append(
            {
                "doc_id": doc_id,
                "page_id": page_id,
                "diff": built["diff"],
                "question_terms_lost": built["question_terms_lost"],
                "source_text": built["source"]["text"][:2000],
                "compiled_text": built["compiled"]["text"][:2000],
                "verdict": built["verdict"],
            }
        )
    return entries


def run(ctx: TaskContext) -> None:
    params = ctx.params
    eval_id = int(params.get("eval_id") or 0)
    if not eval_id:
        raise ValueError("请选择一次评测")

    eval_run = run_store.get_eval_run(ctx.db, eval_id)
    if eval_run is None:
        raise ValueError(f"评测 #{eval_id} 不存在")
    query_run = run_store.get_query_run(ctx.db, int(eval_run["query_id"]))
    if query_run is None:
        raise ValueError("这次评测对应的查询记录已不存在")
    compile_id = int(query_run["compile_id"])

    metric = str(params.get("metric") or DEFAULT_METRIC)
    try:
        definition = registry.get_metric(metric)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    limit = min(int(params.get("sample_limit") or DEFAULT_LIMIT), MAX_LIMIT)
    use_model = bool(params.get("use_model", True))

    provider = None
    provider_id = params.get("provider_id")
    provider_id = int(provider_id) if provider_id else None
    if use_model:
        try:
            provider = resolve_provider(ctx.db, provider_id, "attribution")
        except JudgeConfigError as exc:
            # 模型没配好不算失败，规则结论仍是一条有效归因。
            ctx.log(f"未使用模型：{exc}", "warn")

    reader: LineageReader | None = None
    try:
        reader = LineageReader(load_config(ctx.db).database_url)
    except LineageUnavailable as exc:
        ctx.log(f"链路证据不可用（compiled_away 判不了）：{exc}", "warn")

    name = str(params.get("name") or "").strip() or f"{eval_run['name']}-a"
    existing = run_store.attribution_run_by_name(ctx.db, name)
    if existing:
        if int(existing["eval_id"]) != eval_id:
            raise ValueError(f"归因记录 {name!r} 属于另一次评测，请换个名称")
        attribution_id = int(existing["id"])
        run_store.set_attribution_status(ctx.db, attribution_id, run_store.STATUS_RUNNING)
    else:
        attribution_id = run_store.create_attribution_run(
            ctx.db,
            name=name,
            eval_id=eval_id,
            metric=metric,
            sample_limit=limit,
            provider_id=provider_id if provider else None,
        )
    ctx.db.commit()
    ctx.bind("attribution", attribution_id)

    try:
        _analyze(
            ctx,
            attribution_id,
            eval_id,
            compile_id,
            metric,
            definition.higher_is_better,
            limit,
            provider,
            reader,
        )
    except BaseException:
        run_store.set_attribution_status(
            ctx.db,
            attribution_id,
            run_store.STATUS_PAUSED if ctx.pause_requested else run_store.STATUS_FAILED,
        )
        ctx.db.commit()
        raise

    run_store.set_attribution_status(
        ctx.db, attribution_id, run_store.STATUS_SUCCEEDED, finished=True
    )
    ctx.db.commit()
    counts = run_store.cause_counts(ctx.db, attribution_id)
    ctx.log(f"根因分布：{counts}")


def _analyze(
    ctx: TaskContext,
    attribution_id: int,
    eval_id: int,
    compile_id: int,
    metric: str,
    higher_is_better: bool,
    limit: int,
    provider: Any,
    reader: LineageReader | None,
) -> None:
    ranked = run_store.samples_ranked_by(
        ctx.db, eval_id, metric, ascending=higher_is_better, limit=limit
    )
    if not ranked:
        raise ValueError(f"这次评测没有 {metric} 的逐样本值")

    already = run_store.attributed_sample_ids(ctx.db, attribution_id)
    todo = [row for row in ranked if row["sample_id"] not in already]
    if already:
        ctx.log(f"续跑：已归因 {len(already)} 条，待归因 {len(todo)} 条")

    page_maps: dict[str, dict[str, str]] = {}
    for position, row in enumerate(todo, 1):
        ctx.checkpoint()
        dataset = row["dataset"]
        sample = run_store.sample_eval(ctx.db, eval_id, row["sample_id"])
        if sample is None:
            continue
        sample["metrics"] = run_store.sample_metrics_of(ctx.db, eval_id, row["sample_id"])

        if dataset not in page_maps:
            page_to_doc = run_store.page_to_doc(ctx.db, compile_id, dataset)
            page_maps[dataset] = {doc: page for page, doc in page_to_doc.items()}

        lineage = _lineage_of(
            reader,
            list(sample["detail"].get("gold_doc_ids") or []),
            page_maps[dataset],
            sample["detail"].get("question") or "",
        )
        ruling = attribution.classify(sample, lineage)

        narrative: str | None = None
        rule_based = True
        latency_ms: int | None = None
        if provider is not None:
            system, user = attribution.build_prompt(sample, ruling, lineage)
            with JudgeClient(provider) as client:
                reply = client.complete(system, user)
            latency_ms = reply.latency_ms
            if reply.failure_kind:
                ruling["evidence"]["model_error"] = reply.failure_kind
            else:
                try:
                    payload = parse_json_object(reply.content or "")
                except ValueError as exc:
                    # 模型没按 schema 输出，规则结论照样写。
                    ruling["evidence"]["model_error"] = f"parse_error: {exc}"
                else:
                    narrative = str(payload.get("narrative") or "").strip() or None
                    ruling["evidence"]["model"] = {
                        "contributing_factors": payload.get("contributing_factors"),
                        "disagreement": payload.get("disagreement"),
                        "confidence": payload.get("confidence"),
                    }
                    rule_based = False

        run_store.record_attribution(
            ctx.db,
            attribution_id,
            sample_id=row["sample_id"],
            dataset=dataset,
            root_cause=ruling["root_cause"],
            evidence=ruling["evidence"],
            narrative=narrative,
            rule_based=rule_based,
            latency_ms=latency_ms,
        )
        ctx.db.commit()
        ctx.progress(position, len(todo), "归因")
