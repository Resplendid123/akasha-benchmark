"""归因层：针对某次 编译->查询->评测 链路，逐条推出根因。

规则判据在 :mod:`..attribution`。链路证据（原文 vs 编译产物的 diff）走只读
PostgreSQL，没配 database_url 时跳过那一段判据。
"""

from __future__ import annotations

import uuid
from typing import Any

from .. import attribution, textdiff
from ..config import load_config
from ..judge.client import JudgeConfigError, complete_many, parse_json_object
from ..judge.providers import resolve_provider
from ..lineage import BadPageId, LineageReader, LineageUnavailable
from ..metrics import registry
from ..store import attribution_store, compile_store, eval_store, loads, query_store
from ..task import TaskContext

DEFAULT_LIMIT = 10
MAX_LIMIT = 1000


def _default_metric(eval_run: dict[str, Any]) -> str:
    """选本次评测实际产出的第一个指标，避免依赖固定的 recall@5。"""
    metrics = loads(eval_run.get("metrics_json"), [])
    ks = loads(eval_run.get("ks_json"), [])
    for name in metrics:
        definition = registry.get_metric(name)
        if definition.per_k:
            if ks:
                return f"{name}@{ks[0]}"
        else:
            return name
    raise ValueError("这次评测没有可用于归因的指标")


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

    eval_run = eval_store.get_eval_run(ctx.db, eval_id)
    if eval_run is None:
        raise ValueError(f"评测 #{eval_id} 不存在")
    query_run = query_store.get_query_run(ctx.db, int(eval_run["query_id"]))
    if query_run is None:
        raise ValueError("这次评测对应的查询记录已不存在")
    compile_id = int(query_run["compile_id"])

    metric = str(params.get("metric") or _default_metric(eval_run))
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
            provider_id = provider.provider_id
        except JudgeConfigError as exc:
            # 模型没配好不算失败，规则结论仍是一条有效归因。
            ctx.log(f"未使用模型：{exc}", "warn")

    reader: LineageReader | None = None
    try:
        reader = LineageReader(load_config(ctx.db).database_url)
    except LineageUnavailable as exc:
        ctx.log(f"链路证据不可用（compiled_away 判不了）：{exc}", "warn")

    name = str(params.get("name") or "").strip() or f"{eval_run['name']}-a-{uuid.uuid4().hex[:6]}"
    attribution_id = ctx.target("attribution")
    ctx.freeze(
        name=name, metric=metric, sample_limit=limit, use_model=use_model, provider_id=provider_id
    )
    if attribution_id is None:
        if attribution_store.attribution_run_by_name(ctx.db, name):
            raise ValueError(f"归因名称 {name!r} 已存在，请换个名称或继续原任务")
        attribution_id = attribution_store.create_attribution_run(
            ctx.db,
            name=name,
            eval_id=eval_id,
            metric=metric,
            sample_limit=limit,
            provider_id=provider_id if provider else None,
        )
        ctx.bind("attribution", attribution_id)

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

    counts = attribution_store.cause_counts(ctx.db, attribution_id)
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
    ranked = eval_store.samples_ranked_by(
        ctx.db, eval_id, metric, ascending=higher_is_better, limit=limit
    )
    if not ranked:
        raise ValueError(f"这次评测没有 {metric} 的逐样本值")

    already = attribution_store.attributed_sample_ids(ctx.db, attribution_id)
    todo = [row for row in ranked if row["sample_id"] not in already]
    if already:
        ctx.log(f"续跑：已归因 {len(already)} 条，待归因 {len(todo)} 条")

    page_maps: dict[str, dict[str, str]] = {}

    def prepare(row: dict[str, Any]) -> dict[str, Any] | None:
        """主线程：读样本、算规则结论、拼 prompt。返回一条待落库的记录。"""
        dataset = row["dataset"]
        sample = eval_store.sample_eval(ctx.db, eval_id, row["sample_id"])
        if sample is None:
            return None
        sample["metrics"] = eval_store.sample_metrics_of(ctx.db, eval_id, row["sample_id"])
        if dataset not in page_maps:
            page_to_doc = compile_store.page_to_doc(ctx.db, compile_id, dataset)
            page_maps[dataset] = {doc: page for page, doc in page_to_doc.items()}
        lineage = _lineage_of(
            reader,
            list(sample["detail"].get("gold_doc_ids") or []),
            page_maps[dataset],
            sample["detail"].get("question") or "",
        )
        ruling = attribution.classify(sample, lineage)
        prompt = attribution.build_prompt(sample, ruling, lineage) if provider is not None else None
        return {"row": row, "dataset": dataset, "ruling": ruling, "prompt": prompt}

    concurrency = max(1, provider.concurrency) if provider is not None else 1
    done = 0
    for start in range(0, len(todo), concurrency):
        ctx.checkpoint()
        batch = [p for p in (prepare(row) for row in todo[start : start + concurrency]) if p]
        replies = (
            complete_many(provider, [p["prompt"] for p in batch], concurrency)
            if provider is not None
            else [None] * len(batch)
        )
        for item, reply in zip(batch, replies):
            ruling = item["ruling"]
            narrative: str | None = None
            rule_based = True
            latency_ms: int | None = None
            if reply is not None:
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
            attribution_store.record_attribution(
                ctx.db,
                attribution_id,
                sample_id=item["row"]["sample_id"],
                dataset=item["dataset"],
                root_cause=ruling["root_cause"],
                evidence=ruling["evidence"],
                narrative=narrative,
                rule_based=rule_based,
                latency_ms=latency_ms,
            )
        ctx.db.commit()
        done += len(todo[start : start + concurrency])
        ctx.progress(done, len(todo), "归因")
