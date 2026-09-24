"""归因层：逐样本规则分类，并可选生成一次整轮评测分析报告。

规则判据在 :mod:`..attribution`。链路证据（原文 vs 编译产物的 diff）走只读
PostgreSQL，没配 database_url 时跳过那一段判据。
"""

from __future__ import annotations

import uuid
from typing import Any

from .. import attribution, textdiff
from ..config import load_config
from ..datasets import get_adapter, load_corpus, resolve
from ..io_utils import load_json
from ..judge.client import REPORT_MAX_TOKENS, JudgeConfigError, complete_many, parse_json_object
from ..judge.providers import resolve_provider
from ..lineage import BadPageId, LineageReader, LineageUnavailable
from ..store import attribution_store, compile_store, data_store, eval_store, query_store
from ..task import TaskContext


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
    ctx.freeze(name=name, use_model=use_model, provider_id=provider_id)
    if attribution_id is None:
        if attribution_store.attribution_run_by_name(ctx.db, name):
            raise ValueError(f"归因名称 {name!r} 已存在，请换个名称或继续原任务")
        attribution_id = attribution_store.create_attribution_run(
            ctx.db,
            name=name,
            eval_id=eval_id,
            report_provider_id=provider_id if provider else None,
        )
        ctx.bind("attribution", attribution_id)

    _analyze(
        ctx,
        attribution_id,
        eval_id,
        int(eval_run["query_id"]),
        compile_id,
        reader,
    )

    counts = attribution_store.cause_counts(ctx.db, attribution_id)
    ctx.log(f"根因分布：{counts}")
    chain_counts: dict[str, int] = {}
    false_negative_candidates = 0
    for result in attribution_store.attribution_results(ctx.db, attribution_id):
        chain = (result.get("evidence") or {}).get("evidence_chain") or {}
        status = str(chain.get("status") or "unavailable")
        chain_counts[status] = chain_counts.get(status, 0) + 1
        false_negative_candidates += int(bool(chain.get("model_false_negative_candidate")))
    ctx.log(
        f"evidence chain counts={chain_counts}; "
        f"false-negative candidates={false_negative_candidates}"
    )
    if provider is not None:
        _write_report(ctx, attribution_id, eval_run, provider)


def _analyze(
    ctx: TaskContext,
    attribution_id: int,
    eval_id: int,
    query_id: int,
    compile_id: int,
    reader: LineageReader | None,
) -> None:
    samples = eval_store.sample_evals(ctx.db, eval_id)
    if not samples:
        raise ValueError("这次评测没有样本")

    already = attribution_store.attributed_sample_ids(ctx.db, attribution_id)
    todo = [row for row in samples if row["sample_id"] not in already]
    if already:
        ctx.log(f"续跑：已归因 {len(already)} 条，待归因 {len(todo)} 条")

    page_maps: dict[str, dict[str, str]] = {}
    corpus_maps: dict[str, dict[str, dict[str, Any]]] = {}
    source_metadata: dict[str, dict[str, Any]] = {}
    if any(row["dataset"] == "musique" for row in samples):
        # 从原始数据补齐旧记录缺少的分解字段。
        try:
            resolved = resolve("musique")
            corpus = load_corpus("musique", resolved.corpus_path)
            adapter = get_adapter("musique")
            for index, raw in enumerate(load_json(resolved.qa_path)):
                parsed = adapter.parse_row(raw, index, corpus)
                source_metadata[parsed.dataset_sample_id] = parsed.metadata
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            source_metadata = {}
    sample_metadata = {
        row["sample_id"]: row.get("metadata") or {}
        for row in compile_store.compile_samples(ctx.db, compile_id)
    }

    def prepare(row: dict[str, Any]) -> dict[str, Any]:
        dataset = row["dataset"]
        sample = row
        sample["metrics"] = eval_store.sample_metrics_of(ctx.db, eval_id, row["sample_id"])
        metadata = dict(
            sample_metadata.get(row["sample_id"], sample["detail"].get("metadata") or {})
        )
        if dataset == "musique" and not metadata.get("question_decomposition"):
            native_id = str(row["sample_id"]).split(":", 1)[-1]
            metadata.update(source_metadata.get(native_id) or {})
        if dataset == "musique" and metadata.get("question_decomposition"):
            if dataset not in corpus_maps:
                corpus_maps[dataset] = {
                    str(doc["doc_id"]): doc for doc in data_store.corpus_of(ctx.db, dataset)
                }
            steps = []
            for step in metadata["question_decomposition"]:
                enriched = dict(step)
                doc = corpus_maps[dataset].get(str(step.get("support_doc_id")))
                if doc:
                    enriched["support_text"] = doc.get("text")
                    enriched["support_title"] = doc.get("title")
                steps.append(enriched)
            metadata["question_decomposition"] = steps
        sample["detail"] = {**sample["detail"], "metadata": metadata}
        if dataset not in page_maps:
            page_to_doc = compile_store.page_to_doc(ctx.db, compile_id, dataset)
            page_maps[dataset] = {doc: page for page, doc in page_to_doc.items()}
        lineage = _lineage_of(
            reader,
            list(sample["detail"].get("gold_doc_ids") or []),
            page_maps[dataset],
            sample["detail"].get("question") or "",
        )
        response_row = query_store.response_of(ctx.db, query_id, row["sample_id"])
        evidence_chain = attribution.analyze_evidence_chain(
            sample, (response_row or {}).get("response")
        )
        ruling = attribution.classify(sample, lineage, evidence_chain)
        return {"row": row, "ruling": ruling}

    done = 0
    for start in range(0, len(todo), 50):
        ctx.checkpoint()
        batch = [prepare(row) for row in todo[start : start + 50]]
        for item in batch:
            ruling = item["ruling"]
            attribution_store.record_attribution(
                ctx.db,
                attribution_id,
                sample_id=item["row"]["sample_id"],
                root_cause=ruling["root_cause"],
                evidence=ruling["evidence"],
            )
        ctx.db.commit()
        done += len(batch)
        ctx.progress(done, len(todo), "归因")


def _write_report(
    ctx: TaskContext,
    attribution_id: int,
    eval_run: dict[str, Any],
    provider: Any,
) -> None:
    """用一次模型调用分析整轮指标并保存报告；失败不影响规则归因。"""
    current = attribution_store.get_attribution_run(ctx.db, attribution_id) or {}
    if current.get("report"):
        ctx.log("续跑：整体评测分析报告已存在，跳过模型调用")
        return
    ctx.checkpoint()
    eval_id = int(eval_run["id"])
    general_rows = eval_store.sample_evals(ctx.db, eval_id, answer_mode="general")[:10]
    metrics_by_sample = eval_store.sample_metrics_for(
        ctx.db, eval_id, [str(row["sample_id"]) for row in general_rows]
    )
    general_samples = [
        {
            "dataset": row["dataset"],
            "question": str((row.get("detail") or {}).get("question") or "")[:1000],
            "reference_answers": [
                str(answer)[:500]
                for answer in (row.get("detail") or {}).get("reference_answers") or []
            ][:5],
            "answer": str(row.get("answer") or "")[:1500],
            "metrics": metrics_by_sample.get(str(row["sample_id"]), {}),
        }
        for row in general_rows
    ]
    prompt = attribution.build_report_prompt(
        eval_run,
        eval_store.metric_summaries(ctx.db, eval_id),
        eval_store.dataset_evals(ctx.db, eval_id),
        general_samples,
    )
    reply = complete_many(provider, [prompt], 1, max_tokens=REPORT_MAX_TOKENS)[0]
    report: str | None = None
    error = reply.failure_kind
    if error is None:
        try:
            payload = parse_json_object(reply.content or "")
            report = str(payload.get("report") or "").strip() or None
            if report is None:
                error = "parse_error: missing report"
        except ValueError as exc:
            error = f"parse_error: {exc}"
    attribution_store.record_report(
        ctx.db,
        attribution_id,
        report=report,
        error=error,
        latency_ms=reply.latency_ms,
    )
    ctx.db.commit()
    if error:
        ctx.log(f"整体评测分析报告生成失败：{error}", "warn")
    else:
        ctx.log("整体评测分析报告已生成")
