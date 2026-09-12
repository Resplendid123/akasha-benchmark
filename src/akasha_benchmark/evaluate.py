"""评测：建一个**评测层**，从库里的响应算指标。纯离线，不再碰 Akasha。

每个检索指标都出两份 —— 全样本，以及只算 ``answerMode == "knowledge"`` 的切片。
原因是 ``no_match`` 和 ``general`` 两种模式无条件返回
``retrievedSources: []``（``ai-knowledge-chat.service.ts:641,667``），
不管检索实际找到了什么。所以全样本那份数字把「生成端拒答」也算进了检索指标里，
两份的差值就是这个效应的规模。

指标能不能算，由**数据依赖**决定而不是数据集名字：指标声明 ``requires``、
数据集声明 ``provides``、闸门做集合比对。算不了的指标连同原因一起
记进 ``dataset_eval``，**不伪造 0 分**。

    uv run python -m akasha_benchmark.evaluate --query-label run002-query
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

from . import run_args
from .datasets import DATASET_NAMES, DataDependency, get_adapter
from .datasets.resolver import DEFAULT_DATA_DIR
from .io_utils import atomic_write_json, atomic_write_jsonl, atomic_write_text, utc_now
from .metrics import attribution, multihop, qa, registry, retrieval
from .store import connect, identity, repo

# 各数据集用哪个 metadata 字段做分层报告。
# musique 有真正的跳数，其余三组只能按题型/文档类型切。
STRATIFY_KEYS = {
    "hotpotqa": "type",
    "2wikimultihopqa": "type",
    "musique": "hop_count",
    "narrativeqa": "kind",
}


def reports_dir(label: str, data_dir: Path | None = None) -> Path:
    return (data_dir or DEFAULT_DATA_DIR) / "reports" / label


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _numeric(source: dict[str, Any]) -> dict[str, float]:
    """只取标量项。``reason_counts`` 之类的嵌套结构留在 detail 里，不进扁平指标表。"""
    return {k: float(v) for k, v in source.items() if isinstance(v, (int, float, bool))}


def _omission_reason(dataset: str, has_gold: bool, deselected: list[str]) -> str | None:
    """为什么某些指标这一轮没有值。**两种原因必须分开说。**

    缺依赖是「这个数据集永远算不了」，没勾选是「这一轮没要」。混成一句话的话,
    读者会把后者当成前者，进而以为 narrativeqa 之外的组也缺 gold 标注。
    """
    reasons = []
    if not has_gold:
        reasons.append(
            f"{dataset} does not provide GOLD_DOCS: it has no gold document annotations, "
            "so retrieval, attribution and multihop metrics are undefined. They are "
            "omitted rather than reported as 0, which would silently pollute aggregates."
        )
    if deselected:
        reasons.append(
            f"not selected for this run: {', '.join(deselected)}. These are computable "
            "for this dataset; they were left out of the metric selection."
        )
    return " ".join(reasons) or None


def _metric_filter(selected: frozenset[str] | None, ks: tuple[int, ...]):
    """返回一个「这个实际指标名要不要留」的判定函数。

    实际名带 k（``recall@5``），勾选名不带（``recall``），所以比对前要剥掉 k。
    未知名字一律保留 —— 那说明 registry 里没声明它，让它悄悄消失比留着更糟。
    """
    if selected is None:
        return lambda name: True

    def keep(name: str) -> bool:
        try:
            definition = registry.get_metric(name)
        except KeyError:
            return True
        return definition.name in selected

    return keep


def evaluate_dataset(
    connection: sqlite3.Connection,
    eval_layer_id: int,
    query_layer_id: int,
    index_layer_id: int,
    dataset: str,
    ks: tuple[int, ...],
    selected: frozenset[str] | None = None,
) -> dict[str, Any]:
    """算一个数据集的指标，写进库，返回汇总。

    ``selected`` 是这一轮勾选的指标模板名（``recall`` 而不是 ``recall@5``）。
    ``None`` 表示全量。**过滤发生在写库之前而不是计算之前**：逐样本的检索族
    指标是一次算出来的一组，拆开单算不会更快，而按需过滤能让
    ``sample_metric`` 只存勾了的那些 —— 报告页的列因此与勾选一致。
    """
    adapter = get_adapter(dataset)
    provides = adapter.provides
    # 依赖判定收在 registry：narrativeqa 不提供 GOLD_DOCS，所以整族检索指标
    # 省略而不是算成 0。判据是集合比对，不是数据集名字。
    has_gold = DataDependency.GOLD_DOCS in provides
    omitted = [d.name for d in registry.omitted(provides) if d.kind == registry.KIND_DETERMINISTIC]
    # 勾选之外的确定性指标也算「本轮没出」，与缺依赖的那些一起报告 ——
    # 但两者原因不同，所以在 dataset_eval 里分开记。
    deselected = (
        []
        if selected is None
        else sorted(
            d.name
            for d in registry.available(provides)
            if d.kind == registry.KIND_DETERMINISTIC and d.name not in selected
        )
    )
    keep = _metric_filter(selected, ks)

    samples = {
        s["sample_id"]: s for s in repo.subset_samples(connection, index_layer_id, dataset)
    }
    if not samples:
        raise FileNotFoundError(f"{dataset}: no subset samples in index layer #{index_layer_id}")
    responses = repo.responses_of(connection, query_layer_id, dataset)
    if not responses:
        raise FileNotFoundError(f"{dataset}: no responses in query layer #{query_layer_id}")

    page_to_doc = repo.page_to_doc(connection, index_layer_id, dataset) if has_gold else {}

    per_sample: list[dict[str, Any]] = []
    http_failures = 0
    unmapped_pages: set[str] = set()

    for row in responses:
        sample_id = row["sample_id"]
        sample = samples.get(sample_id)
        if sample is None:
            raise ValueError(
                f"{dataset}: sample_id {sample_id!r} has a response but is not in the subset "
                f"of index layer #{index_layer_id}; this query layer points at another subset"
            )
        # 按 ID 匹配后**再比一次 question 文本**。外键保证了归属，但抓不到
        # 「层还在、子集被原地重建过」—— 那种情况下 ID 对得上而内容已经变了。
        if row["question"] != sample["question"]:
            raise ValueError(
                f"{sample_id}: question text differs between the response and the subset. "
                "The subset was rebuilt after these responses were recorded."
            )

        status = row["http_status"] or 0
        body = row["response"] or {}
        # 失败行照样参与统计（F1 记 0），因为失败率本身是结果的一部分。
        ok = 200 <= status < 300 and isinstance(body, dict)
        if not ok:
            http_failures += 1

        answer_mode = body.get("answerMode") if ok else None
        retrieved = body.get("retrievedSources") or [] if ok else []
        citations = body.get("citations") or [] if ok else []
        citation_evidence = body.get("citationEvidence") or [] if ok else []
        snippets = body.get("snippets") or [] if ok else []
        answer = body.get("answer") or "" if ok else ""

        scored = qa.score_answer(answer, sample["answers"])
        metrics: dict[str, float] = dict(scored)
        detail: dict[str, Any] = {
            "metadata": dict(sample["metadata"]),
            "warnings": body.get("warnings") if ok else None,
            "completeness_notice": body.get("completenessNotice") if ok else None,
            "budget": body.get("budget") if ok else None,
            "error": row["error"],
            "qa": scored,
        }

        if has_gold:
            unmapped_pages.update(retrieval.unmapped_page_ids(retrieved, page_to_doc))
            ranked = retrieval.ranked_doc_ids(retrieved, page_to_doc)
            detail["retrieval"] = retrieval.evaluate_sample(ranked, sample["gold_doc_ids"], ks)
            detail["attribution"] = attribution.evaluate_sample(
                citations, retrieved, citation_evidence, sample["gold_doc_ids"], page_to_doc
            )
            detail["multihop"] = multihop.evaluate_sample(
                snippets, sample["gold_doc_ids"], page_to_doc
            )
            metrics.update(detail["retrieval"])
            metrics.update(_numeric(detail["attribution"]))
            metrics.update(_numeric(detail["multihop"]))

        # 勾选过滤。detail 保留全部明细 —— 那是归因要读的原始链路，
        # 与「这一轮报哪些指标」是两件事。
        metrics = {name: value for name, value in metrics.items() if keep(name)}

        entry = {
            "sample_id": sample_id,
            "dataset": dataset,
            "answer_mode": answer_mode,
            "ok": ok,
            "http_status": status,
            "gold_count": len(sample["gold_doc_ids"]),
            "retrieved_count": len(retrieved),
            "citation_count": len(citations),
            "snippet_count": len(snippets) if ok else None,
            "latency_ms": row["latency_ms"],
            "answer": answer,
            "metrics": metrics,
            "detail": detail,
        }
        per_sample.append(entry)

        repo.record_sample_eval(
            connection,
            eval_layer_id,
            sample_id=sample_id,
            dataset=dataset,
            answer_mode=answer_mode,
            ok=ok,
            http_status=status,
            gold_count=entry["gold_count"],
            retrieved_count=entry["retrieved_count"],
            citation_count=entry["citation_count"],
            snippet_count=entry["snippet_count"],
            latency_ms=entry["latency_ms"],
            answer=answer,
            detail=detail,
        )
        repo.record_sample_metrics(connection, eval_layer_id, sample_id, dataset, metrics)
    connection.commit()

    return _summarize(
        connection,
        eval_layer_id,
        dataset,
        per_sample,
        subset_size=len(samples),
        missing=sorted(set(samples) - {e["sample_id"] for e in per_sample}),
        has_gold=has_gold,
        omitted=omitted,
        deselected=deselected,
        http_failures=http_failures,
        unmapped_pages=sorted(unmapped_pages),
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """一组样本的指标按算术平均汇总。指标名取并集，缺的按 0 计。"""
    if not rows:
        return {}
    names = sorted({name for row in rows for name in row["metrics"]})
    return {
        name: _mean([float(row["metrics"].get(name, 0.0)) for row in rows]) for name in names
    }


def _summarize(
    connection: sqlite3.Connection,
    eval_layer_id: int,
    dataset: str,
    per_sample: list[dict[str, Any]],
    *,
    subset_size: int,
    missing: list[str],
    has_gold: bool,
    omitted: list[str],
    deselected: list[str],
    http_failures: int,
    unmapped_pages: list[str],
) -> dict[str, Any]:
    """写汇总与分层，返回给报告用的结构。"""
    knowledge_rows = [e for e in per_sample if e["answer_mode"] == "knowledge"]
    overall = _aggregate(per_sample)
    knowledge_only = _aggregate(knowledge_rows)

    repo.record_metric_summary(
        connection, eval_layer_id, dataset, "overall", overall, len(per_sample)
    )
    # 两份口径必须都存：差值就是生成端拒答的规模，而不是检索失败。
    repo.record_metric_summary(
        connection, eval_layer_id, dataset, "knowledge_only", knowledge_only, len(knowledge_rows)
    )

    stratified: dict[str, Any] | None = None
    key = STRATIFY_KEYS[dataset]
    buckets = multihop.stratify(
        [{**e, "metadata": e["detail"]["metadata"]} for e in per_sample], key
    )
    strata: dict[str, Any] = {}
    for name, rows in buckets.items():
        bucket_metrics = _aggregate(rows)
        knowledge = [r for r in rows if r["answer_mode"] == "knowledge"]
        repo.record_metric_summary(
            connection, eval_layer_id, dataset, f"stratum:{name}", bucket_metrics, len(rows)
        )
        strata[name] = {
            "count": len(rows),
            "metrics": bucket_metrics,
            "knowledge_answer_share": len(knowledge) / len(rows) if rows else 0.0,
        }
    stratified = {"key": key, "strata": strata}

    reason_totals = (
        multihop.aggregate([e["detail"]["multihop"] for e in per_sample]) if has_gold else {}
    )

    repo.record_dataset_eval(
        connection,
        eval_layer_id,
        dataset,
        samples_in_subset=subset_size,
        responses_evaluated=len(per_sample),
        http_failures=http_failures,
        missing_responses=missing,
        unmapped_page_ids=unmapped_pages,
        omitted_metrics=omitted,
        omission_reason=_omission_reason(dataset, has_gold, deselected),
        answer_mode_distribution=qa.answer_mode_distribution(
            [e["answer_mode"] for e in per_sample]
        ),
        stratified=stratified,
    )
    connection.commit()

    return {
        "dataset": dataset,
        "samples_in_subset": subset_size,
        "responses_evaluated": len(per_sample),
        "missing_responses": missing,
        "http_failures": http_failures,
        "has_gold": has_gold,
        "omitted_metrics": omitted,
        "deselected_metrics": deselected,
        "answer_mode_distribution": qa.answer_mode_distribution(
            [e["answer_mode"] for e in per_sample]
        ),
        "overall": overall,
        "knowledge_only": knowledge_only,
        "knowledge_answer_count": len(knowledge_rows),
        "stratified": stratified,
        "unmapped_page_ids": unmapped_pages,
        "reason_gold_rate": reason_totals.get("reason_gold_rate", {}),
        "latency_ms_mean": _mean([float(e["latency_ms"] or 0) for e in per_sample]),
    }


def _format_report(label: str, summaries: list[dict[str, Any]], context: dict[str, Any]) -> str:
    """生成人读的 report.md。

    开头那段架构说明是必须的，不是客套：不写清楚「召回跑在生成文本上」，
    读者会拿这些数字直接跟公开 baseline 比，而那个比较是无效的。
    """
    lines = [
        f"# Akasha-Benchmark 评测报告 — {label}",
        "",
        f"生成于 {context['generated_at']}。",
        "",
        "## 这些数字该怎么读",
        "",
        "Akasha 的向量召回与词法召回跑在**编译产物**上，而不是原始文档：",
        "`knowledge_chunks` 索引的是 `artifact.markdown`。原文存在",
        "`knowledge_source_chunks` 里，它不参与召回，只在解析引用时提供证据窗口。",
        "编译可能遗漏或改写原文信息，也可能改变跨文档关系。与公开 baseline 比较前，",
        "需对齐语料、样本、检索单位和答案格式；原文基线可帮助隔离编译影响。",
        "",
        "返回空 retrievedSources 的回落响应会得到零检索分。报告同时给出全样本和",
        "knowledge 切片，帮助区分生成侧回落与检索结果。",
        "",
        "**Exact Match 要求归一化后的答案整串相等。**",
        "解释性长答案通常得分较低，但并非必然为零。F1 也受答案长度影响，",
        "应结合引用证据与人工抽查解读。",
        "",
        "## 运行配置",
        "",
        f"- 索引层：#{context['index_layer_id']} `{context['index_label']}`"
        f"（config_hash `{context['index_hash']}`）",
        f"- 查询层：#{context['query_layer_id']} `{context['query_label']}`"
        f"（config_hash `{context['query_hash']}`）",
        f"- 评测层：#{context['eval_layer_id']} `{context['eval_label']}`",
        f"- k = {context['ks']}",
        "",
        "```json",
        context["model_configs_excerpt"],
        "```",
        "",
    ]

    for summary in summaries:
        overall, knowledge = summary["overall"], summary["knowledge_only"]
        lines += [
            f"## {summary['dataset']}",
            "",
            f"- 样本数：{summary['responses_evaluated']}/{summary['samples_in_subset']}"
            f"（HTTP 失败：{summary['http_failures']}）",
            f"- 回答模式分布：{summary['answer_mode_distribution']}",
            f"- answer EM：{overall.get('em', 0.0):.4f}"
            f"（仅 knowledge：{knowledge.get('em', 0.0):.4f}）— 受答案格式影响，见上",
            f"- answer F1：{overall.get('f1', 0.0):.4f}"
            f"（仅 knowledge：{knowledge.get('f1', 0.0):.4f}）",
        ]

        if summary["has_gold"]:
            retrieval_names = sorted(
                name
                for name in overall
                if registry.get_metric(name).family
                in (registry.FAMILY_RETRIEVAL, registry.FAMILY_ATTRIBUTION, registry.FAMILY_MULTIHOP)
            )
            lines += ["", "| 指标 | 全样本 | 仅 knowledge |", "| --- | --- | --- |"]
            for name in retrieval_names:
                lines.append(
                    f"| {name} | {overall[name]:.4f} | {knowledge.get(name, 0.0):.4f} |"
                )
            if summary["reason_gold_rate"]:
                lines += [
                    "",
                    "- 各检索信号的 gold 命中率："
                    + "、".join(f"{r}={v:.3f}" for r, v in summary["reason_gold_rate"].items()),
                ]
            lines += [
                "",
                f"### 按 `{summary['stratified']['key']}` 分层",
                "",
                "| 分层 | n | recall@10 | full_coverage@10 | F1 | knowledge 占比 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for name, bucket in summary["stratified"]["strata"].items():
                metrics = bucket["metrics"]
                lines.append(
                    f"| {name} | {bucket['count']} |"
                    f" {metrics.get('recall@10', 0.0):.4f} |"
                    f" {metrics.get('full_coverage@10', 0.0):.4f} |"
                    f" {metrics.get('f1', 0.0):.4f} |"
                    f" {bucket['knowledge_answer_share']:.4f} |"
                )
            if summary["unmapped_page_ids"]:
                lines += [
                    "",
                    f"> 有 {len(summary['unmapped_page_ids'])} 个被检索到的 page id 不在"
                    " `page_map` 里。它们仍占据排名位次，但永远不可能被算作 gold。",
                ]
        else:
            lines += [
                "",
                f"> 省略的指标：{', '.join(summary['omitted_metrics'])}。",
                f"> {summary['dataset']} 没有 gold 文档标注，这些指标无定义，"
                "因此省略而不是报 0 —— 报 0 会静默污染任何包含它的汇总。",
            ]
        lines.append("")

    return "\n".join(lines) + "\n"


def ensure_eval_layer(
    connection: sqlite3.Connection,
    *,
    query_layer_id: int,
    label: str,
    ks: tuple[int, ...],
    metrics: list[str],
) -> tuple[int, bool]:
    """取或建评测层。同 label 复用，并清掉旧的确定性结果（judge 判决保留）。"""
    existing = repo.eval_layer_by_label(connection, label)
    if existing:
        layer_id = int(existing["id"])
        # 重跑确定性指标要先清旧结果，否则汇总会翻倍。judge_verdict 与
        # annotation 刻意不动 —— 前者要花钱，后者无法重算。
        repo.clear_eval_results(connection, layer_id)
        connection.commit()
        return layer_id, False

    query_layer = repo.get_query_layer(connection, query_layer_id) or {}
    layer_id = repo.create_eval_layer(
        connection,
        query_layer_id=query_layer_id,
        label=label,
        config_hash=identity.eval_layer_hash(
            query_config_hash=query_layer.get("config_hash") or "",
            ks=ks,
            metrics=metrics,
        ),
        ks=ks,
        metrics=metrics,
    )
    connection.commit()
    return layer_id, True


def export_report(
    connection: sqlite3.Connection, eval_layer_id: int, data_dir: Path | None = None
) -> Path:
    """导出 metrics.json / per_sample.jsonl / report.md。**可选**，权威在库里。"""
    layer = repo.get_eval_layer(connection, eval_layer_id) or {}
    query_layer = repo.get_query_layer(connection, int(layer["query_layer_id"])) or {}
    index_layer = repo.get_index_layer(connection, int(query_layer["index_layer_id"])) or {}
    out_dir = reports_dir(str(layer.get("label") or eval_layer_id), data_dir)

    dataset_rows = repo.dataset_evals(connection, eval_layer_id)
    summaries: list[dict[str, Any]] = []
    for row in dataset_rows:
        dataset = row["dataset"]
        scopes: dict[str, dict[str, float]] = {}
        for entry in repo.metric_summaries(connection, eval_layer_id, dataset):
            scopes.setdefault(entry["scope"], {})[entry["metric"]] = entry["value"]
        stratified = repo.loads(row["stratified_json"])
        summaries.append(
            {
                "dataset": dataset,
                "samples_in_subset": row["samples_in_subset"],
                "responses_evaluated": row["responses_evaluated"],
                "http_failures": row["http_failures"],
                "missing_responses": repo.loads(row["missing_responses_json"], []),
                "unmapped_page_ids": repo.loads(row["unmapped_page_ids_json"], []),
                "omitted_metrics": repo.loads(row["omitted_metrics_json"], []),
                "omission_reason": row["omission_reason"],
                "has_gold": not repo.loads(row["omitted_metrics_json"], []),
                "answer_mode_distribution": repo.loads(row["answer_mode_distribution_json"], {}),
                "overall": scopes.get("overall", {}),
                "knowledge_only": scopes.get("knowledge_only", {}),
                "stratified": stratified,
                "reason_gold_rate": {},
            }
        )

    context = {
        "generated_at": utc_now(),
        "index_layer_id": index_layer.get("id"),
        "index_label": index_layer.get("label"),
        "index_hash": index_layer.get("config_hash"),
        "query_layer_id": query_layer.get("id"),
        "query_label": query_layer.get("label"),
        "query_hash": query_layer.get("config_hash"),
        "eval_layer_id": eval_layer_id,
        "eval_label": layer.get("label"),
        "ks": repo.loads(layer.get("ks_json"), []),
        "model_configs_excerpt": str(repo.loads(query_layer.get("model_configs_json")))[:2000],
    }
    atomic_write_json(
        out_dir / "metrics.json",
        {
            "stage": "evaluate",
            "note": "exported from the database; the database is the source of truth",
            **context,
            "datasets": summaries,
        },
    )
    atomic_write_jsonl(
        out_dir / "per_sample.jsonl",
        (
            {
                **{k: v for k, v in row.items() if k != "detail_json"},
                "detail": repo.loads(row["detail_json"], {}),
                "metrics": repo.sample_metrics_of(connection, eval_layer_id, row["sample_id"]),
            }
            for row in repo.sample_evals(connection, eval_layer_id)
        ),
    )
    atomic_write_text(
        out_dir / "report.md",
        _format_report(str(layer.get("label") or eval_layer_id), summaries, context),
    )
    return out_dir


def resolve_metrics(selected: list[str] | None) -> tuple[list[str], frozenset[str] | None]:
    """把勾选的指标名整成 ``(存库的列表, 过滤用的集合)``。

    ``None`` 或空表示全量 —— 命令行不传就是这条路。未知名字直接抛错而不是忽略：
    UI 传了一个拼错的指标名却静默跑全量，那份报告会比预期多出好几列。
    """
    if not selected:
        return sorted(registry.METRIC_REGISTRY), None
    unknown = sorted(set(selected) - set(registry.METRIC_REGISTRY))
    if unknown:
        raise ValueError(
            f"unknown metrics {unknown}. Known: {', '.join(sorted(registry.METRIC_REGISTRY))}"
        )
    names = sorted(set(selected))
    return names, frozenset(names)


def run(
    query_label: str,
    datasets: list[str],
    db_path: Path | None,
    ks: tuple[int, ...],
    eval_label: str | None = None,
    export: bool = False,
    data_dir: Path | None = None,
    metrics: list[str] | None = None,
) -> int:
    """执行评测。返回退出码。"""
    connection = connect(db_path)
    try:
        query_layer = repo.query_layer_by_label(connection, query_label)
        if query_layer is None:
            print(f"ERROR no query layer labelled {query_label!r}", file=sys.stderr)
            return 1
        query_layer_id = int(query_layer["id"])
        index_layer_id = int(query_layer["index_layer_id"])

        metric_names, selected = resolve_metrics(metrics)
        eval_layer_id, created = ensure_eval_layer(
            connection,
            query_layer_id=query_layer_id,
            label=eval_label or f"{query_label}-eval",
            ks=ks,
            metrics=metric_names,
        )
        print(
            f"eval layer #{eval_layer_id} "
            f"({'created' if created else 'reused, previous results cleared'}), "
            f"metrics={'all' if selected is None else len(metric_names)}"
        )

        summaries: list[dict[str, Any]] = []
        failures = 0
        for dataset in datasets:
            try:
                summaries.append(
                    evaluate_dataset(
                        connection,
                        eval_layer_id,
                        query_layer_id,
                        index_layer_id,
                        dataset,
                        ks,
                        selected,
                    )
                )
            except FileNotFoundError as exc:
                # 该数据集没跑过查询，跳过而不算失败。
                print(f"skip {dataset}: {exc}", file=sys.stderr)
                continue
            except Exception as exc:  # noqa: BLE001 - 报告后继续做下一个数据集
                failures += 1
                print(f"FAIL {dataset}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue

        if not summaries:
            print("no datasets evaluated", file=sys.stderr)
            return 1

        repo.finish_eval_layer(connection, eval_layer_id)
        connection.commit()

        for summary in summaries:
            line = (
                f"ok   {summary['dataset']:<16} n={summary['responses_evaluated']:<4} "
                f"EM={summary['overall'].get('em', 0.0):.3f} "
                f"F1={summary['overall'].get('f1', 0.0):.3f}"
            )
            if summary["has_gold"]:
                line += (
                    f" R@10={summary['overall'].get('recall@10', 0.0):.3f}"
                    f" (knowledge-only {summary['knowledge_only'].get('recall@10', 0.0):.3f})"
                    f" FC@10={summary['overall'].get('full_coverage@10', 0.0):.3f}"
                )
            else:
                line += f"  [omitted: {len(summary['omitted_metrics'])} gold-dependent metrics]"
            print(line)

        if export:
            print(f"\nexported -> {export_report(connection, eval_layer_id, data_dir)}")
        return 1 if failures else 0
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-label", default=None, help="查询层标签")
    parser.add_argument("--eval-label", default=None, help="评测层标签，默认 {query-label}-eval")
    parser.add_argument("--dataset", action="append", choices=list(DATASET_NAMES))
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None, help="仅 --export 用")
    parser.add_argument("--k", action="append", type=int, default=None)
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        dest="metrics",
        help="只算这些指标（模板名，如 recall）。不传则全量",
    )
    parser.add_argument("--export", action="store_true", help="额外导出报告文件（可选）")
    run_args.add_argument(parser)

    try:
        args = run_args.apply(parser.parse_args(argv), stage="evaluate")
        run_args.require(args.query_label, "query_label", "evaluate")
        ks = tuple(args.k) if args.k else retrieval.DEFAULT_KS
        return run(
            args.query_label,
            args.dataset or list(DATASET_NAMES),
            args.db,
            ks,
            args.eval_label,
            args.export,
            args.data_dir,
            args.metrics,
        )
    except (RuntimeError, ValueError, FileNotFoundError, sqlite3.Error) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
