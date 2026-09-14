"""评测层：从库里的响应算指标，需要时执行 Judge。确定性指标在本地计算，Judge 调用模型端点。

每个检索指标出两份：全样本，以及只算 ``answerMode == knowledge`` 的切片，
两份均值须结合回答模式分布与 HTTP 失败数解读。

指标能不能算由数据依赖决定：指标声明 requires、数据集声明 provides，
闸门做集合比对。算不了的记进 ``dataset_eval``，不伪造 0 分。
"""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any

from ..datasets import DataDependency, get_adapter
from ..judge import (
    answer_correctness,
    answer_relevancy,
    context_relevancy,
    faithfulness,
)
from ..judge.client import (
    JudgeClient,
    JudgeProvider,
    parse_json_object,
)
from ..judge.providers import resolve_provider
from ..metrics import attribution, multihop, qa, registry, retrieval
from ..store import compile_store, eval_store, query_store, run_store
from ..task import TaskContext

DEFAULT_KS = (2, 5, 10)
# judge 失败率超过这个值就判整轮失败：排除得太多时均值不代表整体。
MAX_JUDGE_FAILURE_RATE = 0.1


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _scalars(source: dict[str, Any]) -> dict[str, float]:
    """只取标量项，嵌套结构留在 detail 里。"""
    return {k: float(v) for k, v in source.items() if isinstance(v, (int, float, bool))}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    names = sorted({name for row in rows for name in row["metrics"]})
    return {n: _mean([float(r["metrics"].get(n, 0.0)) for r in rows]) for n in names}


def resolve_metrics(selected: list[str] | None) -> list[str]:
    """校验勾选的指标名，空表示全量。拼错的名字报错而不是忽略。"""
    if not selected:
        return sorted(registry.METRIC_REGISTRY)
    unknown = sorted(set(selected) - set(registry.METRIC_REGISTRY))
    if unknown:
        raise ValueError(f"未知指标：{unknown}")
    return sorted(set(selected))


def _keep(selected: frozenset[str]):
    """判某个实际指标名是否被勾选。实际名带 k（``recall@5``），比对前剥掉。"""

    def keep(name: str) -> bool:
        try:
            return registry.get_metric(name).name in selected
        except KeyError:
            # registry 里没声明的项保留，不让它悄悄消失。
            return True

    return keep


def evaluate_dataset(
    connection: sqlite3.Connection,
    eval_id: int,
    query_id: int,
    compile_id: int,
    dataset: str,
    ks: tuple[int, ...],
    selected: frozenset[str],
    ctx: TaskContext | None = None,
) -> dict[str, Any]:
    """算一个数据集的指标并写库，返回汇总。"""
    adapter = get_adapter(dataset)
    provides = adapter.provides
    has_gold = DataDependency.GOLD_DOCS in provides
    omitted = [d.name for d in registry.omitted(provides) if d.kind == registry.KIND_DETERMINISTIC]

    samples = {
        s["sample_id"]: s for s in compile_store.compile_samples(connection, compile_id, dataset)
    }
    responses = query_store.responses_of(connection, query_id, dataset)
    if not responses:
        raise ValueError(f"{dataset} 没有查询响应")

    page_to_doc = compile_store.page_to_doc(connection, compile_id, dataset) if has_gold else {}
    keep = _keep(selected)
    eval_store.clear_eval_results(connection, eval_id, dataset)

    per_sample: list[dict[str, Any]] = []
    http_failures = 0

    for position, row in enumerate(responses, 1):
        sample = samples.get(row["sample_id"])
        if sample is None:
            raise ValueError(f"{dataset}: 样本 {row['sample_id']!r} 有响应但不在这次编译的子集里")
        # 按 ID 匹配后再比一次问题文本，抓「ID 对得上而内容变了」。
        if row["question"] != sample["question"]:
            raise ValueError(f"{row['sample_id']}: 响应与子集的问题文本不一致，子集被重建过")

        status = row["http_status"] or 0
        body = row["response"] or {}
        ok = 200 <= status < 300 and isinstance(body, dict)
        if not ok:
            http_failures += 1

        retrieved = body.get("retrievedSources") or [] if ok else []
        citations = body.get("citations") or [] if ok else []
        evidence = body.get("citationEvidence") or [] if ok else []
        snippets = body.get("snippets") or [] if ok else []
        answer = (body.get("answer") or "") if ok else ""

        scored = qa.score_answer(answer, sample["answers"])
        metrics: dict[str, float] = dict(scored)
        detail: dict[str, Any] = {
            "metadata": dict(sample["metadata"]),
            "gold_doc_ids": list(sample["gold_doc_ids"]),
            "reference_answers": list(sample["answers"]),
            "question": sample["question"],
            "error": row["error"],
            "qa": scored,
        }

        if has_gold:
            ranked = retrieval.ranked_doc_ids(retrieved, page_to_doc)
            detail["unmapped_page_ids"] = retrieval.unmapped_page_ids(retrieved, page_to_doc)
            detail["retrieval"] = retrieval.evaluate_sample(ranked, sample["gold_doc_ids"], ks)
            detail["attribution"] = attribution.evaluate_sample(
                citations, retrieved, evidence, sample["gold_doc_ids"], page_to_doc
            )
            detail["multihop"] = multihop.evaluate_sample(
                snippets, sample["gold_doc_ids"], page_to_doc
            )
            metrics.update(detail["retrieval"])
            metrics.update(_scalars(detail["attribution"]))
            metrics.update(_scalars(detail["multihop"]))

        # 勾选过滤只作用于指标；detail 保留全部明细，供归因读。
        metrics = {name: value for name, value in metrics.items() if keep(name)}
        entry = {
            "sample_id": row["sample_id"],
            "answer_mode": row["answer_mode"],
            "metrics": metrics,
        }
        per_sample.append(entry)

        eval_store.record_sample_eval(
            connection,
            eval_id,
            sample_id=row["sample_id"],
            dataset=dataset,
            answer_mode=row["answer_mode"],
            http_status=status,
            answer=answer,
            detail=detail,
            metrics=metrics,
        )
        if position % 50 == 0 or position == len(responses):
            connection.commit()
            if ctx:
                ctx.progress(position, len(responses), dataset)
    connection.commit()

    knowledge = [e for e in per_sample if e["answer_mode"] == "knowledge"]
    overall = _aggregate(per_sample)
    eval_store.record_metric_summary(
        connection, eval_id, dataset, "overall", overall, len(per_sample)
    )
    # 保存全样本与 knowledge 子集的汇总，便于对照样本范围。
    eval_store.record_metric_summary(
        connection, eval_id, dataset, "knowledge_only", _aggregate(knowledge), len(knowledge)
    )
    eval_store.record_dataset_eval(
        connection,
        eval_id,
        dataset,
        responses_evaluated=len(per_sample),
        http_failures=http_failures,
        omitted_metrics=omitted,
        answer_modes=qa.answer_mode_distribution([e["answer_mode"] for e in per_sample]),
    )
    connection.commit()

    return {
        "dataset": dataset,
        "responses_evaluated": len(per_sample),
        "http_failures": http_failures,
        "knowledge_count": len(knowledge),
        "overall": overall,
        "omitted_metrics": omitted,
    }


def _judge_task(
    metric: str, question: str, answer: str, reference: str, body: dict[str, Any]
) -> tuple[tuple[str, str], Any] | None:
    """把一条 judge 指标摊成 ``((system, user), 解析函数)``。

    四个判据的差异收在这里，调用方只有一条路径。返回 ``None`` 表示这一条
    在这个样本上无定义，应当跳过而不是记 0。
    """
    if metric == "faithfulness":
        prompt = faithfulness.build_prompt(question, answer, body)
        return (prompt, faithfulness.parse_verdict) if prompt else None
    if metric == "answer_relevancy":
        prompt = answer_relevancy.build_prompt(question, answer, body)
        return (prompt, answer_relevancy.parse_verdict) if prompt else None
    if metric == "context_relevancy":
        built = context_relevancy.build_prompt(question, answer, body)
        if not built:
            return None
        system, user, count = built
        return (system, user), lambda payload: context_relevancy.parse_verdict(payload, count)
    if metric == "answer_correctness":
        prompt = answer_correctness.build_prompt(question, answer, reference)
        return (prompt, answer_correctness.parse_verdict) if prompt else None
    raise ValueError(f"没有实现 judge 指标 {metric!r}")


def _judge(
    ctx: TaskContext,
    eval_id: int,
    query_id: int,
    provider: JudgeProvider,
    metrics: list[str],
) -> None:
    """逐条 judge 指标跑。失败该条排除，不记 0。"""
    for metric in metrics:
        _judge_one(ctx, eval_id, query_id, provider, metric)


def _judge_one(
    ctx: TaskContext, eval_id: int, query_id: int, provider: JudgeProvider, metric: str
) -> None:
    # 逐指标问，否则判过 faithfulness 的样本会让 answer_relevancy 整批跳过。
    eval_store.restore_judge_metrics(ctx.db, eval_id, metric)
    ctx.db.commit()
    already = eval_store.judged_sample_ids(ctx.db, eval_id, metric)
    rows = [r for r in eval_store.sample_evals(ctx.db, eval_id) if r["sample_id"] not in already]
    if already:
        ctx.log(f"{metric} 续跑：已判 {len(already)} 条，待判 {len(rows)} 条")
    with JudgeClient(provider) as client:
        for position, row in enumerate(rows, 1):
            ctx.checkpoint()
            response = query_store.response_of(ctx.db, query_id, row["sample_id"])
            body = (response or {}).get("response") or {}
            references = (row["detail"] or {}).get("reference_answers") or []
            task = _judge_task(
                metric,
                (response or {}).get("question") or "",
                row["answer"] or "",
                references[0] if references else "",
                body,
            )
            if task is None:
                # 这一条在这个样本上无定义，记 None 并跳过。
                eval_store.record_judge_verdict(
                    ctx.db,
                    eval_id,
                    sample_id=row["sample_id"],
                    metric=metric,
                    score=None,
                    failure_kind=None,
                    detail={"skipped": "not applicable to this sample"},
                )
            else:
                prompt, parse = task
                reply = client.complete(*prompt)
                if reply.failure_kind:
                    eval_store.record_judge_verdict(
                        ctx.db,
                        eval_id,
                        sample_id=row["sample_id"],
                        metric=metric,
                        score=None,
                        failure_kind=reply.failure_kind,
                        latency_ms=reply.latency_ms,
                        detail={"raw": (reply.raw or "")[:500]},
                    )
                else:
                    try:
                        score, reasoning = parse(parse_json_object(reply.content or ""))
                    except ValueError as exc:
                        # 模型没按 schema 输出，记 parse_error 而不是猜一个分数。
                        eval_store.record_judge_verdict(
                            ctx.db,
                            eval_id,
                            sample_id=row["sample_id"],
                            metric=metric,
                            score=None,
                            failure_kind="parse_error",
                            detail={"error": str(exc)[:300]},
                        )
                    else:
                        eval_store.record_judge_verdict(
                            ctx.db,
                            eval_id,
                            sample_id=row["sample_id"],
                            metric=metric,
                            score=score,
                            failure_kind=None,
                            detail=reasoning,
                        )
                        if score is not None:
                            eval_store.record_sample_eval(
                                ctx.db,
                                eval_id,
                                sample_id=row["sample_id"],
                                dataset=row["dataset"],
                                answer_mode=row["answer_mode"],
                                http_status=row["http_status"],
                                answer=row["answer"] or "",
                                detail=row["detail"],
                                metrics={metric: score},
                            )
            ctx.db.commit()
            if position % 5 == 0 or position == len(rows):
                ctx.progress(position, len(rows), metric)

    summary = eval_store.judge_summary(ctx.db, eval_id, metric)
    for dataset, mean, count in eval_store.judge_means_by_dataset(ctx.db, eval_id, metric):
        eval_store.record_metric_summary(ctx.db, eval_id, dataset, "judge", {metric: mean}, count)
    ctx.db.commit()
    ctx.log(
        f"{metric} 均值 {summary['mean']}，已评分 {summary['scored']}，"
        f"失败率 {summary['failure_rate']:.3f}"
    )
    if summary["failure_rate"] > MAX_JUDGE_FAILURE_RATE:
        raise RuntimeError(
            f"{metric} 的 Judge 失败率 {summary['failure_rate']:.3f} 超过 "
            f"{MAX_JUDGE_FAILURE_RATE}。均值只覆盖成功的那部分，已不代表整体。"
            f"失败类型：{summary['failures_by_kind']}"
        )


def run(ctx: TaskContext) -> None:
    params = ctx.params
    query_id = int(params.get("query_id") or 0)
    if not query_id:
        raise ValueError("请选择一次查询")

    query_run = query_store.get_query_run(ctx.db, query_id)
    if query_run is None:
        raise ValueError(f"查询 #{query_id} 不存在")
    if query_run["status"] != run_store.STATUS_SUCCEEDED:
        raise ValueError("请选择已完成的查询记录")

    available = query_store.response_datasets(ctx.db, query_id)
    datasets = list(params.get("datasets") or available)
    unknown = sorted(set(datasets) - set(available))
    if unknown:
        raise ValueError(f"这些数据集没有查询响应：{unknown}")

    ks = tuple(int(k) for k in (params.get("ks") or DEFAULT_KS))
    if any(k < 1 for k in ks):
        raise ValueError("k 必须大于 0")
    metrics = resolve_metrics(list(params.get("metrics") or []))
    judge_selected = any(registry.get_metric(name).kind == registry.KIND_JUDGE for name in metrics)

    provider: JudgeProvider | None = None
    provider_id = params.get("judge_provider_id")
    provider_id = int(provider_id) if provider_id else None
    if judge_selected:
        provider = resolve_provider(ctx.db, provider_id, "judge")
        provider_id = provider.provider_id

    name = str(params.get("name") or "").strip() or f"{query_run['name']}-e-{uuid.uuid4().hex[:6]}"
    eval_id = ctx.target("eval")
    ctx.freeze(
        name=name, datasets=datasets, ks=list(ks), metrics=metrics, judge_provider_id=provider_id
    )
    if eval_id is None:
        if eval_store.eval_run_by_name(ctx.db, name):
            raise ValueError(f"评测名称 {name!r} 已存在，请换个名称或继续原任务")
        eval_id = eval_store.create_eval_run(
            ctx.db,
            name=name,
            query_id=query_id,
            ks=list(ks),
            metrics=metrics,
            judge_provider_id=provider_id if judge_selected else None,
        )
        ctx.bind("eval", eval_id)

    for index, dataset in enumerate(datasets):
        ctx.checkpoint()
        ctx.progress(index, len(datasets), f"{dataset} 计算指标")
        summary = evaluate_dataset(
            ctx.db,
            eval_id,
            query_id,
            int(query_run["compile_id"]),
            dataset,
            ks,
            frozenset(metrics),
            ctx,
        )
        ctx.log(
            f"{dataset}: 样本 {summary['responses_evaluated']}，"
            f"HTTP 失败 {summary['http_failures']}，"
            f"EM {summary['overall'].get('em', 0.0):.3f} "
            f"F1 {summary['overall'].get('f1', 0.0):.3f}"
        )
        if summary["omitted_metrics"]:
            ctx.log(
                f"{dataset}: 缺 gold 标注，省略 {len(summary['omitted_metrics'])} 个指标"
                "（不伪造 0 分）"
            )

    if provider is not None:
        judge_metrics = [
            name for name in metrics if registry.get_metric(name).kind == registry.KIND_JUDGE
        ]
        ctx.log(f"执行 Judge：{provider.model}，指标 {judge_metrics}")
        _judge(ctx, eval_id, query_id, provider, judge_metrics)

    ctx.progress(len(datasets), len(datasets), "评测完成")
