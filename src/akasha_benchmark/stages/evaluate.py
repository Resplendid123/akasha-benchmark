"""评测层：从库里的响应算指标，需要时执行 Judge。纯离线，不碰 Akasha。

每个检索指标出两份 —— 全样本，以及只算 ``answerMode == knowledge`` 的切片。
``no_match`` / ``general`` 无条件返回空 ``retrievedSources``，所以全样本那份
把「生成端拒答」也算进了检索指标里，两份的差值就是这个效应的规模。

指标能不能算由**数据依赖**决定而不是数据集名字：指标声明 requires、
数据集声明 provides，闸门做集合比对。算不了的记进 ``dataset_eval``，
**不伪造 0 分** —— 假分数会静默污染任何包含它的汇总。
"""

from __future__ import annotations

import sqlite3
from typing import Any

from ..datasets import DataDependency, get_adapter
from ..judge import faithfulness
from ..judge.client import (
    JudgeClient,
    JudgeConfigError,
    JudgeProvider,
    parse_json_object,
)
from ..metrics import attribution, multihop, qa, registry, retrieval
from ..store import config_store, run_store
from ..task import TaskContext

DEFAULT_KS = (2, 5, 10)
# 失败率超过这个值就判整轮失败：排除得太多时那个均值不代表整体。
MAX_JUDGE_FAILURE_RATE = 0.1


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _scalars(source: dict[str, Any]) -> dict[str, float]:
    """只取标量项。``reason_counts`` 之类的嵌套结构留在 detail 里。"""
    return {k: float(v) for k, v in source.items() if isinstance(v, (int, float, bool))}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    names = sorted({name for row in rows for name in row["metrics"]})
    return {n: _mean([float(r["metrics"].get(n, 0.0)) for r in rows]) for n in names}


def resolve_metrics(selected: list[str] | None) -> list[str]:
    """校验勾选的指标名。空表示全量。

    拼错的名字直接报错而不是忽略：静默跑全量会让报告比预期多出好几列。
    """
    if not selected:
        return sorted(registry.METRIC_REGISTRY)
    unknown = sorted(set(selected) - set(registry.METRIC_REGISTRY))
    if unknown:
        raise ValueError(f"未知指标：{unknown}")
    return sorted(set(selected))


def _keep(selected: frozenset[str]):
    """实际指标名带 k（``recall@5``），勾选名不带，比对前要剥掉。"""

    def keep(name: str) -> bool:
        try:
            return registry.get_metric(name).name in selected
        except KeyError:
            # registry 里没声明的项保留：让它悄悄消失比留着更糟。
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
    omitted = [
        d.name for d in registry.omitted(provides) if d.kind == registry.KIND_DETERMINISTIC
    ]

    samples = {
        s["sample_id"]: s for s in run_store.compile_samples(connection, compile_id, dataset)
    }
    responses = run_store.responses_of(connection, query_id, dataset)
    if not responses:
        raise ValueError(f"{dataset} 没有查询响应")

    page_to_doc = run_store.page_to_doc(connection, compile_id, dataset) if has_gold else {}
    keep = _keep(selected)
    run_store.clear_eval_results(connection, eval_id, dataset)

    per_sample: list[dict[str, Any]] = []
    http_failures = 0

    for position, row in enumerate(responses, 1):
        sample = samples.get(row["sample_id"])
        if sample is None:
            raise ValueError(
                f"{dataset}: 样本 {row['sample_id']!r} 有响应但不在这次编译的子集里"
            )
        # 按 ID 匹配后再比一次问题文本：ID 对得上而内容变了的情况抓不到别的办法。
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

        # 勾选过滤发生在写库前。detail 保留全部明细 —— 那是归因要读的链路。
        metrics = {name: value for name, value in metrics.items() if keep(name)}
        entry = {
            "sample_id": row["sample_id"],
            "answer_mode": row["answer_mode"],
            "metrics": metrics,
        }
        per_sample.append(entry)

        run_store.record_sample_eval(
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
                ctx.progress(position, len(responses), f"{dataset} {position}/{len(responses)}")
    connection.commit()

    knowledge = [e for e in per_sample if e["answer_mode"] == "knowledge"]
    overall = _aggregate(per_sample)
    run_store.record_metric_summary(
        connection, eval_id, dataset, "overall", overall, len(per_sample)
    )
    # 两份口径都要存：差值就是生成端拒答的规模，而不是检索失败。
    run_store.record_metric_summary(
        connection, eval_id, dataset, "knowledge_only", _aggregate(knowledge), len(knowledge)
    )
    run_store.record_dataset_eval(
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


def resolve_provider(
    connection: sqlite3.Connection, provider_id: int | None, role: str
) -> JudgeProvider:
    """从库里取 provider 配置，凑不齐就抛 :class:`JudgeConfigError`。"""
    record = config_store.get_provider(connection, provider_id) if provider_id else None
    if record is None:
        candidates = config_store.list_providers(connection, role)
        record = candidates[0] if candidates else None
    if record is None:
        raise JudgeConfigError(f"没有配置 {role} 模型端点，请在配置页填写。")
    if record["role"] != role:
        raise JudgeConfigError(f"模型端点 {record['label']!r} 的角色不是 {role}")
    if not (record["api_key"] or "").strip():
        raise JudgeConfigError(f"模型端点 {record['label']!r} 没有 api key。")
    return JudgeProvider(
        base_url=record["base_url"],
        model=record["model"],
        api_key=record["api_key"],
    )


def _judge(ctx: TaskContext, eval_id: int, query_id: int, provider: JudgeProvider) -> None:
    """跑 faithfulness。**失败该条排除，不记 0** —— 记 0 会让限流伪装成质量差。"""
    already = run_store.judged_sample_ids(ctx.db, eval_id)
    rows = [r for r in run_store.sample_evals(ctx.db, eval_id) if r["sample_id"] not in already]
    if already:
        ctx.log(f"Judge 续跑：已判 {len(already)} 条，待判 {len(rows)} 条")

    with JudgeClient(provider) as client:
        for position, row in enumerate(rows, 1):
            ctx.checkpoint()
            response = run_store.response_of(ctx.db, query_id, row["sample_id"])
            body = (response or {}).get("response") or {}
            prompt = faithfulness.build_prompt(
                (response or {}).get("question") or "", row["answer"] or "", body
            )
            if prompt is None:
                # 上下文为空（拒答会清空 retrievedSources），此时指标无定义。
                run_store.record_judge_verdict(
                    ctx.db,
                    eval_id,
                    sample_id=row["sample_id"],
                    score=None,
                    failure_kind=None,
                    detail={"skipped": "no retrieved context"},
                )
            else:
                reply = client.complete(*prompt)
                if reply.failure_kind:
                    run_store.record_judge_verdict(
                        ctx.db,
                        eval_id,
                        sample_id=row["sample_id"],
                        score=None,
                        failure_kind=reply.failure_kind,
                        detail={"raw": (reply.raw or "")[:500]},
                    )
                else:
                    try:
                        score, reasoning = faithfulness.parse_verdict(
                            parse_json_object(reply.content or "")
                        )
                    except ValueError as exc:
                        # 模型没按 schema 输出。**不猜分数** —— 那会把
                        # 「prompt 不听话」伪装成「答案不忠实」。
                        run_store.record_judge_verdict(
                            ctx.db,
                            eval_id,
                            sample_id=row["sample_id"],
                            score=None,
                            failure_kind="parse_error",
                            detail={"error": str(exc)[:300]},
                        )
                    else:
                        run_store.record_judge_verdict(
                            ctx.db,
                            eval_id,
                            sample_id=row["sample_id"],
                            score=score,
                            failure_kind=None,
                            detail=reasoning,
                        )
                        if score is not None:
                            run_store.record_sample_eval(
                                ctx.db,
                                eval_id,
                                sample_id=row["sample_id"],
                                dataset=row["dataset"],
                                answer_mode=row["answer_mode"],
                                http_status=row["http_status"],
                                answer=row["answer"] or "",
                                detail=row["detail"],
                                metrics={"faithfulness": score},
                            )
            ctx.db.commit()
            if position % 5 == 0 or position == len(rows):
                ctx.progress(position, len(rows), f"Judge {position}/{len(rows)}")

    summary = run_store.judge_summary(ctx.db, eval_id)
    for dataset, mean, count in run_store.judge_means_by_dataset(ctx.db, eval_id):
        run_store.record_metric_summary(
            ctx.db, eval_id, dataset, "judge", {"faithfulness": mean}, count
        )
    ctx.db.commit()
    ctx.log(
        f"faithfulness 均值 {summary['mean']}，已评分 {summary['scored']}，"
        f"失败率 {summary['failure_rate']:.3f}"
    )
    if summary["failure_rate"] > MAX_JUDGE_FAILURE_RATE:
        raise RuntimeError(
            f"Judge 失败率 {summary['failure_rate']:.3f} 超过 {MAX_JUDGE_FAILURE_RATE}。"
            f"均值只覆盖成功的那部分，已不代表整体。失败类型：{summary['failures_by_kind']}"
        )


def run(ctx: TaskContext) -> None:
    params = ctx.params
    query_id = int(params.get("query_id") or 0)
    if not query_id:
        raise ValueError("请选择一次查询")

    query_run = run_store.get_query_run(ctx.db, query_id)
    if query_run is None:
        raise ValueError(f"查询 #{query_id} 不存在")
    if query_run["status"] != run_store.STATUS_SUCCEEDED:
        raise ValueError("请选择已完成的查询记录")

    available = run_store.response_datasets(ctx.db, query_id)
    datasets = list(params.get("datasets") or available)
    unknown = sorted(set(datasets) - set(available))
    if unknown:
        raise ValueError(f"这些数据集没有查询响应：{unknown}")

    ks = tuple(int(k) for k in (params.get("ks") or DEFAULT_KS))
    if any(k < 1 for k in ks):
        raise ValueError("k 必须大于 0")
    metrics = resolve_metrics(list(params.get("metrics") or []))
    judge_selected = any(
        registry.get_metric(name).kind == registry.KIND_JUDGE for name in metrics
    )

    provider: JudgeProvider | None = None
    provider_id = params.get("judge_provider_id")
    provider_id = int(provider_id) if provider_id else None
    if judge_selected:
        provider = resolve_provider(ctx.db, provider_id, "judge")

    name = str(params.get("name") or "").strip() or f"{query_run['name']}-e"
    existing = run_store.eval_run_by_name(ctx.db, name)
    if existing:
        if int(existing["query_id"]) != query_id:
            raise ValueError(f"评测记录 {name!r} 属于另一次查询，请换个名称")
        eval_id = int(existing["id"])
        run_store.set_eval_status(ctx.db, eval_id, run_store.STATUS_RUNNING)
    else:
        eval_id = run_store.create_eval_run(
            ctx.db,
            name=name,
            query_id=query_id,
            ks=list(ks),
            metrics=metrics,
            judge_provider_id=provider_id if judge_selected else None,
        )
    ctx.db.commit()
    ctx.bind("eval", eval_id)

    try:
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
            ctx.log(f"执行 Judge：{provider.model}")
            _judge(ctx, eval_id, query_id, provider)
    except BaseException:
        run_store.set_eval_status(
            ctx.db,
            eval_id,
            run_store.STATUS_PAUSED if ctx.pause_requested else run_store.STATUS_FAILED,
        )
        ctx.db.commit()
        raise

    run_store.set_eval_status(ctx.db, eval_id, run_store.STATUS_SUCCEEDED, finished=True)
    ctx.db.commit()
    ctx.progress(len(datasets), len(datasets), "评测完成")
