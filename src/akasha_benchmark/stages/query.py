"""查询层：选一次编译的空间，逐条跑 query，把完整响应写库。

这一层不算指标，只产出证据。存完整响应体，免得以后想看某个新字段时被迫重跑。
失败也照样写一行：失败率本身是结果，静默跳过会把后面所有均值算高。
"""

from __future__ import annotations

import queue
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from ..akasha_client import AkashaClient, AkashaError
from ..config import load_config
from ..model_configs import drift, matches
from ..store import compile_store, loads, query_store
from ..task import TaskContext


def _query_one(
    client: AkashaClient,
    sample: dict[str, Any],
    space_id: str,
) -> dict[str, Any]:
    """跑一条 query。失败也返回可落库的行。"""
    try:
        response = client.query(sample["question"], [space_id])
        status, body, latency = response.status, response.body, response.latency_ms
        error = None
    except (AkashaError, httpx.RequestError, OSError) as exc:
        status, body, latency, error = 0, None, None, f"{type(exc).__name__}: {exc}"

    return {
        "sample_id": sample["sample_id"],
        "dataset": sample["dataset"],
        "question": sample["question"],
        "http_status": status,
        "latency_ms": latency,
        "error": error,
        "response": body,
    }


def _select_samples(
    connection: sqlite3.Connection,
    compile_id: int,
    datasets: list[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for dataset in datasets:
        samples = compile_store.compile_samples(connection, compile_id, dataset)
        picked.extend(samples[:limit] if limit else samples)
    return picked


def run(ctx: TaskContext) -> None:
    params = ctx.params
    compile_id = int(params.get("compile_id") or 0)
    if not compile_id:
        raise ValueError("请选择一次编译")

    compile_run = compile_store.get_compile_run(ctx.db, compile_id)
    if compile_run is None:
        raise ValueError(f"编译 #{compile_id} 不存在")

    # 不可跳过的前置闸门：半成品索引会产出一份看着合理的坏报告。
    readiness = compile_store.compile_ready(ctx.db, compile_id)
    if not readiness["ready"]:
        raise ValueError("这次编译还不能用于查询：" + "；".join(readiness["reasons"]))

    datasets = list(params.get("datasets") or loads(compile_run["datasets_json"], []))
    available = set(compile_store.compile_stats(ctx.db, compile_id))
    unknown = sorted(set(datasets) - available)
    if unknown:
        raise ValueError(f"这次编译不含数据集：{unknown}")

    limit = params.get("sample_limit")
    limit = int(limit) if limit else None
    config = load_config(ctx.db)
    config.require_credentials()
    concurrency = max(1, int(params.get("concurrency") or 3))
    query_id = ctx.target("query")
    resuming = query_id is not None

    with AkashaClient(config) as client:
        client.login()

        # workspace 不匹配拒绝执行：换个地方跑会打到一个空 space，且不报错。
        me = client.current_user()
        mismatch = compile_store.workspace_mismatch(
            ctx.db, compile_id, ((me or {}).get("workspace") or {}).get("id")
        )
        if mismatch:
            raise RuntimeError(mismatch)

        current = client.get_model_configs()
        snapshot = loads(compile_run["model_configs_json"])
        changed = drift(current, snapshot)
        if changed["embedding"]:
            raise RuntimeError("远端 embedding 配置与编译时不同，必须重新编译")
        if resuming:
            existing_query = query_store.get_query_run(ctx.db, query_id) or {}
            query_snapshot = loads(existing_query.get("model_configs_json"))
            resume_changed = [
                feature
                for feature in ("embedding", "answer")
                if query_snapshot and not matches(current, query_snapshot, feature)
            ]
            if resume_changed:
                raise RuntimeError(
                    "远端模型配置已变化，不能继续原查询：" + ", ".join(resume_changed)
                )
        for feature in ("compiler", "embedding", "answer", "image"):
            if changed[feature] and feature != "embedding":
                ctx.log(
                    f"{feature} 配置与编译时不同；允许查询，两次运行不可完全对比",
                    "warn",
                )

        name = (
            str(params.get("name") or "").strip()
            or f"{compile_run['run_id']}-q-{uuid.uuid4().hex[:6]}"
        )
        ctx.freeze(
            name=name,
            datasets=datasets,
            concurrency=concurrency,
            sample_limit=limit,
        )
        if query_id is None:
            if query_store.query_run_by_name(ctx.db, name):
                raise ValueError(f"查询名称 {name!r} 已存在，请换个名称或继续原任务")
            query_id = query_store.create_query_run(
                ctx.db,
                name=name,
                compile_id=compile_id,
                concurrency=concurrency,
                model_configs=current,
            )
            ctx.bind("query", query_id)

        # 固化这一轮问哪些样本，续跑以它为准。
        if not query_store.query_samples(ctx.db, query_id):
            selected = _select_samples(ctx.db, compile_id, datasets, limit)
            if not selected:
                raise ValueError("所选数据集在这次编译里没有样本")
            query_store.freeze_query_samples(ctx.db, query_id, selected)
            ctx.db.commit()

        if params.get("retry_failed"):
            removed = query_store.delete_failed_responses(ctx.db, query_id)
            ctx.db.commit()
            ctx.log(f"已清除 {removed} 条失败响应以便重试")
            ctx.freeze(retry_failed=False)

        _issue(ctx, client, query_id, compile_run["space_id"], concurrency)
        final_configs = client.get_model_configs()
        changed_during_run = [
            feature
            for feature in ("embedding", "answer")
            if not matches(final_configs, current, feature)
        ]
        if changed_during_run:
            raise RuntimeError(
                "查询期间远端模型配置发生变化，结果可能混合："
                + ", ".join(changed_during_run)
            )

    stats = query_store.query_stats(ctx.db, query_id)
    for dataset, row in stats.items():
        ctx.log(
            f"{dataset}: 响应 {row['responses']}，失败 {row['failures']}，"
            f"平均 {int(row['latency_mean'] or 0)}ms"
        )


def _issue(
    ctx: TaskContext,
    client: AkashaClient,
    query_id: int,
    space_id: str,
    concurrency: int,
) -> None:
    """发请求并逐条落库。已有响应的样本跳过，所以暂停后继续即续跑。"""
    todo = query_store.pending_query_samples(ctx.db, query_id)
    total = len(query_store.query_samples(ctx.db, query_id))
    done = total - len(todo)
    if not todo:
        ctx.log(f"{total} 条样本均已有响应，跳过")
        return
    ctx.log(f"待查询 {len(todo)} / 共 {total}，并发 {concurrency}")

    def emit(row: dict[str, Any]) -> None:
        nonlocal done
        query_store.record_response(
            ctx.db,
            query_id,
            sample_id=row["sample_id"],
            dataset=row["dataset"],
            question=row["question"],
            http_status=row["http_status"],
            latency_ms=row["latency_ms"],
            error=row["error"],
            response=row["response"],
        )
        # 逐条提交，前端因此看得到进度。
        ctx.db.commit()
        done += 1
        if done % 5 == 0 or done == total:
            ctx.progress(done, total, "查询")

    if concurrency <= 1:
        for sample in todo:
            ctx.checkpoint()
            emit(_query_one(client, sample, space_id))
        return

    # 每个 worker 一个独立客户端：限流器用实例上的 _last_request_at，
    # 共用一个实例会退化成「一起睡、一起发」。落库只在主线程做。
    extra = [AkashaClient(client.config) for _ in range(concurrency - 1)]
    try:
        for spare in extra:
            spare.login()
        pool: queue.Queue[AkashaClient] = queue.Queue()
        for worker in (client, *extra):
            pool.put(worker)

        def task(sample: dict[str, Any]) -> dict[str, Any]:
            borrowed = pool.get()
            try:
                return _query_one(borrowed, sample, space_id)
            finally:
                pool.put(borrowed)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # 分批提交，暂停时只需要等当前这批收尾。
            for start in range(0, len(todo), concurrency):
                ctx.checkpoint()
                batch = todo[start : start + concurrency]
                for row in executor.map(task, batch):
                    emit(row)
    finally:
        for spare in extra:
            spare.close()
