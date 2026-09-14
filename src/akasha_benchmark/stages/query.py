"""查询层：选一次编译的空间，逐条跑 query，把完整响应写库。

**这一层不算任何指标。** 它只产出证据，解释证据是评测的事。
存完整响应体而不是当下用得到的那几个字段 —— 重跑要烧 LLM 调用，
以后想看某个新字段时不该被迫重跑。

失败也照样写一行：失败率本身是结果，静默跳过会把后面所有均值算高。
"""

from __future__ import annotations

import queue
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from ..akasha_client import AkashaClient, AkashaError
from ..config import load_config
from ..model_configs import drift
from ..store import loads, run_store
from ..task import TaskContext


def _query_one(
    client: AkashaClient,
    sample: dict[str, Any],
    space_id: str,
    score_threshold: float | None,
) -> dict[str, Any]:
    """跑一条 query。失败也返回可落库的行。"""
    try:
        response = client.query(
            sample["question"], [space_id], score_threshold=score_threshold
        )
        status, body, latency = response.status, response.body, response.latency_ms
        error = None
    except (AkashaError, httpx.RequestError, OSError) as exc:
        # httpx.RequestError 不是 OSError 的子类，少了它一次网络抖动会让
        # 整个阶段带 traceback 崩掉，而那一行也不会落库。
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
        samples = run_store.compile_samples(connection, compile_id, dataset)
        picked.extend(samples[:limit] if limit else samples)
    return picked


def run(ctx: TaskContext) -> None:
    params = ctx.params
    compile_id = int(params.get("compile_id") or 0)
    if not compile_id:
        raise ValueError("请选择一次编译")

    compile_run = run_store.get_compile_run(ctx.db, compile_id)
    if compile_run is None:
        raise ValueError(f"编译 #{compile_id} 不存在")

    # 不可跳过的前置闸门。半成品索引会产出一份看起来像配置问题的报告。
    readiness = run_store.compile_ready(ctx.db, compile_id)
    if not readiness["ready"]:
        raise ValueError("这次编译还不能用于查询：" + "；".join(readiness["reasons"]))

    datasets = list(params.get("datasets") or loads(compile_run["datasets_json"], []))
    available = set(run_store.compile_stats(ctx.db, compile_id))
    unknown = sorted(set(datasets) - available)
    if unknown:
        raise ValueError(f"这次编译不含数据集：{unknown}")

    limit = params.get("sample_limit")
    limit = int(limit) if limit else None
    score_threshold = params.get("score_threshold")
    score_threshold = float(score_threshold) if score_threshold not in (None, "") else None

    config = load_config(ctx.db)
    config.require_credentials()
    concurrency = max(1, int(params.get("concurrency") or config.concurrency))

    with AkashaClient(config) as client:
        client.login()

        # workspace 不匹配拒绝执行。这一层的 space_id 只在它编译时那个 workspace 里
        # 解析得到 —— 换个地方跑，每条 query 都会打到一个空 space，而那不报错，
        # 只会给出一份 recall 全 0 的报告。
        me = client.current_user()
        mismatch = run_store.workspace_mismatch(
            ctx.db, compile_id, ((me or {}).get("workspace") or {}).get("id")
        )
        if mismatch:
            raise RuntimeError(mismatch)

        current = client.get_model_configs()
        snapshot = loads(compile_run["model_configs_json"])
        changed = drift(current, snapshot)
        if changed["embedding"]:
            # 这一项不可绕过：旧 chunk 的 embedding_profile 对不上，
            # 那些 chunk 永远召回不到，而评测会照常算出一份看着合理的坏报告。
            raise RuntimeError(
                "embedding 模型在这次编译之后改过了。旧 chunk 永远召回不到，"
                "检索指标会全部错但不报错。请重新编译。"
            )
        for feature in ("compiler", "answer", "image"):
            if changed[feature]:
                ctx.log(f"{feature} 模型与编译时不同，两次运行不可比", "warn")

        name = str(params.get("name") or "").strip() or f"{compile_run['run_id']}-q"
        existing = run_store.query_run_by_name(ctx.db, name)
        if existing:
            if int(existing["compile_id"]) != compile_id:
                raise ValueError(f"查询记录 {name!r} 属于另一次编译，请换个名称")
            query_id = int(existing["id"])
            run_store.set_query_status(ctx.db, query_id, run_store.STATUS_RUNNING)
        else:
            query_id = run_store.create_query_run(
                ctx.db,
                name=name,
                compile_id=compile_id,
                score_threshold=score_threshold,
                concurrency=concurrency,
                model_configs=current,
            )
        ctx.db.commit()
        ctx.bind("query", query_id)

        # 固化这一轮问哪些样本，之后续跑以它为准。
        selected = _select_samples(ctx.db, compile_id, datasets, limit)
        if not selected:
            raise ValueError("所选数据集在这次编译里没有样本")
        run_store.freeze_query_samples(ctx.db, query_id, selected)
        ctx.db.commit()

        if params.get("retry_failed"):
            removed = run_store.delete_failed_responses(ctx.db, query_id)
            ctx.db.commit()
            ctx.log(f"已清除 {removed} 条失败响应以便重试")

        try:
            _issue(ctx, client, query_id, compile_run["space_id"], score_threshold, concurrency)
        except BaseException:
            run_store.set_query_status(
                ctx.db,
                query_id,
                run_store.STATUS_PAUSED if ctx.pause_requested else run_store.STATUS_FAILED,
            )
            ctx.db.commit()
            raise

    run_store.set_query_status(ctx.db, query_id, run_store.STATUS_SUCCEEDED, finished=True)
    ctx.db.commit()
    stats = run_store.query_stats(ctx.db, query_id)
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
    score_threshold: float | None,
    concurrency: int,
) -> None:
    """发请求并逐条落库。已有响应的样本跳过，所以暂停后继续是续跑。"""
    todo = run_store.pending_query_samples(ctx.db, query_id)
    total = len(run_store.query_samples(ctx.db, query_id))
    done = total - len(todo)
    if not todo:
        ctx.log(f"{total} 条样本均已有响应，跳过")
        return
    ctx.log(f"待查询 {len(todo)} / 共 {total}，并发 {concurrency}")

    def emit(row: dict[str, Any]) -> None:
        nonlocal done
        run_store.record_response(
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
        # 逐条提交，前端才看得到进度。
        ctx.db.commit()
        done += 1
        if done % 5 == 0 or done == total:
            ctx.progress(done, total, f"查询 {done}/{total}")

    if concurrency <= 1:
        for sample in todo:
            ctx.checkpoint()
            emit(_query_one(client, sample, space_id, score_threshold))
        return

    # 每个 worker 一个独立客户端：限流器用实例上的 _last_request_at，
    # 多线程共用一个实例会退化成「一起睡、一起发」。
    # 落库只在主线程做，所以 sqlite 连接不跨线程使用。
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
                return _query_one(borrowed, sample, space_id, score_threshold)
            finally:
                pool.put(borrowed)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # 分批提交而不是一次全提交：暂停时只需要等当前这批收尾。
            for start in range(0, len(todo), concurrency):
                ctx.checkpoint()
                batch = todo[start : start + concurrency]
                for row in executor.map(task, batch):
                    emit(row)
    finally:
        for spare in extra:
            spare.close()
