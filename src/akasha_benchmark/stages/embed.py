from __future__ import annotations

import time
from typing import Any

from ..akasha_client import AkashaClient
from ..config import load_config
from ..lineage import LineageReader, LineageUnavailable, active_coverage
from ..model_configs import feature_of, matches
from ..store import compile_store
from ..task import TaskContext

POLL_INTERVAL_SECONDS = 15.0
MAX_IDLE_POLLS = 20
# 远端 AdminKnowledgeSpaceActionDto 限制 1-50，留空则用它的默认并发 2
MAX_EMBEDDING_CONCURRENCY = 50


def _target_model(configs: dict[str, Any]) -> str:
    entry = feature_of(configs, "embedding") or {}
    model = str(entry.get("model") or "")
    if not model:
        raise RuntimeError("远端没有配置 embedding 模型，无法重建")
    return model


def _coverage(reader: LineageReader, space_id: str, model: str) -> dict[str, Any]:
    report = reader.space_embedding_generations(space_id)
    return {
        **active_coverage(report["generations"], model),
        "generations": report["generations"],
    }


def _usage_note(generations: list[dict[str, Any]], model: str) -> str:
    """累计调用耗时按单次 embed 墙钟相加，远端并发执行，不等于真实墙钟。"""
    picked = [
        entry for entry in generations if entry["active"] and entry["model"] == model
    ]
    chunks = sum(entry["chunks"] for entry in picked)
    tokens = sum(entry.get("total_tokens") or 0 for entry in picked)
    duration = sum(entry.get("duration_ms_sum") or 0 for entry in picked)
    per_chunk = duration / chunks if chunks else 0
    return (
        f"{chunks} 个 chunk，token 合计 {tokens}，"
        f"累计调用耗时 {duration / 1000:.1f}s（非墙钟），约 {per_chunk:.0f}ms/chunk"
    )


def run(ctx: TaskContext) -> None:
    compile_id = int(ctx.params.get("compile_id") or 0)
    if not compile_id:
        raise ValueError("请选择一次编译")

    raw_concurrency = ctx.params.get("embedding_concurrency")
    concurrency = None if raw_concurrency is None else int(raw_concurrency)
    if concurrency is not None and not 1 <= concurrency <= MAX_EMBEDDING_CONCURRENCY:
        raise ValueError(f"嵌入并发必须在 1 到 {MAX_EMBEDDING_CONCURRENCY} 之间")

    compile_run = compile_store.get_compile_run(ctx.db, compile_id)
    if compile_run is None:
        raise ValueError(f"编译 #{compile_id} 不存在")
    space_id = compile_run["space_id"]
    if not space_id:
        raise ValueError(f"编译 {compile_run['run_id']!r} 还没有 Akasha 空间")

    config = load_config(ctx.db)
    config.require_credentials()
    try:
        reader = LineageReader(config.database_url)
    except LineageUnavailable as exc:
        raise RuntimeError(
            f"重建嵌入需要 PostgreSQL 判定完成度：{exc}。"
            "重建时旧向量是标 stale 而不是置空，Akasha 的质量闸门全程为 0，HTTP 侧判不了进度。"
        ) from exc

    with AkashaClient(config) as client:
        client.login()
        me = client.current_user()
        role = ((me or {}).get("user") or {}).get("role")
        if role not in {"owner", "admin"}:
            raise RuntimeError(f"当前账号角色是 {role!r}，重建嵌入需要 owner 或 admin")
        mismatch = compile_store.workspace_mismatch(
            ctx.db, compile_id, ((me or {}).get("workspace") or {}).get("id")
        )
        if mismatch:
            raise RuntimeError(mismatch)

        current = client.get_model_configs()
        target = _target_model(current)
        ctx.freeze(model_configs=current, target_model=target)
        ctx.log(f"重建 {compile_run['run_id']} 的嵌入，目标模型 {target}")

        coverage = _coverage(reader, space_id, target)
        if not coverage["total"]:
            ctx.log("空间里没有活跃 chunk，先编译再重建")
            return
        ctx.log(f"当前活跃代模型：{', '.join(coverage['models']) or '（无）'}")

        ctx.checkpoint()
        result = client.rebuild_space_embeddings(
            [str(space_id)], embedding_concurrency=concurrency
        )
        job_ids = [str(value) for value in result.get("jobIds") or []]
        if not job_ids:
            raise RuntimeError(f"Akasha 没有返回重建 job：{result!r}")
        ctx.freeze(rebuild_job_ids=job_ids)
        ctx.log(
            f"已请求重建，远端 job {', '.join(job_ids)}，"
            f"嵌入并发 {concurrency if concurrency else '远端默认'}"
        )

        if coverage["ready"]:
            coverage = _wait_for_refresh(ctx, reader, str(space_id), target, coverage)
        else:
            coverage = _wait_for_rebuild(ctx, reader, str(space_id), target, coverage)

        final_configs = client.get_model_configs()
        if not matches(final_configs, current, "embedding"):
            raise RuntimeError("重建期间远端 embedding 配置发生变化，结果可能混合")

    ctx.log(f"重建完成：{_usage_note(coverage['generations'], target)}")


def _wait_for_rebuild(
    ctx: TaskContext,
    reader: LineageReader,
    space_id: str,
    target: str,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    """远端对重建没有 run 记录，只能看覆盖率；长时间不涨就停，避免无限等。"""
    matched = coverage["matched"]
    idle = 0
    while not coverage["ready"]:
        ctx.progress(coverage["matched"], coverage["total"] or None, "重建嵌入")
        time.sleep(POLL_INTERVAL_SECONDS)
        ctx.checkpoint()
        coverage = _coverage(reader, space_id, target)
        if coverage["matched"] > matched:
            matched = coverage["matched"]
            idle = 0
            continue
        idle += 1
        if idle >= MAX_IDLE_POLLS:
            raise RuntimeError(
                f"重建停滞：{int(POLL_INTERVAL_SECONDS * MAX_IDLE_POLLS / 60)} 分钟内"
                f"覆盖率停在 {coverage['matched']} / {coverage['total']}，"
                "请检查 Akasha 的 worker 和 embedding 端点"
            )
    ctx.progress(coverage["total"], coverage["total"], "重建完成")
    return coverage


def _usage_fingerprint(generations: list[dict[str, Any]], model: str) -> tuple:
    picked = [
        entry for entry in generations if entry["active"] and entry["model"] == model
    ]
    return tuple(
        sorted(
            (
                entry["chunks"],
                entry.get("embedded") or 0,
                entry.get("total_tokens") or 0,
                entry.get("duration_ms_sum") or 0,
                entry.get("last_created_at") or "",
            )
            for entry in picked
        )
    )


def _wait_for_refresh(
    ctx: TaskContext,
    reader: LineageReader,
    space_id: str,
    target: str,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    """模型没变时覆盖率恒为满，改看活跃代用量指纹：先变化再稳定才算重建完。"""
    ctx.log("模型与活跃代一致，按用量变化判定重建进度")
    before = _usage_fingerprint(coverage["generations"], target)
    current = before
    idle = 0
    while idle < MAX_IDLE_POLLS:
        ctx.progress(
            coverage["total"] if current != before else 0,
            coverage["total"] or None,
            "重建嵌入" if current == before else "等待用量稳定",
        )
        time.sleep(POLL_INTERVAL_SECONDS)
        ctx.checkpoint()
        coverage = _coverage(reader, space_id, target)
        fingerprint = _usage_fingerprint(coverage["generations"], target)
        if fingerprint != current:
            current = fingerprint
            idle = 0
            continue
        idle += 1
    if current == before:
        raise RuntimeError(
            f"重建没有生效：{int(POLL_INTERVAL_SECONDS * MAX_IDLE_POLLS / 60)} 分钟内"
            f"活跃代 {target} 的用量没有变化，请检查 Akasha 的 worker 和 embedding 端点"
        )
    ctx.progress(coverage["total"], coverage["total"], "重建完成")
    return coverage
