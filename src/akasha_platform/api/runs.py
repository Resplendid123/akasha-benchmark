from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

from akasha_benchmark.akasha_client import AkashaClient, AkashaError
from akasha_benchmark.config import load_config
from akasha_benchmark.lineage import (
    BadPageId,
    LineageReader,
    LineageUnavailable,
    active_coverage,
)
from akasha_benchmark.model_configs import feature_of
from akasha_benchmark.store import (
    compile_store,
    query_store,
)

from ..run_tree import build_compile_tree
from ..compile_control import CompileRemoteService
from ._common import db, reject_if_busy, writable

router = APIRouter(prefix="/api")
compile_remote = CompileRemoteService()

DEFAULT_PAGE = 20
MAX_PAGE = 200


@router.get("/compiles")
def list_compiles(request: Request) -> dict[str, Any]:

    with db(request) as connection:
        return {"compiles": build_compile_tree(connection)}


@router.get("/compiles/{compile_id}/docs")
def compile_docs(
    request: Request,
    compile_id: int,
    dataset: str | None = None,
    gold_only: bool = False,
    q: str | None = None,
    limit: int = Query(DEFAULT_PAGE, ge=1, le=MAX_PAGE),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    with db(request) as connection:
        if compile_store.get_compile_run(connection, compile_id) is None:
            raise HTTPException(404, f"编译 #{compile_id} 不存在")
        total, imported, docs = compile_store.compile_doc_page(
            connection,
            compile_id,
            dataset,
            gold_only=gold_only,
            search=(q or "").strip() or None,
            limit=limit,
            offset=offset,
        )
    return {
        "compile_id": compile_id,
        "total": total,
        "imported": imported,
        "offset": offset,
        "limit": limit,
        "docs": docs,
    }


@router.get("/compiles/{compile_id}/embedding")
def compile_embedding(request: Request, compile_id: int) -> dict[str, Any]:
    """远端当前 embedding 配置，与空间里各代 chunk 的覆盖率和用量。"""
    with db(request) as connection:
        row = compile_store.get_compile_run(connection, compile_id)
        if row is None:
            raise HTTPException(404, f"编译 #{compile_id} 不存在")
        config = load_config(connection)
    space_id = row["space_id"]
    if not space_id:
        raise HTTPException(409, f"编译 {row['run_id']!r} 还没有 Akasha 空间")

    try:
        config.require_credentials()
        with AkashaClient(config) as client:
            client.login()
            live = client.get_model_configs()
    except AkashaError as exc:
        raise HTTPException(502, str(exc)) from exc
    except (httpx.RequestError, OSError, ValueError) as exc:
        raise HTTPException(502, f"读远端模型配置失败：{type(exc).__name__}: {exc}") from exc

    target = feature_of(live, "embedding") or {}
    try:
        report = LineageReader(config.database_url).space_embedding_generations(str(space_id))
    except LineageUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except BadPageId as exc:
        raise HTTPException(400, str(exc)) from exc

    return {
        "compile_id": compile_id,
        "space_id": space_id,
        "target": {
            "model": target.get("model"),
            "dimensions": (target.get("parameters") or {}).get("dimension"),
        },
        "coverage": active_coverage(report["generations"], str(target.get("model") or "")),
        "generations": report["generations"],
    }


@router.delete("/compiles/{compile_id}")
def delete_compile(request: Request, compile_id: int) -> dict[str, Any]:

    with writable(request) as connection:
        row = compile_store.get_compile_run(connection, compile_id)
        if row is None:
            raise HTTPException(404, f"编译 #{compile_id} 不存在")
        reject_if_busy(connection, "compile", compile_id)
        try:
            remote = compile_remote.cancel_compile(
                connection,
                compile_id,
                "Akasha-Benchmark compile cleaned up",
                client_factory=AkashaClient,
            )
        except (AkashaError, ValueError) as exc:
            raise HTTPException(502, f"远端编译取消失败，本地记录未删除：{exc}") from exc
        removed = compile_store.delete_compile_run(connection, compile_id)
    return {
        "deleted": removed,
        "space_id": row["space_id"],
        "cancelled_runs": remote["cancelled"],
        "removed_bullmq_jobs": remote["removed_jobs"],
        "note": "数据库内容已清理，活动编译 Run 已取消；Akasha 空间没有删除。",
    }


@router.get("/queries/{query_id}/responses")
def query_responses(
    request: Request,
    query_id: int,
    dataset: str | None = None,
    answer_mode: str | None = None,
    q: str | None = None,
    limit: int = Query(DEFAULT_PAGE, ge=1, le=MAX_PAGE),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    with db(request) as connection:
        if query_store.get_query_run(connection, query_id) is None:
            raise HTTPException(404, f"查询 #{query_id} 不存在")
        total, counts, rows = query_store.response_page(
            connection,
            query_id,
            dataset=dataset,
            answer_mode=answer_mode,
            search=(q or "").strip() or None,
            limit=limit,
            offset=offset,
        )

    return {
        "query_id": query_id,
        "total": total,
        "count_by_answer_mode": counts,
        "offset": offset,
        "limit": limit,
        "responses": rows,
    }


@router.get("/queries/{query_id}/responses/{sample_id}")
def query_response(request: Request, query_id: int, sample_id: str) -> dict[str, Any]:

    with db(request) as connection:
        row = query_store.response_of(connection, query_id, sample_id)
        if row is None:
            raise HTTPException(404, f"查询 #{query_id} 里没有 {sample_id!r} 的响应")
        return row


@router.post("/queries/{query_id}/retry-failed")
def retry_failed_query(request: Request, query_id: int) -> dict[str, Any]:
    from ..tasks import TaskRejected

    try:
        return request.app.state.runner.retry_failed_query(query_id)
    except TaskRejected as exc:
        raise HTTPException(409, str(exc)) from exc


@router.delete("/queries/{query_id}")
def delete_query(request: Request, query_id: int) -> dict[str, Any]:

    with writable(request) as connection:
        if query_store.get_query_run(connection, query_id) is None:
            raise HTTPException(404, f"查询 #{query_id} 不存在")
        reject_if_busy(connection, "query", query_id)
        removed = query_store.delete_query_run(connection, query_id)
    return {"deleted": removed}
