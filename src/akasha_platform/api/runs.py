"""编译层与查询层的读取与清理。

清理就是删主表那一行：产物表全部 ``ON DELETE CASCADE``，所以一条 DELETE
清掉这一层及其下游的全部数据库内容。编译层清理前会取消空间中的活动 Run，
远端 space 与审计日志保留。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from akasha_benchmark.akasha_client import (
    ACTIVE_RUN_STATUSES,
    AkashaClient,
    AkashaError,
    validate_cancel_result,
)
from akasha_benchmark.config import load_config
from akasha_benchmark.store import (
    compile_store,
    query_store,
)

from ..run_tree import build_compile_tree
from ._common import db, reject_if_busy, writable

router = APIRouter(prefix="/api")

DEFAULT_PAGE = 20
MAX_PAGE = 200



@router.get("/compiles")
def list_compiles(request: Request) -> dict[str, Any]:
    """编译记录树：每次编译连同它的查询、评测、归因。各层的选择器都读它。"""
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
    """本次编译的语料，带 ``page_id``（走链路视图的钥匙）。

    没导入成功的 ``page_id`` 为空，它们照样列出 —— 缺篇本身是要看的信息。
    """
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


@router.delete("/compiles/{compile_id}")
def delete_compile(request: Request, compile_id: int) -> dict[str, Any]:
    """取消活动的远端 Run，再清理编译及其下游；远端 space 保留。"""
    with writable(request) as connection:
        row = compile_store.get_compile_run(connection, compile_id)
        if row is None:
            raise HTTPException(404, f"编译 #{compile_id} 不存在")
        reject_if_busy(connection, "compile", compile_id)
        cancelled = 0
        removed_jobs = 0
        if row.get("space_id"):
            try:
                with AkashaClient(load_config(connection)) as client:
                    client.login()
                    runs = client.run_diagnostics([row["space_id"]], limit=50).get("items") or []
                    for run in runs:
                        if run.get("runId") and str(run.get("status")) in ACTIVE_RUN_STATUSES:
                            result = client.cancel_compile_run(
                                str(run["runId"]), "Akasha-Benchmark compile cleaned up"
                            )
                            remote = validate_cancel_result(str(run["runId"]), result)
                            cancelled += remote["disposition"] == "cancelled"
                            removed_jobs += int(result.get("removedJobCount") or 0)
            except (AkashaError, ValueError) as exc:
                raise HTTPException(502, f"远端编译取消失败，本地记录未删除：{exc}") from exc
        removed = compile_store.delete_compile_run(connection, compile_id)
    return {
        "deleted": removed,
        "space_id": row["space_id"],
        "cancelled_runs": cancelled,
        "removed_bullmq_jobs": removed_jobs,
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
    """查询结果列表，按 answerMode 另给一份计数。

    计数是必要的：``no_match`` / ``general`` 的检索得分按定义为 0，
    混在一起读会把生成端拒答误当成检索失败。
    """
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
    """单条完整响应体。"""
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
    """清理一次查询及其下游的评测、归因。"""
    with writable(request) as connection:
        if query_store.get_query_run(connection, query_id) is None:
            raise HTTPException(404, f"查询 #{query_id} 不存在")
        reject_if_busy(connection, "query", query_id)
        removed = query_store.delete_query_run(connection, query_id)
    return {"deleted": removed}
