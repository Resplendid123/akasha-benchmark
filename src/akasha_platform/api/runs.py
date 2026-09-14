"""编译层与查询层的读取与清理。

清理就是删主表那一行：产物表全部 ``ON DELETE CASCADE``，所以一条 DELETE
清掉这一层及其下游的全部数据库内容。远端的 space 与审计日志都不删。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from akasha_benchmark.store import loads, run_store

from ._common import db, writable

router = APIRouter(prefix="/api")

DEFAULT_PAGE = 20
MAX_PAGE = 200


# ------------------------------------------------------------------ 编译层


@router.get("/compiles")
def list_compiles(request: Request) -> dict[str, Any]:
    """编译记录树：每次编译连同它的查询、评测、归因。各层的选择器都读它。"""
    with db(request) as connection:
        runs = []
        for row in run_store.list_compile_runs(connection):
            compile_id = int(row["id"])
            queries = []
            for q in run_store.list_query_runs(connection, compile_id):
                query_id = int(q["id"])
                queries.append(
                    {
                        **_public(q),
                        "stats": run_store.query_stats(connection, query_id),
                        "evals": [
                            {
                                **_public(e),
                                "ks": loads(e["ks_json"], []),
                                "metrics": loads(e["metrics_json"], []),
                                "attributions": [
                                    _public(a)
                                    for a in run_store.list_attribution_runs(
                                        connection, int(e["id"])
                                    )
                                ],
                            }
                            for e in run_store.list_eval_runs(connection, query_id)
                        ],
                    }
                )
            runs.append(
                {
                    **_public(row),
                    "datasets": loads(row["datasets_json"], []),
                    "stats": run_store.compile_stats(connection, compile_id),
                    "quality": loads(row["quality_json"]),
                    "pace": loads(row["pace_json"]),
                    "readiness": run_store.compile_ready(connection, compile_id),
                    "queries": queries,
                }
            )
    return {"compiles": runs}


def _public(row: dict[str, Any]) -> dict[str, Any]:
    """去掉 ``*_json`` 原始列。需要的那些由调用方解开后单独加回。"""
    return {k: v for k, v in row.items() if not k.endswith("_json")}


@router.get("/compiles/{compile_id}/docs")
def compile_docs(
    request: Request,
    compile_id: int,
    dataset: str | None = None,
    gold_only: bool = False,
    limit: int = Query(DEFAULT_PAGE, le=MAX_PAGE),
    offset: int = 0,
) -> dict[str, Any]:
    """本次编译的语料，带 ``page_id``（走链路视图的钥匙）。

    没导入成功的 ``page_id`` 为空，它们照样列出 —— 缺篇本身是要看的信息。
    """
    with db(request) as connection:
        if run_store.get_compile_run(connection, compile_id) is None:
            raise HTTPException(404, f"编译 #{compile_id} 不存在")
        docs = run_store.compile_docs(connection, compile_id, dataset)
    if gold_only:
        docs = [d for d in docs if d["is_gold"]]
    return {
        "compile_id": compile_id,
        "total": len(docs),
        "imported": sum(1 for d in docs if d["page_id"]),
        "offset": offset,
        "limit": limit,
        "docs": docs[offset : offset + limit],
    }


@router.delete("/compiles/{compile_id}")
def delete_compile(request: Request, compile_id: int) -> dict[str, Any]:
    """清理一次编译及其下游的查询、评测、归因。远端 space 不删。"""
    with writable(request) as connection:
        row = run_store.get_compile_run(connection, compile_id)
        if row is None:
            raise HTTPException(404, f"编译 #{compile_id} 不存在")
        _reject_if_busy(connection, "compile", compile_id)
        removed = run_store.delete_compile_run(connection, compile_id)
    return {
        "deleted": removed,
        "space_id": row["space_id"],
        "note": "数据库内容已清理；Akasha 那边的空间没有删除，可自行处理。",
    }


# ------------------------------------------------------------------ 查询层


@router.get("/queries/{query_id}/responses")
def query_responses(
    request: Request,
    query_id: int,
    dataset: str | None = None,
    answer_mode: str | None = None,
    limit: int = Query(DEFAULT_PAGE, le=MAX_PAGE),
    offset: int = 0,
) -> dict[str, Any]:
    """查询结果列表，按 answerMode 另给一份计数。

    计数是必要的：``no_match`` / ``general`` 的检索得分按定义为 0，
    混在一起读会把生成端拒答误当成检索失败。
    """
    with db(request) as connection:
        if run_store.get_query_run(connection, query_id) is None:
            raise HTTPException(404, f"查询 #{query_id} 不存在")
        rows = run_store.responses_of(connection, query_id, dataset)

    counts: dict[str, int] = {}
    for row in rows:
        key = row["answer_mode"] or "missing"
        counts[key] = counts.get(key, 0) + 1
    if answer_mode:
        rows = [r for r in rows if (r["answer_mode"] or "missing") == answer_mode]

    return {
        "query_id": query_id,
        "total": len(rows),
        "count_by_answer_mode": counts,
        "offset": offset,
        "limit": limit,
        "responses": [
            {
                **{k: v for k, v in row.items() if k != "response"},
                "answer": ((row["response"] or {}).get("answer") or "")[:600]
                if isinstance(row["response"], dict)
                else None,
                "retrieved_count": len((row["response"] or {}).get("retrievedSources") or [])
                if isinstance(row["response"], dict)
                else 0,
                "citation_count": len((row["response"] or {}).get("citations") or [])
                if isinstance(row["response"], dict)
                else 0,
            }
            for row in rows[offset : offset + limit]
        ],
    }


@router.get("/queries/{query_id}/responses/{sample_id}")
def query_response(request: Request, query_id: int, sample_id: str) -> dict[str, Any]:
    """单条完整响应体。"""
    with db(request) as connection:
        row = run_store.response_of(connection, query_id, sample_id)
        if row is None:
            raise HTTPException(404, f"查询 #{query_id} 里没有 {sample_id!r} 的响应")
        return row


@router.delete("/queries/{query_id}")
def delete_query(request: Request, query_id: int) -> dict[str, Any]:
    """清理一次查询及其下游的评测、归因。"""
    with writable(request) as connection:
        if run_store.get_query_run(connection, query_id) is None:
            raise HTTPException(404, f"查询 #{query_id} 不存在")
        _reject_if_busy(connection, "query", query_id)
        removed = run_store.delete_query_run(connection, query_id)
    return {"deleted": removed}


def _reject_if_busy(connection, kind: str, target_id: int) -> None:
    """有任务正在往这条记录里写就拒绝清理。"""
    from akasha_benchmark.store import task_store

    for task in task_store.active_tasks(connection):
        if task["target_kind"] == kind and task["target_id"] == target_id:
            raise HTTPException(
                409, f"任务 #{task['id']} 正在写这条记录，请先暂停它再清理"
            )
