"""任务层：起、暂停、继续、清理，以及增量日志。

暂停是协作式的，停下的位置总是已落库的；继续用同一条任务记录重跑。
注意暂停编译只是「不再往下走」，已提交给 Akasha 的编译不会因此停止。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from akasha_benchmark.stages import describe
from akasha_benchmark.store import task_store

from ..tasks import TaskRejected
from ._common import db, runner_of

router = APIRouter(prefix="/api")


@router.get("/stages")
def stages() -> list[dict[str, Any]]:
    """各阶段接受哪些参数，以及它的代价。前端的任务表单据此生成。"""
    return describe()


@router.get("/tasks")
def list_tasks(
    request: Request, status: str | None = None, limit: int = Query(50, le=500)
) -> list[dict[str, Any]]:
    with db(request) as connection:
        return task_store.list_tasks(connection, status=status, limit=limit)


@router.get("/tasks/{task_id}")
def task_detail(request: Request, task_id: int, after_id: int = 0) -> dict[str, Any]:
    """任务详情与增量日志。``after_id`` 让前端只拉新增的行。"""
    with db(request) as connection:
        task = task_store.get_task(connection, task_id)
        if task is None:
            raise HTTPException(404, f"任务 #{task_id} 不存在")
        return {**task, "logs": task_store.task_logs(connection, task_id, after_id=after_id)}


@router.post("/chain")
def start_chain(request: Request, args: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """起一条链路测试：编译到归因四条普通任务，前一条成功时自动接上后一条。"""
    try:
        return runner_of(request).start_chain(args)
    except TaskRejected as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/tasks/{stage}")
def start_task(
    request: Request, stage: str, args: dict[str, Any] = Body(default={})
) -> dict[str, Any]:
    """起一个阶段任务。参数按阶段声明过滤，未声明的键不会传给阶段代码。"""
    try:
        return runner_of(request).start(stage, args)
    except TaskRejected as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/tasks/{task_id}/pause")
def pause_task(request: Request, task_id: int) -> dict[str, Any]:
    try:
        return runner_of(request).pause(task_id)
    except TaskRejected as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/tasks/{task_id}/resume")
def resume_task(request: Request, task_id: int) -> dict[str, Any]:
    try:
        return runner_of(request).resume(task_id)
    except TaskRejected as exc:
        raise HTTPException(409, str(exc)) from exc


@router.delete("/tasks/{task_id}")
def cleanup_task(request: Request, task_id: int) -> dict[str, Any]:
    """删一条任务记录，审计日志保留。在跑的任务不许删，先暂停。"""
    try:
        return runner_of(request).cleanup(task_id)
    except TaskRejected as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/tasks/cleanup/inactive")
def cleanup_inactive(request: Request) -> dict[str, Any]:
    """清掉所有非运行中的任务记录。审计日志保留。"""
    try:
        return runner_of(request).cleanup(None)
    except TaskRejected as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/audit")
def audit(
    request: Request, stage: str | None = None, limit: int = Query(200, le=2000)
) -> list[dict[str, Any]]:
    """审计日志。只追加，清理任务不删它。"""
    with db(request) as connection:
        return task_store.audit_logs(connection, stage=stage, limit=limit)
