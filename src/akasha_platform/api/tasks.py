"""任务层：起、停、清理，以及增量日志。

阶段任务走 subprocess（决策 7）：**15 小时的 ingest 不能与 Web 后端同生命周期。**
后端重启、崩掉、被 Ctrl-C，那个任务都该继续跑。

「暂停」做成可续跑的**停止**：Windows 上没有 SIGSTOP，而 ingest 与 query 本来
就跳过已完成项，所以「停掉再起」与「暂停再继续」在效果上等价 —— 而前者不需要
假装持有一个挂起的进程。取消 ingest 的语义是「不再盯着了」，编译在 Akasha 的
BullMQ worker 里，不会因此停止。
"""

from __future__ import annotations

from typing import Any

from akasha_benchmark.store import repo
from fastapi import APIRouter, Body, HTTPException, Query, Request

from .. import tasks as tasks_mod
from ._common import db, settings_of

router = APIRouter(prefix="/api")


@router.get("/tasks")
def list_tasks(
    request: Request, status: str | None = None, limit: int = Query(50, le=500)
) -> list[dict[str, Any]]:
    with db(request) as connection:
        tasks = repo.list_tasks(connection, status=status, limit=limit)
        return [
            {
                **{k: v for k, v in task.items() if k != "argv_json"},
                "argv": repo.loads(task["argv_json"], []),
                # argv 里只剩一个 run_config id，参数本身在库里 —— 一起给出来,
                # 否则任务列表看不出这一轮跑的是什么配置。
                "args": _args_of(connection, repo.loads(task["argv_json"], [])),
            }
            for task in tasks
        ]


def _args_of(connection, argv: list[str]) -> dict[str, Any]:
    """从 argv 里的 ``--run-config <id>`` 反查这一轮的参数。"""
    try:
        index = argv.index("--run-config")
        run_config_id = int(argv[index + 1])
    except (ValueError, IndexError):
        return {}
    record = repo.get_run_config(connection, run_config_id)
    return (record or {}).get("args") or {}


@router.get("/tasks/{task_id}")
def task_detail(
    request: Request, task_id: int, after_id: int = 0
) -> dict[str, Any]:
    """任务详情与增量日志。``after_id`` 让前端只拉新增的事件。"""
    with db(request) as connection:
        task = repo.get_task(connection, task_id)
        if task is None:
            raise HTTPException(404, f"no task #{task_id}")
        argv = repo.loads(task["argv_json"], [])
        return {
            **{k: v for k, v in task.items() if k != "argv_json"},
            "argv": argv,
            "args": _args_of(connection, argv),
            "events": repo.task_events(connection, task_id, after_id=after_id),
        }


@router.post("/tasks/{stage}")
def start_task(
    request: Request, stage: str, args: dict[str, Any] = Body(default={})
) -> dict[str, Any]:
    """起一个阶段任务。

    参数写进库里的 ``run_config``，argv 只带它的 id。白名单在
    :data:`..tasks.STAGE_ARGS` 里 —— 请求体里的其他键不会进库，
    阶段代码也就读不到它们。
    """
    try:
        return tasks_mod.start(stage, args, settings_of(request))
    except tasks_mod.TaskRejected as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/tasks/{task_id}/cancel")
def cancel_task(request: Request, task_id: int) -> dict[str, Any]:
    """停掉一个在跑的任务。**可续跑** —— 重新起同一阶段会接着上次的进度。"""
    try:
        return tasks_mod.cancel(task_id, settings_of(request))
    except tasks_mod.TaskRejected as exc:
        raise HTTPException(409, str(exc)) from exc


@router.delete("/tasks/{task_id}")
def cleanup_task(request: Request, task_id: int) -> dict[str, Any]:
    """删一条任务记录与它的日志。在跑的任务不许删 —— 先 cancel。"""
    try:
        return tasks_mod.cleanup(task_id, settings_of(request))
    except tasks_mod.TaskRejected as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/tasks/cleanup/finished")
def cleanup_finished(request: Request) -> dict[str, Any]:
    """清掉所有已终态的任务。在跑的不动。"""
    try:
        return tasks_mod.cleanup(None, settings_of(request))
    except tasks_mod.TaskRejected as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/stages")
def stages() -> list[dict[str, Any]]:
    """各阶段接受哪些参数，以及它的代价。前端的任务表单据此生成。

    代价写在这里是因为它决定了「这个按钮该不该二次确认」：ingest 是 15 小时,
    judge 会花钱，其余是秒级或分钟级。
    """
    return [
        {
            "stage": "normalize",
            "label": "归一化",
            "args": sorted(tasks_mod.STAGE_ARGS["normalize"]),
            "cost": "秒级，离线",
            "needs_akasha": False,
        },
        {
            "stage": "subset",
            "label": "抽子集",
            "args": sorted(tasks_mod.STAGE_ARGS["subset"]),
            "cost": "秒级，离线",
            "needs_akasha": False,
        },
        {
            "stage": "ingest",
            "label": "入库编译",
            "args": sorted(tasks_mod.STAGE_ARGS["ingest"]),
            "cost": "约 40 秒/篇（Akasha 的 BullMQ worker 吞吐，客户端调不动）",
            "needs_akasha": True,
        },
        {
            "stage": "query",
            "label": "跑查询",
            "args": sorted(tasks_mod.STAGE_ARGS["query"]),
            "cost": "10-14 秒每条",
            "needs_akasha": True,
        },
        {
            "stage": "evaluate",
            "label": "算指标",
            "args": sorted(tasks_mod.STAGE_ARGS["evaluate"]),
            "cost": "秒级，纯离线",
            "needs_akasha": False,
        },
        {
            "stage": "judge",
            "label": "judge 指标",
            "args": sorted(tasks_mod.STAGE_ARGS["judge"]),
            "cost": "会调模型花钱；失败率超阈值整轮判失败",
            "needs_akasha": False,
        },
        {
            "stage": "audit",
            "label": "审计归因",
            "args": sorted(tasks_mod.STAGE_ARGS["audit"]),
            "cost": "秒级，需要只读 Postgres",
            "needs_akasha": False,
        },
    ]
