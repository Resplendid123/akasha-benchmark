"""任务运行器：在后台线程里跑阶段，支持暂停、继续、清理。

阶段在本进程里跑，参数与产物都在库里。后端重启会中断在跑的任务，
启动时 :meth:`TaskRunner.recover` 把它们标成暂停，让用户显式继续。

暂停是协作式的：阶段在每个可续跑的边界调 ``ctx.checkpoint()``。
继续使用同一任务保存的参数与产物 ID，各阶段按自身规则重算或跳过。
"""

from __future__ import annotations

import threading
from typing import Any

from akasha_benchmark.akasha_client import ACTIVE_RUN_STATUSES, AkashaClient, AkashaError
from akasha_benchmark.config import load_config
from akasha_benchmark.stages import STAGES, chain, clean_params
from akasha_benchmark.store import compile_store, connect, task_store
from akasha_benchmark.task import Paused, TaskContext, execute

from .settings import Settings

# 这些阶段与任何在跑的任务互斥：它们改的是下游所有层的输入。
EXCLUSIVE = frozenset({"download", "normalize"})
# 编译产物与远端 Space 按任务隔离，允许有限并行；其他同名阶段仍串行。
STAGE_CONCURRENCY = {"compile": 3}


class TaskRejected(RuntimeError):
    """请求的任务不合法（未知阶段、参数不对，或同类任务已在跑）。"""


class TaskRunner:
    """在跑的任务。一个 stage 一条线程，暂停信号逐任务持有。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pauses: dict[int, threading.Event] = {}
        self._lock = threading.Lock()

    # --- 启动与恢复 ---

    def recover(self) -> int:
        """把上次进程留下的「运行中」标成暂停，返回处理了几条。

        那些线程已经没了，留着 running 会让记录卡住（清理不许删在跑的任务）。
        """
        connection = connect(self.settings.db_path)
        try:
            active = task_store.active_tasks(connection)
            for task in active:
                task_store.transition(connection, int(task["id"]), task_store.PAUSED)
                task_store.log(
                    connection,
                    task_id=int(task["id"]),
                    stage=task["stage"],
                    level="warn",
                    message="后端重启，任务已标为暂停；点击继续可接着跑",
                )
            connection.commit()
            return len(active)
        finally:
            connection.close()

    def start(self, stage: str, args: dict[str, Any]) -> dict[str, Any]:
        """建一条任务并起线程，立刻返回那条记录。"""
        if stage not in STAGES:
            raise TaskRejected(f"未知阶段 {stage!r}；可用：{sorted(STAGES)}")
        try:
            params = clean_params(stage, args)
        except ValueError as exc:
            raise TaskRejected(str(exc)) from exc

        connection = connect(self.settings.db_path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._require_free(connection, stage)
            task_id = task_store.create_task(connection, stage=stage, params=params)
            connection.commit()
            record = task_store.get_task(connection, task_id) or {}
        finally:
            connection.close()

        self._spawn(task_id, stage, params)
        return record

    def start_chain(self, args: dict[str, Any]) -> dict[str, Any]:
        """起一条链路测试：建链首那条编译任务，余下几步挂在它上面。

        返回链首任务；后面三条在前一条成功时自动出现。
        """
        connection = connect(self.settings.db_path)
        try:
            try:
                steps = chain.build(args, connection)
            except ValueError as exc:
                raise TaskRejected(str(exc)) from exc
            head, rest = steps[0], steps[1:]
            params = clean_params(head["stage"], head["params"])
            connection.execute("BEGIN IMMEDIATE")
            self._require_free(connection, head["stage"])
            task_id = task_store.create_task(connection, stage=head["stage"], params=params)
            # 链首的 id 就是链号，四条任务凭它归到一起。
            task_store.set_task_chain(connection, task_id, chain=rest, chain_id=task_id)
            connection.commit()
            record = task_store.get_task(connection, task_id) or {}
        finally:
            connection.close()

        self._spawn(task_id, head["stage"], params)
        return record

    def resume(self, task_id: int) -> dict[str, Any]:
        """继续一个暂停的任务：同一条记录、同一组参数，重新起线程。"""
        connection = connect(self.settings.db_path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            task = task_store.get_task(connection, task_id)
            if task is None:
                raise TaskRejected(f"任务 #{task_id} 不存在")
            if task["status"] not in (task_store.PAUSED, task_store.FAILED):
                raise TaskRejected(f"任务 #{task_id} 当前是 {task['status']}，无需继续")
            self._require_free(connection, task["stage"])
            task_store.transition(connection, task_id, task_store.QUEUED)
            connection.commit()
        finally:
            connection.close()

        self._spawn(task_id, task["stage"], task["params"])
        connection = connect(self.settings.db_path)
        try:
            return task_store.get_task(connection, task_id) or {}
        finally:
            connection.close()

    def pause(self, task_id: int) -> dict[str, Any]:
        """请求暂停。阶段跑到下一个 checkpoint 时停下并落库。"""
        connection = connect(self.settings.db_path)
        try:
            task = task_store.get_task(connection, task_id)
            if task is None:
                raise TaskRejected(f"任务 #{task_id} 不存在")
            if task["status"] not in task_store.ACTIVE:
                raise TaskRejected(f"任务 #{task_id} 当前是 {task['status']}，不在运行")
            self._cancel_remote_compile(connection, task, "Akasha-Benchmark task paused")
            with self._lock:
                event = self._pauses.get(task_id)
            if event is None:
                # 没有对应线程时同步暂停任务及产物。
                task_store.transition(connection, task_id, task_store.PAUSED)
                connection.commit()
            else:
                event.set()
                task_store.log(
                    connection,
                    task_id=task_id,
                    stage=task["stage"],
                    level="info",
                    message="已请求暂停，等待当前步骤收尾",
                )
                connection.commit()
            return task_store.get_task(connection, task_id) or {}
        finally:
            connection.close()

    def cleanup(self, task_id: int | None) -> dict[str, int]:
        """删任务记录，``None`` 时清掉所有非运行中的。审计日志不删。"""
        connection = connect(self.settings.db_path)
        try:
            try:
                targets = (
                    [task_store.get_task(connection, task_id)]
                    if task_id is not None
                    else task_store.list_tasks(connection, limit=500)
                )
                for task in targets:
                    if task and task["status"] not in task_store.ACTIVE:
                        self._cancel_remote_compile(
                            connection, task, "Akasha-Benchmark task cleaned up"
                        )
                deleted = (
                    task_store.delete_task(connection, task_id)
                    if task_id is not None
                    else task_store.delete_inactive_tasks(connection)
                )
            except ValueError as exc:
                raise TaskRejected(str(exc)) from exc
            connection.commit()
            return {"deleted": deleted}
        finally:
            connection.close()

    # --- 内部 ---

    def _cancel_remote_compile(
        self, connection, task: dict[str, Any], reason: str
    ) -> None:
        if task.get("stage") != "compile":
            return
        run_ids = task.get("params", {}).get("remote_compile_run_ids") or []
        try:
            with AkashaClient(load_config(connection)) as client:
                client.login()
                if not run_ids and task.get("target_kind") == "compile":
                    compile_run = compile_store.get_compile_run(
                        connection, int(task.get("target_id") or 0)
                    )
                    space_id = (compile_run or {}).get("space_id")
                    if space_id:
                        diagnostics = client.run_diagnostics([space_id], limit=50)
                        run_ids = [
                            str(run["runId"])
                            for run in diagnostics.get("items") or []
                            if run.get("runId")
                            and str(run.get("status")) in ACTIVE_RUN_STATUSES
                        ]
                for run_id in run_ids:
                    result = client.cancel_compile_run(str(run_id), reason)
                    task_store.log(
                        connection,
                        task_id=int(task["id"]),
                        stage="compile",
                        level="info",
                        message=(
                            f"远端编译 Run {run_id} 已处理：{result.get('disposition', 'unknown')}，"
                            f"清理 BullMQ job {result.get('removedJobCount', 0)} 个"
                        ),
                    )
            connection.commit()
        except (AkashaError, ValueError) as exc:
            connection.rollback()
            raise TaskRejected(f"远端编译 Run 取消失败，本地状态未变：{exc}") from exc

    def _require_free(self, connection, stage: str) -> None:
        active = task_store.active_tasks(connection)
        for task in active:
            if stage in EXCLUSIVE or task["stage"] in EXCLUSIVE:
                raise TaskRejected(
                    f"{task['stage']} 任务 #{task['id']} 正在运行，请先等它结束或暂停"
                )

        same_stage = [task for task in active if task["stage"] == stage]
        limit = STAGE_CONCURRENCY.get(stage, 1)
        if len(same_stage) >= limit:
            if limit == 1:
                task = same_stage[0]
                raise TaskRejected(
                    f"{task['stage']} 任务 #{task['id']} 正在运行，请先等它结束或暂停"
                )
            raise TaskRejected(
                f"{stage} 任务已达到并发上限 {limit}，请先等其中一个结束或暂停"
            )

    def _verify(self, connection, task_id: int, stage: str) -> None:
        """校验链上任务的产物契约，抛出的异常按失败处理。

        只对链上的任务生效，手动起的单阶段任务不受约束。
        """
        chain_id, _ = task_store.task_chain(connection, task_id)
        check = chain.VERIFY.get(stage)
        if chain_id is None or check is None:
            return
        task = task_store.get_task(connection, task_id) or {}
        target_id = task.get("target_id")
        if target_id is None:
            raise RuntimeError(f"{stage} 没有产出可校验的记录")
        check(connection, int(target_id))
        task_store.log(
            connection,
            task_id=task_id,
            stage=stage,
            level="info",
            message="链路契约：通过",
        )
        connection.commit()

    def _advance_chain(self, connection, task_id: int, stage: str) -> None:
        """接上链的下一步，把这一步的产物 id 填进它的 ``link`` 参数。"""
        chain_id, remaining = task_store.task_chain(connection, task_id)
        if chain_id is None or not remaining:
            return
        task = task_store.get_task(connection, task_id) or {}
        target_id = task.get("target_id")
        if target_id is None:
            task_store.log(
                connection,
                task_id=task_id,
                stage=stage,
                level="error",
                message="没有产物记录，链路中断",
            )
            connection.commit()
            return

        step, rest = remaining[0], remaining[1:]
        params = {k: v for k, v in step["params"].items() if v is not None}
        if link := step.get("link"):
            params[link] = int(target_id)
        try:
            params = clean_params(step["stage"], params)
            connection.execute("BEGIN IMMEDIATE")
            self._require_free(connection, step["stage"])
        except (ValueError, TaskRejected) as exc:
            task_store.log(
                connection,
                task_id=task_id,
                stage=stage,
                level="error",
                message=f"链路中断，下一步 {step['stage']} 起不来：{exc}",
            )
            connection.commit()
            return

        next_id = task_store.create_task(connection, stage=step["stage"], params=params)
        task_store.set_task_chain(connection, next_id, chain=rest, chain_id=chain_id)
        task_store.log(
            connection,
            task_id=task_id,
            stage=stage,
            level="info",
            message=f"链路继续：{STAGES[step['stage']].label} #{next_id}",
        )
        connection.commit()
        self._spawn(next_id, step["stage"], params)

    def _spawn(self, task_id: int, stage: str, params: dict[str, Any]) -> None:
        event = threading.Event()
        with self._lock:
            self._pauses[task_id] = event
        threading.Thread(
            target=self._run, args=(task_id, stage, params, event), daemon=True
        ).start()

    def _run(
        self, task_id: int, stage: str, params: dict[str, Any], event: threading.Event
    ) -> None:
        # 每个任务线程一条独立连接，sqlite 连接不跨线程共用。
        connection = connect(self.settings.db_path)
        try:
            task = task_store.get_task(connection, task_id)
            if task is None or task["status"] != task_store.QUEUED:
                return
            ctx = TaskContext(
                task_id=task_id,
                stage=stage,
                params=params,
                connection=connection,
                pause_event=event,
            )
            ctx.log(f"开始 {STAGES[stage].label}：{params}")
            try:
                execute(STAGES[stage].run, ctx, lambda: self._verify(connection, task_id, stage))
            except Paused as exc:
                task_store.log(
                    connection,
                    task_id=task_id,
                    stage=stage,
                    level="warn",
                    message=f"已暂停：{exc}",
                )
                connection.commit()
                return
            except Exception as exc:  # noqa: BLE001 - 失败要落库，不能只留在线程里
                task_store.log(
                    connection,
                    task_id=task_id,
                    stage=stage,
                    level="error",
                    message=f"{type(exc).__name__}: {exc}",
                )
                connection.commit()
                return
            task_store.log(
                connection, task_id=task_id, stage=stage, level="info", message="任务完成"
            )
            connection.commit()
            self._advance_chain(connection, task_id, stage)
        finally:
            connection.close()
            with self._lock:
                if self._pauses.get(task_id) is event:
                    self._pauses.pop(task_id, None)
