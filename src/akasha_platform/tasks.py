"""任务运行器：在后台线程里跑阶段，支持暂停、继续、清理。

阶段在**本进程**里跑（不再起子进程），因为参数与产物都在库里，
而暂停需要一个能被阶段看见的信号 —— 跨进程做这件事要额外一套 IPC，
而它换不来别的好处。代价是后端重启会中断在跑的任务：启动时把它们标成暂停,
让用户显式继续，而不是留一个状态是「运行中」但其实没人在跑的记录。

「暂停」是协作式的：阶段在每个可续跑的边界调 ``ctx.checkpoint()``，
所以暂停总是停在一个已落库的位置上。继续 = 用同一条任务记录重跑，
阶段自己跳过已完成的部分。
"""

from __future__ import annotations

import threading
from typing import Any

from akasha_benchmark.stages import STAGES, chain, clean_params
from akasha_benchmark.store import connect, task_store
from akasha_benchmark.task import Paused, TaskContext

from .settings import Settings

# 同一阶段不并行：两个 compile 同时写同一批表只会互相覆盖。
# 数据准备类阶段之间也互斥 —— 它们改的是下游所有层的输入。
EXCLUSIVE = frozenset({"download", "normalize"})


class TaskRejected(RuntimeError):
    """请求的任务不合法（未知阶段、参数不对，或同类任务已在跑）。"""


class TaskRunner:
    """在跑的任务。一个 stage 一条线程，暂停信号逐任务持有。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pauses: dict[int, threading.Event] = {}
        self._threads: dict[int, threading.Thread] = {}
        self._lock = threading.Lock()

    # --- 启动与恢复 ---

    def recover(self) -> int:
        """把上次进程留下的「运行中」标成暂停。返回处理了几条。

        重启之后那些线程已经没了，留着 running 会让界面显示一个不存在的任务,
        而清理又不许删在跑的任务 —— 于是那条记录卡住。
        """
        connection = connect(self.settings.db_path)
        try:
            active = task_store.active_tasks(connection)
            for task in active:
                task_store.pause_task(connection, int(task["id"]))
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

        返回链首任务，前端据此跳到任务列表 —— 后面三条会在前一条成功时自动出现。
        """
        connection = connect(self.settings.db_path)
        try:
            try:
                steps = chain.build(args, connection)
            except ValueError as exc:
                raise TaskRejected(str(exc)) from exc
            head, rest = steps[0], steps[1:]
            params = clean_params(head["stage"], head["params"])
            self._require_free(connection, head["stage"])
            task_id = task_store.create_task(connection, stage=head["stage"], params=params)
            # 链首自己就是链号，四条任务凭它归到一起。
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
            task = task_store.get_task(connection, task_id)
            if task is None:
                raise TaskRejected(f"任务 #{task_id} 不存在")
            if task["status"] not in (task_store.PAUSED, task_store.FAILED):
                raise TaskRejected(f"任务 #{task_id} 当前是 {task['status']}，无需继续")
            self._require_free(connection, task["stage"])
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
            with self._lock:
                event = self._pauses.get(task_id)
            if event is None:
                # 没有对应线程（比如后端重启过），直接改状态。
                task_store.pause_task(connection, task_id)
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
        """删任务记录。``None`` 时清掉所有非运行中的任务。

        **审计日志不删** —— 它是审计记录，任务记录清掉之后仍然查得到。
        """
        connection = connect(self.settings.db_path)
        try:
            try:
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

    def _require_free(self, connection, stage: str) -> None:
        for task in task_store.active_tasks(connection):
            if task["stage"] == stage or (
                stage in EXCLUSIVE or task["stage"] in EXCLUSIVE
            ):
                raise TaskRejected(
                    f"{task['stage']} 任务 #{task['id']} 正在运行，请先等它结束或暂停"
                )

    def _verify(self, connection, task_id: int, stage: str) -> None:
        """链上的任务要过契约校验才算成功。抛出的异常按失败处理。

        只对链上的任务生效：手动起的单阶段任务不受这套断言约束。
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
        """成功之后接上链的下一步：上一步的产物 id 填进它的关联参数。"""
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
        thread = threading.Thread(
            target=self._run, args=(task_id, stage, params, event), daemon=True
        )
        with self._lock:
            self._threads[task_id] = thread
        thread.start()

    def _run(
        self, task_id: int, stage: str, params: dict[str, Any], event: threading.Event
    ) -> None:
        # 每个任务线程一条独立连接：sqlite 连接不跨线程共用。
        connection = connect(self.settings.db_path)
        try:
            task_store.start_task(connection, task_id)
            connection.commit()
            ctx = TaskContext(
                task_id=task_id,
                stage=stage,
                params=params,
                connection=connection,
                pause_event=event,
            )
            ctx.log(f"开始 {STAGES[stage].label}：{params}")
            try:
                STAGES[stage].run(ctx)
                self._verify(connection, task_id, stage)
            except Paused as exc:
                task_store.pause_task(connection, task_id)
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
                task_store.finish_task(
                    connection,
                    task_id,
                    status=task_store.FAILED,
                    error=f"{type(exc).__name__}: {exc}"[:2000],
                )
                task_store.log(
                    connection,
                    task_id=task_id,
                    stage=stage,
                    level="error",
                    message=f"{type(exc).__name__}: {exc}",
                )
                connection.commit()
                return
            task_store.finish_task(connection, task_id, status=task_store.SUCCEEDED)
            task_store.log(
                connection, task_id=task_id, stage=stage, level="info", message="任务完成"
            )
            connection.commit()
            self._advance_chain(connection, task_id, stage)
        finally:
            connection.close()
            with self._lock:
                self._pauses.pop(task_id, None)
                self._threads.pop(task_id, None)
