"""阶段与任务运行器之间的契约。

阶段拿到一个 :class:`TaskContext`，用它写日志、报进度、检查暂停。
暂停是协作式的：阶段在每个可续跑的边界调 :meth:`TaskContext.checkpoint`，
它在被请求暂停时抛 :class:`Paused`，所以暂停总停在已落库的位置上。
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Any, Protocol


class Paused(BaseException):
    """请求暂停时由 checkpoint 抛出。

    继承 BaseException 而不是 Exception，免得被阶段里的 ``except Exception`` 吞掉。
    """


class TaskContext:
    """一次任务运行的上下文。"""

    def __init__(
        self,
        *,
        task_id: int,
        stage: str,
        params: dict[str, Any],
        connection: sqlite3.Connection,
        pause_event: threading.Event,
    ) -> None:
        self.task_id = task_id
        self.stage = stage
        self.params = params
        self.db = connection
        self._pause = pause_event

    # --- 暂停 ---

    @property
    def pause_requested(self) -> bool:
        return self._pause.is_set()

    def checkpoint(self) -> None:
        """可续跑的边界。被请求暂停时抛 :class:`Paused`。"""
        if self._pause.is_set():
            raise Paused(f"任务 #{self.task_id} 已暂停")

    # --- 日志与进度 ---

    def log(self, message: str, level: str = "info") -> None:
        from .store import task_store

        task_store.log(
            self.db, task_id=self.task_id, stage=self.stage, level=level, message=message
        )
        self.db.commit()

    def progress(self, done: int, total: int | None, note: str | None = None) -> None:
        from .store import task_store

        task_store.update_progress(self.db, self.task_id, done=done, total=total, note=note)
        self.db.commit()

    def bind(self, kind: str, target_id: int) -> None:
        """记下这次任务产出属于哪一层的哪条记录，供层视图反查。"""
        from .store import task_store

        task_store.set_task_target(self.db, self.task_id, kind, target_id)
        self.db.commit()


class Stage(Protocol):
    """一个阶段。参数不合法时抛 ``ValueError``，任务因此记为失败。"""

    def __call__(self, ctx: TaskContext) -> None: ...
