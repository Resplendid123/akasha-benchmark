"""启动白名单阶段子进程，由后台线程读取 stdout 并记录任务进度。

日志采集依赖 Web 进程；当前不保证后端重启后继续采集或托管任务。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from akasha_benchmark.store import connect, repo
from akasha_benchmark import progress

from .settings import Settings

# 阶段名 -> 模块。**白名单**：argv 由这里拼，不接受请求体里的任意命令。
STAGE_MODULES = {
    "verify": "akasha_platform.verify",
    "download": "akasha_platform.download",
    "normalize": "akasha_platform.prepare_normalize",
    "subset": "akasha_benchmark.subset",
    "ingest": "akasha_benchmark.ingest",
    "query": "akasha_benchmark.run_queries",
    "evaluate": "akasha_platform.evaluation",
    "audit": "akasha_benchmark.audit_join",
    "judge": "akasha_benchmark.judge.run",
    "reindex": "akasha_benchmark.store.reindex",
}

# 阶段自己打印的进度行，形如 "  hotpotqa: 120/400 (failures=0)"
# 或 "  hotpotqa: imported 25/400"。解析它就够了 —— 让阶段代码去写库会把
# 「跑批」和「伺候 UI」两件事缠在一起。
_PROGRESS = re.compile(r"(\w[\w-]*)\s*:\s*(?:imported\s+)?(\d+)\s*/\s*(\d+)")


class TaskRejected(RuntimeError):
    """请求的任务不合法（未知阶段，或同类任务已在跑）。"""


# 阶段参数白名单，写入 run_config 后由对应 CLI 读取。
STAGE_ARGS: dict[str, dict[str, type]] = {
    "verify": {"dataset": str, "samples": int},
    "download": {},
    "normalize": {"datasets": list, "export": bool},
    "subset": {
        "label": str,
        "datasets": list,
        "seed": int,
        "qa_limit": int,
        "negatives_ratio": float,
        "narrativeqa_docs": int,
        "export": bool,
    },
    "ingest": {"label": str, "datasets": list, "skip_compile": bool},
    "query": {
        "label": str,
        "query_label": str,
        "datasets": list,
        "score_threshold": float,
        "limit": int,
        "allow_config_drift": bool,
        "retry_failed": bool,
    },
    "evaluate": {
        "provider_label": str,
        "query_label": str,
        "eval_label": str,
        "datasets": list,
        "k": list,
        "metrics": list,
        "export": bool,
    },
    "audit": {"query_label": str, "eval_label": str, "datasets": list},
    "judge": {
        "eval_label": str,
        "datasets": list,
        "provider_label": str,
        "limit": int,
        "max_failure_rate": float,
    },
    "reindex": {"run_id": str, "label": str},
}


def _clean_args(stage: str, args: dict[str, Any]) -> dict[str, Any]:
    """按白名单过滤并转换类型。未声明的键直接丢掉。

    ``run_config.args_json`` 是通过 HTTP 写进来的，而阶段进程会把它当参数读。
    不过滤等于让请求体决定阶段代码看到什么 —— 那是个远程执行面。
    """
    allowed = STAGE_ARGS.get(stage)
    if allowed is None:
        raise TaskRejected(f"unknown stage {stage!r}; known: {sorted(STAGE_ARGS)}")

    cleaned: dict[str, Any] = {}
    for key, kind in allowed.items():
        if key not in args:
            continue
        value = args[key]
        if value is None or value == "":
            continue
        try:
            if kind is list:
                if not isinstance(value, list):
                    raise ValueError("expected a JSON array")
                cleaned[key] = [str(v) for v in value] if key != "k" else [int(v) for v in value]
            elif kind is bool:
                if not isinstance(value, bool):
                    raise ValueError("expected a JSON boolean")
                cleaned[key] = value
            else:
                cleaned[key] = kind(value)
        except (TypeError, ValueError) as exc:
            raise TaskRejected(f"{stage}.{key}: expected {kind.__name__}, got {value!r}") from exc
    return cleaned


def _argv(stage: str, run_config_id: int, settings: Settings) -> list[str]:
    """拼出 argv。**只从白名单取模块名，参数走库不走命令行。**

    argv 的长度因此是固定的，不随参数个数增长。
    """
    module = STAGE_MODULES.get(stage)
    if module is None:
        raise TaskRejected(f"unknown stage {stage!r}; known: {sorted(STAGE_MODULES)}")
    return [
        sys.executable,
        "-m",
        module,
        "--db",
        str(settings.db_path),
        "--run-config",
        str(int(run_config_id)),
    ]


def _pump(task_id: int, process: subprocess.Popen[str], settings: Settings, log_path: Path) -> None:
    """读子进程 stdout，写日志文件与库。**逐行提交**，Web 端才看得到进度。"""
    connection = connect(settings.db_path)
    try:
        structured_progress = False
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip("\n")
                event = progress.parse(line)
                if event:
                    line = f"[进度] {event['note']}"
                log.write(line + "\n")
                log.flush()
                repo.add_task_event(connection, task_id, "info", line[:2000])
                match = _PROGRESS.search(line)
                if event:
                    structured_progress = True
                    repo.update_task_progress(
                        connection, task_id,
                        done=min(event['done'], event['total'] - 1),
                        total=event['total'], note=event['note'],
                    )
                elif match and not structured_progress:
                    repo.update_task_progress(
                        connection,
                        task_id,
                        done=int(match.group(2)),
                        total=int(match.group(3)),
                        note=match.group(1),
                    )
                # 每行都提交：憋着的话 15 小时里 Web 端看到的是一个空任务。
                connection.commit()

        code = process.wait()
        repo.finish_task(
            connection,
            task_id,
            status="succeeded" if code == 0 else "failed",
            exit_code=code,
            error=None if code == 0 else f"exit code {code}; see {log_path.name}",
        )
        connection.commit()
    finally:
        connection.close()


def start(stage: str, args: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """起一个阶段任务，立刻返回。返回库里那条 task 记录。"""
    connection = connect(settings.db_path)
    try:
        # 数据准备会改写阶段输入，小样本验证跨多个阶段；两者都独占执行。
        running = [
            t for t in repo.running_tasks(connection)
            if t["stage"] == stage or stage in {"verify", "download", "normalize"}
            or t["stage"] in {"verify", "download", "normalize"}
        ]
        if running:
            raise TaskRejected(
                f"a {running[0]['stage']} task is already running (task #{running[0]['id']}). "
                "Data preparation and verification run exclusively; wait for the active task."
            )

        cleaned = _clean_args(stage, args)
        if stage == "verify":
            from .verify import DATASETS
            from akasha_benchmark.config import load_config

            if cleaned.get("dataset", "hotpotqa") not in DATASETS:
                raise TaskRejected("小样本验证仅支持 hotpotqa、2wikimultihopqa、musique")
            if not 1 <= cleaned.get("samples", 3) <= 5:
                raise TaskRejected("验证样本数必须在 1–5 之间")
            if repo.get_dataset(connection, cleaned.get("dataset", "hotpotqa")) is None:
                raise TaskRejected("请先归一化所选数据集")
            try:
                load_config(connection).require_credentials()
            except ValueError as exc:
                raise TaskRejected(str(exc)) from exc
        run_config_id = repo.create_run_config(connection, stage, cleaned)
        argv = _argv(stage, run_config_id, settings)
        task_id = repo.create_task(
            connection,
            stage=stage,
            argv=argv,
            index_layer_id=args.get("index_layer_id"),
            query_layer_id=args.get("query_layer_id"),
            eval_layer_id=args.get("eval_layer_id"),
        )
        connection.commit()

        log_path = settings.log_dir / f"task-{task_id}-{stage}.log"
        environment = {
            **os.environ,
            # 子进程的 stdout 必须是 UTF-8：Windows 上默认跟随控制台（实测 gbk），
            # 中文日志会变成乱码写进库。
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
        process = subprocess.Popen(  # noqa: S603 - argv 来自白名单，逐项显式映射
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            cwd=str(Path(settings.db_path).resolve().parent),
        )
        repo.start_task(connection, task_id, pid=process.pid, log_path=str(log_path))
        connection.commit()

        thread = threading.Thread(
            target=_pump, args=(task_id, process, settings, log_path), daemon=True
        )
        thread.start()
        return repo.get_task(connection, task_id) or {}
    finally:
        connection.close()


def cleanup(task_id: int | None, settings: Settings) -> dict[str, Any]:
    """删任务记录与日志文件。``task_id`` 为 None 时清掉所有已终态的任务。

    在跑的任务不删 —— 那会留下一个没人认领的子进程，而它还在往库里写。
    要停先 :func:`cancel`。
    """
    connection = connect(settings.db_path)
    try:
        targets = (
            [repo.get_task(connection, task_id)]
            if task_id is not None
            else [t for t in repo.list_tasks(connection, limit=10000)
                  if t["status"] not in {"queued", "running"}]
        )
        if task_id is not None and targets[0] is None:
            raise TaskRejected(f"no task #{task_id}")

        logs_removed = 0
        for task in targets:
            if not task:
                continue
            log_path = task.get("log_path")
            if log_path:
                try:
                    Path(log_path).unlink(missing_ok=True)
                    logs_removed += 1
                except OSError:
                    # 日志删不掉不该让整个清理失败：库里的记录才是要清的东西。
                    pass

        try:
            removed = (
                repo.delete_task(connection, task_id)
                if task_id is not None
                else repo.delete_finished_tasks(connection)
            )
        except ValueError as exc:
            raise TaskRejected(str(exc)) from exc
        connection.commit()
        return {"deleted": removed, "logs_removed": logs_removed}
    finally:
        connection.close()


def cancel(task_id: int, settings: Settings) -> dict[str, Any]:
    """终止一个在跑的任务。

    注意这只杀客户端进程 —— 编译在 Akasha 的 BullMQ worker 里，它不会因此停。
    所以取消 ingest 的语义是「不再盯着了」，不是「编译取消了」。
    """
    connection = connect(settings.db_path)
    try:
        task = repo.get_task(connection, task_id)
        if task is None:
            raise TaskRejected(f"no task #{task_id}")
        if task["status"] not in {"queued", "running"}:
            raise TaskRejected(f"task #{task_id} is already {task['status']}")
        pid = task["pid"]
        if pid:
            try:
                os.kill(int(pid), 9)
            except (ProcessLookupError, PermissionError, OSError) as exc:
                repo.add_task_event(connection, task_id, "warn", f"kill failed: {exc}")
        repo.finish_task(
            connection,
            task_id,
            status="cancelled",
            exit_code=None,
            error="cancelled by user; server-side compilation is unaffected",
        )
        connection.commit()
        return repo.get_task(connection, task_id) or {}
    finally:
        connection.close()
