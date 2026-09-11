"""阶段参数：从库里的 ``run_config`` 读，而不是从命令行拼。

平台起任务时先写一行 ``run_config``，argv 里只剩
``python -m <module> --db <path> --run-config <id>``。

这么改的直接理由是 argv 不再随参数个数增长。原先 14 个参数在 ``tasks.py``
里逐项映射成命令行标志，每加一个旋钮要同时改那张映射表和阶段的 argparse ——
两处漂了就会出现「UI 上改了但跑的还是默认值」，而那不报错。

命令行入口保留，本地调试还用得上；``--run-config`` 存在时它的值覆盖命令行。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


class RunConfigMissing(RuntimeError):
    """给的 run_config id 在库里不存在。"""


def load(run_config_id: int | None, db_path: Path | None) -> dict[str, Any]:
    """读一份运行参数。``run_config_id`` 为空时返回空字典。"""
    if run_config_id is None:
        return {}

    from .store import connect, repo

    connection = connect(db_path, read_only=True)
    try:
        record = repo.get_run_config(connection, run_config_id)
    finally:
        connection.close()
    if record is None:
        raise RunConfigMissing(f"no run_config #{run_config_id}")
    return dict(record["args"])


def apply(args: argparse.Namespace, *, stage: str) -> argparse.Namespace:
    """把库里的运行参数盖到 argparse 的结果上。

    只覆盖 parser 已声明的属性 —— 库里多出来的键静默忽略。这一条是安全边界:
    ``run_config.args_json`` 是通过 HTTP 写进来的，允许它设任意属性就等于
    让请求体决定阶段代码读到什么。

    ``datasets``（复数）是特例：阶段的 argparse 用 ``--dataset`` 追加成
    ``args.dataset``，而 UI 那边是一个数组，名字对不上就会静默跑全量。
    """
    values = load(getattr(args, "run_config", None), getattr(args, "db", None))
    if not values:
        return args

    if "datasets" in values and hasattr(args, "dataset"):
        args.dataset = list(values["datasets"]) or None

    for key, value in values.items():
        if key == "datasets":
            continue
        if not hasattr(args, key):
            continue
        setattr(args, key, value)
    args.stage = stage
    return args


def add_argument(parser: argparse.ArgumentParser) -> None:
    """给阶段的 parser 加 ``--run-config``。"""
    parser.add_argument(
        "--run-config",
        type=int,
        default=None,
        help="从库里的 run_config 读参数（平台起任务时用）。它的值覆盖命令行",
    )


def require(value: Any, name: str, stage: str) -> Any:
    """必填项的校验。

    这些项原先靠 argparse 的 ``required=True`` 保证。加了 ``--run-config``
    之后不能再那样声明 —— 参数可能来自库里，argparse 会在读库之前就拒绝。
    所以改成合并之后再校验，报错信息里同时给出两条来路。
    """
    if value in (None, ""):
        raise ValueError(
            f"{stage}: {name} is required. Pass --{name.replace('_', '-')} "
            "or start this stage from the platform (which writes a run_config)."
        )
    return value
