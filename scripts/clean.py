"""删产物。给 Makefile 的 clean-data / distclean 用。

存在的理由是跨平台：Makefile 要能从 PowerShell 跑，而那里 ``SHELL := /bin/sh``
不生效 —— GNU Make 找不到 ``/bin/sh`` 时回落到 ``cmd.exe``，于是 ``rm -rf``
和 ``test`` 都不存在。Python 本来就是这个项目的工具链。

**两条永不删的东西**：

* ``akasha_bench.db`` —— 它是事实来源，不是缓存。
  删它要显式来，不能被一句 ``make distclean`` 顺手带走。
* ``dataset/`` —— 重下要 137MB。

    uv run python scripts/clean.py exports --label run002
    uv run python scripts/clean.py all
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# distclean 的范围：导出、缓存、前端构建产物与依赖。
DISPOSABLE = (
    "data",
    ".pytest_cache",
    ".ruff_cache",
    "web/dist",
    "web/node_modules",
    "logs",
)

# 无论如何都不碰。判据写在代码里，不是写在注释里。
PROTECTED = ("akasha_bench.db", "dataset")


def _remove(relative: str) -> bool:
    """删一个仓库内的相对路径。返回是否真的删了东西。"""
    if relative in PROTECTED or relative.startswith(tuple(f"{p}/" for p in PROTECTED)):
        raise ValueError(f"refusing to delete protected path {relative!r}")
    target = (REPO_ROOT / relative).resolve()
    # 越界检查：relative 来自代码里的常量与 --label，后者是用户输入。
    if not target.is_relative_to(REPO_ROOT):
        raise ValueError(f"{relative!r} escapes the repository root")
    if target.is_dir():
        shutil.rmtree(target)
        return True
    if target.exists():
        target.unlink()
        return True
    return False


def clean_exports(label: str, eval_label: str) -> int:
    """删某一层的文件导出。库里的那一层不动。"""
    removed = 0
    for relative in (
        f"data/subsets/{label}",
        f"data/ingest/{label}",
        f"data/responses/{label}",
        f"data/reports/{eval_label}",
    ):
        if _remove(relative):
            print(f"removed {relative}")
            removed += 1
    if not removed:
        print(f"nothing to remove for label {label!r}")
    print("kept akasha_bench.db and dataset/ — the database is the source of truth")
    return 0


def clean_all() -> int:
    for relative in DISPOSABLE:
        if _remove(relative):
            print(f"removed {relative}")
    print("kept akasha_bench.db and dataset/ (137MB to re-download); delete those by hand")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=("exports", "all"))
    parser.add_argument("--label", default="run002")
    parser.add_argument("--eval-label", default=None)
    args = parser.parse_args(argv)

    try:
        if args.what == "exports":
            return clean_exports(args.label, args.eval_label or f"{args.label}-query-eval")
        return clean_all()
    except (ValueError, OSError) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
