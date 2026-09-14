"""数据集层：从 Hugging Face 下载 HippoRAG_2 的四组数据，下载后校验。

来源：https://huggingface.co/datasets/osunlp/HippoRAG_2
已存在且字节数符合预期的文件跳过，所以中断后继续即可。
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from ..datasets import DATASET_NAMES, all_adapters
from ..datasets.resolver import DEFAULT_DATASET_DIR
from ..task import TaskContext

REPO_ID = "osunlp/HippoRAG_2"
REPO_TYPE = "dataset"
# 按顺序尝试，第一个连得上的就用。
ENDPOINTS = ("https://hf-mirror.com", "https://huggingface.co")

# 仓库里的文件名 -> 存到本地的文件名。
REMOTE_NAMES = {
    "narrativeqa.json": "narrativeqa_dev_10_doc.json",
    "narrativeqa_corpus.json": "narrativeqa_dev_10_doc_corpus.json",
}

MAX_ATTEMPTS = 3

# 行数缓存：键是 (路径, mtime_ns, 大小)，文件一改键就变，不用手动失效。
# 没有它每次 file_status() 都要解析上百 MB JSON（narrativeqa 单文件 94MB），
# 而四个前端视图都读这个端点。
_ROW_CACHE: dict[tuple[str, int, int], tuple[int | None, str | None]] = {}


def _count_rows(path: Path, stat: os.stat_result) -> tuple[int | None, str | None]:
    """解析成 JSON 数组并数行，返回 ``(rows, error)``。同一份文件只解析一次。"""
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _ROW_CACHE:
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or not rows:
                raise ValueError("expected a non-empty JSON array")
            _ROW_CACHE[key] = (len(rows), None)
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            _ROW_CACHE[key] = (None, str(exc)[:200])
    return _ROW_CACHE[key]


def local_files() -> list[str]:
    """四组适配器声明的 QA 与语料文件名。"""
    names: list[str] = []
    for adapter in all_adapters():
        names += [adapter.qa_filename, adapter.corpus_filename]
    return names


def file_status(dataset_dir: Path | None = None) -> list[dict[str, object]]:
    """各原始文件在不在、多大、能不能解析成 JSON 数组。"""
    root = dataset_dir or DEFAULT_DATASET_DIR
    entries: list[dict[str, object]] = []
    for adapter in all_adapters():
        for kind, name in (("qa", adapter.qa_filename), ("corpus", adapter.corpus_filename)):
            path = root / name
            stat = path.stat() if path.is_file() else None
            rows, error = _count_rows(path, stat) if stat else (None, None)
            entries.append(
                {
                    "dataset": adapter.name,
                    "kind": kind,
                    "file": name,
                    "present": stat is not None,
                    "size_bytes": stat.st_size if stat else None,
                    "rows": rows,
                    "error": error,
                }
            )
    return entries


def _resolve_endpoint(ctx: TaskContext) -> tuple[str, dict[str, int]]:
    errors: list[str] = []
    for endpoint in ENDPOINTS:
        try:
            info = HfApi(endpoint=endpoint).repo_info(
                REPO_ID, repo_type=REPO_TYPE, files_metadata=True
            )
        except Exception as exc:  # noqa: BLE001 - 换下一个站点
            errors.append(f"{endpoint}: {type(exc).__name__}: {str(exc)[:120]}")
            ctx.log(f"下载源不可用：{endpoint}", "warn")
            continue
        sizes = {s.rfilename: s.size for s in (info.siblings or []) if s.size is not None}
        ctx.log(f"使用下载源 {endpoint}")
        return endpoint, sizes
    raise RuntimeError("没有可用的 Hugging Face 站点：" + "; ".join(errors))


def _fetch(local: str, endpoint: str, expected: int | None, dest: Path) -> str:
    target = dest / local
    if target.is_file() and (expected is None or target.stat().st_size == expected):
        return "skipped"

    remote = REMOTE_NAMES.get(local, local)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            cached = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=remote,
                endpoint=endpoint,
                cache_dir=dest / ".hf_cache",
            )
        except (HfHubHTTPError, OSError):
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(2**attempt)
            continue
        # 从缓存拷出来，让 dataset/ 下是真实文件而不是链接。
        shutil.copyfile(cached, target)
        return "downloaded"
    raise RuntimeError(f"unreachable: {remote}")


def run(ctx: TaskContext) -> None:
    """下载 + 校验。``datasets`` 为空时下载全部四组。"""
    selected = list(ctx.params.get("datasets") or DATASET_NAMES)
    unknown = sorted(set(selected) - set(DATASET_NAMES))
    if unknown:
        raise ValueError(f"未知数据集：{unknown}")

    dest = DEFAULT_DATASET_DIR
    dest.mkdir(parents=True, exist_ok=True)
    wanted = [
        name
        for adapter in all_adapters()
        if adapter.name in selected
        for name in (adapter.qa_filename, adapter.corpus_filename)
    ]

    ctx.progress(0, len(wanted) + 1, "连接下载源")
    endpoint, sizes = _resolve_endpoint(ctx)

    for index, local in enumerate(wanted):
        ctx.checkpoint()
        ctx.progress(index, len(wanted) + 1, f"下载 {local}")
        outcome = _fetch(local, endpoint, sizes.get(REMOTE_NAMES.get(local, local)), dest)
        ctx.log(f"{local}: {'已存在，跳过' if outcome == 'skipped' else '下载完成'}")

    cache = dest / ".hf_cache"
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)

    ctx.progress(len(wanted), len(wanted) + 1, "校验原始文件")
    broken = [
        entry
        for entry in file_status()
        if entry["file"] in wanted and (not entry["present"] or entry["error"])
    ]
    for entry in broken:
        ctx.log(f"{entry['file']}: {entry['error'] or '缺失'}", "error")
    if broken:
        raise RuntimeError(f"{len(broken)} 个原始文件校验失败")

    ctx.progress(len(wanted) + 1, len(wanted) + 1, "下载与校验完成")
    ctx.log(f"{len(wanted)} 个原始文件就绪")
