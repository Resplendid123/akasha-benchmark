"""从 Hugging Face Hub 下载 HippoRAG_2 的四组评测数据。

来源：https://huggingface.co/datasets/osunlp/HippoRAG_2

四组 QA 文件与对应语料下载到 ``dataset/``。仓库里 NarrativeQA 那一份叫
``narrativeqa_dev_10_doc``，本地统一存成 ``narrativeqa``，让四组的命名一致。

用法：
    uv run python download_dataset.py
    uv run python download_dataset.py --check      # 只校验，不下载
    HF_ENDPOINT=https://huggingface.co uv run python download_dataset.py

重跑很便宜：已存在且字节数符合预期的文件会跳过，
所以下载中断后直接重跑即可。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "osunlp/HippoRAG_2"
REPO_TYPE = "dataset"
DEST = Path(__file__).parent / "dataset"

# 候选站点，按顺序尝试。hf-mirror.com 是只读的社区镜像，
# 在访问不到 huggingface.co 的网络环境下有用。
ENDPOINTS = ("https://hf-mirror.com", "https://huggingface.co")

# (仓库里的文件名, 存到本地的文件名)
FILES: tuple[tuple[str, str], ...] = (
    ("hotpotqa.json", "hotpotqa.json"),
    ("hotpotqa_corpus.json", "hotpotqa_corpus.json"),
    ("musique.json", "musique.json"),
    ("musique_corpus.json", "musique_corpus.json"),
    ("2wikimultihopqa.json", "2wikimultihopqa.json"),
    ("2wikimultihopqa_corpus.json", "2wikimultihopqa_corpus.json"),
    ("narrativeqa_dev_10_doc.json", "narrativeqa.json"),
    ("narrativeqa_dev_10_doc_corpus.json", "narrativeqa_corpus.json"),
)

MAX_ATTEMPTS = 3


def resolve_endpoint() -> tuple[str, dict[str, int]]:
    """返回第一个连得上的站点，以及它报告的各文件大小。"""
    candidates = (os.environ["HF_ENDPOINT"],) if os.environ.get("HF_ENDPOINT") else ENDPOINTS
    errors: list[str] = []

    for endpoint in candidates:
        try:
            info = HfApi(endpoint=endpoint).repo_info(
                REPO_ID, repo_type=REPO_TYPE, files_metadata=True
            )
        except Exception as exc:  # noqa: BLE001 - 这里就是在探测连通性
            errors.append(f"  {endpoint}: {type(exc).__name__}: {str(exc)[:200]}")
            print(f"unreachable: {endpoint}", file=sys.stderr)
            continue

        sizes = {s.rfilename: s.size for s in info.siblings if s.size is not None}
        print(f"using endpoint: {endpoint}")
        return endpoint, sizes

    raise SystemExit("No reachable Hugging Face endpoint.\n" + "\n".join(errors))


def fetch(remote: str, local: str, endpoint: str, expected: int | None) -> Path:
    """下载一个文件到 DEST，用本地名保存，失败时按指数退避重试。"""
    target = DEST / local

    if target.exists() and (expected is None or target.stat().st_size == expected):
        print(f"  skip (present)  {local}")
        return target

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            cached = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=remote,
                endpoint=endpoint,
                cache_dir=DEST / ".hf_cache",
            )
        except (HfHubHTTPError, OSError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = 2**attempt
            print(f"  retry {attempt}/{MAX_ATTEMPTS - 1} in {wait}s ({type(exc).__name__})")
            time.sleep(wait)
            continue

        # 从缓存里拷出来，让 dataset/ 下是真实文件而不是链接。
        shutil.copyfile(cached, target)
        label = f"  downloaded     {local}"
        print(f"{label}" if remote == local else f"{label}   (from {remote})")
        return target

    raise RuntimeError(f"unreachable: {remote}")  # pragma: no cover


def summarize(paths: list[Path]) -> int:
    """逐个确认能解析成 JSON，并打印体积和记录数。返回解析失败的文件数。"""
    print(f"\n{'file':<32}{'size':>12}{'records':>12}")
    print("-" * 56)

    failures = 0
    for path in paths:
        size = f"{path.stat().st_size / 1e6:.2f} MB"
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            count = str(len(data)) if isinstance(data, (list, dict)) else "n/a"
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            count = "INVALID"
            failures += 1
            print(f"  {path.name}: {exc}", file=sys.stderr)
        print(f"{path.name:<32}{size:>12}{count:>12}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="只校验已有文件，不下载"
    )
    parser.add_argument(
        "--keep-cache", action="store_true", help="保留中间的 HF 缓存目录"
    )
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)

    if args.check:
        present = [DEST / local for _, local in FILES if (DEST / local).exists()]
        missing = [local for _, local in FILES if not (DEST / local).exists()]
        failures = summarize(present) if present else 0
        for name in missing:
            print(f"MISSING: {name}", file=sys.stderr)
        return 1 if (failures or missing) else 0

    endpoint, sizes = resolve_endpoint()
    print(f"downloading {len(FILES)} files to {DEST}")

    paths = [fetch(remote, local, endpoint, sizes.get(remote)) for remote, local in FILES]

    failures = summarize(paths)

    cache = DEST / ".hf_cache"
    if cache.exists() and not args.keep_cache:
        shutil.rmtree(cache, ignore_errors=True)

    if failures:
        print(f"\n{failures} file(s) failed to parse as JSON", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
