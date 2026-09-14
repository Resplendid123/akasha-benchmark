"""文件哈希与 JSON 读取。

原始文件的 sha256 进库，作为「这批产物是从哪份数据抽出来的」的上游凭据。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """分块读，所以 94MB 的输入也不贵。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
