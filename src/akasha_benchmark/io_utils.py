"""文件哈希与 JSON 读取。原始文件的 sha256 进库，作为产物的上游凭据。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
