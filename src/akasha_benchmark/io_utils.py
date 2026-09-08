"""原子写文件与内容哈希。

每一轮都会写 manifest，记录它读到的和产出的内容的 sha256，
这样下游轮次能证明自己看到的字节和上游看到的是同一份。

所有写入走「临时文件 + ``os.replace``」，中断时留下的要么是旧文件、
要么是新文件，不会是一个被截断的半成品。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """文件的 sha256 十六进制值。分块读，所以 94MB 的输入也不贵。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    """manifest 用的时间戳：UTC、秒精度、以 Z 结尾。"""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _replace(tmp: Path, target: Path) -> None:
    os.replace(tmp, target)


def atomic_write_text(path: Path, text: str) -> None:
    """写文本。newline 固定为 \\n，避免 Windows 下写出 CRLF 让 sha256 对不上。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        _replace(tmp, path)
    except BaseException:
        # 包括 KeyboardInterrupt：任何情况下都不留临时文件。
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    """ensure_ascii=False，中文直接落盘，不转成 \\uXXXX。"""
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    """每行一个 JSON 对象。返回写入的行数。"""
    count = 0
    lines: list[str] = []
    for row in rows:
        lines.append(json.dumps(row, ensure_ascii=False))
        count += 1
    atomic_write_text(path, "".join(f"{line}\n" for line in lines))
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """流式读 jsonl，跳过空行。解析失败时报出具体行号。"""
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON: {exc}") from exc


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
