"""Akasha 连接配置从 SQLite 的单例 connection 表读取。

库中保存凭据；入库与查询的快照使用 redacted() 隐去密钥。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AkashaConfig:
    base_url: str = "http://localhost:3000"
    email: str = ""
    password: str = ""
    api_prefix: str = "/api"
    timeout_seconds: float = 180.0
    concurrency: int = 1
    request_interval_seconds: float = 0.5
    poll_interval_seconds: float = 10.0
    poll_timeout_seconds: float = 7200.0
    # 仅审计归因与血缘视图需要，且是可选的：不填则跳过，其余指标照常算。
    database_url: str = ""

    def api(self, path: str) -> str:
        """拼出完整 API URL，两侧的斜杠都容错。"""
        return f"{self.base_url.rstrip('/')}{self.api_prefix}/{path.lstrip('/')}"

    def require_credentials(self) -> None:
        missing = [n for n in ("base_url", "email", "password") if not getattr(self, n)]
        if missing:
            raise ValueError(
                f"the Akasha connection is missing {missing}. "
                "Fill it in the platform's settings view."
            )

    def redacted(self) -> dict[str, Any]:
        """可以安全写进 manifest 的视图：只有连接形态，没有密钥。"""
        return {
            "base_url": self.base_url,
            "api_prefix": self.api_prefix,
            "email": self.email,
            "password": "***" if self.password else "",
            "database_url": "***" if self.database_url else "",
            "timeout_seconds": self.timeout_seconds,
            "concurrency": self.concurrency,
            "request_interval_seconds": self.request_interval_seconds,
        }

    def for_ui(self) -> dict[str, Any]:
        """给配置层的视图。

        白名单式 —— 新增字段的默认行为是不输出，避免漏配。连接配置是单例，
        凭据与 base_url 走同一条路径：库中存明文，UI 直接读写。
        """
        return {
            "base_url": self.base_url,
            "email": self.email,
            "api_prefix": self.api_prefix,
            "password": self.password,
            "database_url": self.database_url,
            "timeout_seconds": self.timeout_seconds,
            "concurrency": self.concurrency,
            "request_interval_seconds": self.request_interval_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "poll_timeout_seconds": self.poll_timeout_seconds,
        }


# 连接行里属于 AkashaConfig 的列。那张表另有 id / 时间戳 / 上次测连接的结果,
# 那些不进配置。
FIELD_NAMES = frozenset(f.name for f in fields(AkashaConfig))

_FLOATS = {
    "timeout_seconds",
    "request_interval_seconds",
    "poll_interval_seconds",
    "poll_timeout_seconds",
}
_INTS = {"concurrency"}


def _cast(name: str, raw: Any) -> Any:
    if name in _FLOATS:
        return float(raw)
    if name in _INTS:
        return int(raw)
    return str(raw)


def from_row(row: Any) -> AkashaConfig:
    """把 ``connection`` 那一行转成 :class:`AkashaConfig`。"""
    return AkashaConfig(**{name: row[name] for name in FIELD_NAMES if name in row.keys()})


def load_config(connection: sqlite3.Connection | None = None) -> AkashaConfig:
    """那一份连接配置。

    ``connection`` 为 None 时返回默认值 —— 报错留给
    :meth:`AkashaConfig.require_credentials`，那里的提示能指向缺的具体是哪一项。
    """
    if connection is None:
        return AkashaConfig()

    from .store import repo

    row = repo.get_connection_row(connection)
    return from_row(row) if row is not None else AkashaConfig()


def load_config_from_db_path(db_path: Path | str | None = None) -> AkashaConfig:
    """按库路径开一个只读连接读配置。"""
    from .store.db import DEFAULT_DB_PATH, connect

    target = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not target.is_file():
        return AkashaConfig()
    connection = connect(target, read_only=True)
    try:
        return load_config(connection)
    finally:
        connection.close()


def sanitize_updates(payload: dict[str, Any]) -> dict[str, Any]:
    """把配置表单提交的内容整成可写库的形状。

    只认已知字段（未知键丢掉，不让它们进表变成噪音）。表单里没碰过的字段
    不会出现在 payload 里，自然不会被改写 —— 这同时是 password / dburl 的
    「空 = 不改」语义，不需要单独处理。
    """
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in FIELD_NAMES:
            continue
        try:
            cleaned[key] = _cast(key, value)
        except (TypeError, ValueError) as exc:
            kind = "number" if key in _FLOATS | _INTS else "string"
            raise ValueError(f"{key}: expected a {kind}, got {value!r}") from exc
    return cleaned
