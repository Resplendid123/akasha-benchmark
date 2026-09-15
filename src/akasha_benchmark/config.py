"""Akasha 连接配置，从库里的单例 akasha_connection 表读。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, fields
from typing import Any

# Akasha 的全局前缀，不随部署变，所以不做成配置项。
API_PREFIX = "/api"


@dataclass(frozen=True)
class AkashaConfig:
    base_url: str = "http://127.0.0.1:3000"
    email: str = ""
    password: str = ""
    # 只读 PG，仅归因层的链路视图需要，不填则跳过那一段判据。
    database_url: str = ""
    timeout_seconds: float = 120.0
    request_interval_seconds: float = 0.5
    poll_interval_seconds: float = 30.0
    poll_timeout_seconds: float = 7200.0

    def api(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{API_PREFIX}/{path.lstrip('/')}"

    def require_credentials(self) -> None:
        missing = [n for n in ("base_url", "email", "password") if not getattr(self, n)]
        if missing:
            raise ValueError(f"Akasha 连接缺少 {missing}，请在配置页填写。")

    def redacted(self) -> dict[str, Any]:
        """可以落库或记日志的视图，不含密钥。"""
        return {
            "base_url": self.base_url,
            "email": self.email,
            "password_set": bool(self.password),
            "database_url_set": bool(self.database_url),
        }


_FIELDS = frozenset(f.name for f in fields(AkashaConfig))


def load_config(connection: sqlite3.Connection | None) -> AkashaConfig:
    """读那一份连接配置。``None`` 时返回默认值。"""
    if connection is None:
        return AkashaConfig()
    from .store import config_store

    row = config_store.get_connection_row(connection)
    return AkashaConfig(**{name: row[name] for name in _FIELDS if name in row})
