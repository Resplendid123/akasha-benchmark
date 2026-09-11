"""需要连 Akasha 的那几个阶段的配置。**只存在库里。**

配置的唯一来源是评测库的 ``connection`` 表，在平台的配置层里填。曾经有过两条
旁路（``akasha.config.json``、``AKASHA_*`` 环境变量覆盖），两条都删了 ——
同一份配置有多个来源时，「我改了但没生效」是查不出来的，而那个成本远高于
少一条旁路带来的不便。

**只有一份配置**（``connection`` 表的 ``CHECK (id = 1)`` 把这一点写进了 schema）,
只能改，不能新增。:class:`AkashaConfig` 是它的运行时形态。

历史记录不靠外键：``index_layer.connection_json`` 存了入库时那份配置的 redacted
快照，所以「这一层当时跑在什么上」查得到，而不必让配置本身变成多行。

密钥只存在 :class:`AkashaConfig` 里，绝不写进 manifest —— 入库与查询记录的是
:meth:`AkashaConfig.redacted` 的结果。
"""

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
        """给配置层的视图：密钥字段只报「是否已设置」，不回传取值。

        白名单式 —— 新增字段的默认行为是不输出，漏写一个不会泄露密钥。
        取值与「是否设置」分开，这样 UI 能显示占位符而不必拿到明文。
        """
        return {
            "base_url": self.base_url,
            "email": self.email,
            "api_prefix": self.api_prefix,
            "timeout_seconds": self.timeout_seconds,
            "concurrency": self.concurrency,
            "request_interval_seconds": self.request_interval_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "poll_timeout_seconds": self.poll_timeout_seconds,
            "password_set": bool(self.password),
            "database_url_set": bool(self.database_url),
        }


# 这些字段是密钥，UI 传空串表示「不改」而不是「清空」。
SECRET_FIELDS = frozenset({"password", "database_url"})

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

    两条规则：只认已知字段（未知键丢掉，不让它们进表变成噪音）；
    **密钥字段的空串表示「不改」**。后者是因为 UI 拿不到明文密钥，
    表单里那一格提交上来必然是空的 —— 当成「清空」会让每次改 base_url
    都顺手把密码删掉。
    """
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in FIELD_NAMES:
            continue
        if key in SECRET_FIELDS and (value is None or value == ""):
            continue
        try:
            cleaned[key] = _cast(key, value)
        except (TypeError, ValueError) as exc:
            kind = "number" if key in _FLOATS | _INTS else "string"
            raise ValueError(f"{key}: expected a {kind}, got {value!r}") from exc
    return cleaned
