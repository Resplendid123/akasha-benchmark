"""配置层存取：Akasha 连接与模型 provider。"""

from __future__ import annotations

import sqlite3
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .db import utc_now

CONNECTION_FIELDS = (
    "base_url",
    "email",
    "password",
    "database_url",
    "timeout_seconds",
    "concurrency",
    "request_interval_seconds",
    "poll_interval_seconds",
    "poll_timeout_seconds",
)

_FLOATS = {
    "timeout_seconds",
    "request_interval_seconds",
    "poll_interval_seconds",
    "poll_timeout_seconds",
}
_INTS = {"concurrency"}

ROLES = ("judge", "attribution")

# 显式列而不是 SELECT *：建表用 IF NOT EXISTS，老库里已删的列还在，
# SELECT * 会把它们带回响应里。
_PROVIDER_COLUMNS = "id, role, label, base_url, model, api_key, updated_at"
_CONNECTION_COLUMNS = ", ".join(("id", *CONNECTION_FIELDS, "updated_at"))

# 这两个字段的主机名要过 _prefer_ipv4。
_HOST_URLS = {"base_url", "database_url"}


def _prefer_ipv4(url: str) -> str:
    """把主机名恰好是 ``localhost`` 的换成 ``127.0.0.1``。

    Windows 上 ``localhost`` 会先解析到 ``::1``，而服务通常只听 IPv4，
    于是每个请求都要先等 ``::1`` 被拒。写 ``[::1]`` 的不动。
    """
    parts = urlsplit(url)
    if parts.hostname != "localhost":
        return url
    port = f":{parts.port}" if parts.port else ""
    credentials = parts.netloc.rpartition("@")[0]
    netloc = f"{credentials}@127.0.0.1{port}" if credentials else f"127.0.0.1{port}"
    return urlunsplit(parts._replace(netloc=netloc))


def get_connection_row(connection: sqlite3.Connection) -> dict[str, Any]:
    row = connection.execute(
        f"SELECT {_CONNECTION_COLUMNS} FROM akasha_connection WHERE id = 1"
    ).fetchone()
    return dict(row) if row is not None else {}


def normalize_hosts(connection: sqlite3.Connection) -> list[str]:
    """把已存的 localhost 换成 127.0.0.1，返回改了哪几个字段。

    改库里的值而不是读出来再换，配置页显示的与实际连的因此是同一个地址。
    """
    row = get_connection_row(connection)
    changed = {
        name: _prefer_ipv4(row[name])
        for name in _HOST_URLS
        if row.get(name) and _prefer_ipv4(row[name]) != row[name]
    }
    if changed:
        update_connection(connection, **changed)
    return sorted(changed)


def sanitize_connection(payload: dict[str, Any]) -> dict[str, Any]:
    """只认已知字段并转好类型。没提交的字段不出现在结果里，因此保持原值。"""
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in CONNECTION_FIELDS:
            continue
        try:
            if key in _FLOATS:
                cleaned[key] = float(value)
            elif key in _INTS:
                cleaned[key] = int(value)
            elif key in _HOST_URLS:
                cleaned[key] = _prefer_ipv4(str(value).strip())
            else:
                cleaned[key] = str(value)
        except (TypeError, ValueError) as exc:
            kind = "number" if key in _FLOATS | _INTS else "string"
            raise ValueError(f"{key}: expected a {kind}, got {value!r}") from exc
    return cleaned


def update_connection(connection: sqlite3.Connection, **fields: Any) -> None:
    unknown = set(fields) - set(CONNECTION_FIELDS)
    if unknown:
        raise ValueError(f"unknown connection fields: {sorted(unknown)}")
    if not fields:
        return
    assignments = ", ".join(f"{name} = ?" for name in fields)
    connection.execute(
        f"UPDATE akasha_connection SET {assignments}, updated_at = ? WHERE id = 1",
        (*fields.values(), utc_now()),
    )


def upsert_provider(
    connection: sqlite3.Connection,
    *,
    role: str,
    label: str,
    base_url: str,
    model: str,
    api_key: str,
    provider_id: int | None = None,
) -> int:
    """存一个端点。给了 ``provider_id`` 就改那一行（可改 label），否则按
    (role, label) 认行。"""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if provider_id is not None:
        connection.execute(
            """
            UPDATE model_provider
               SET label = ?, base_url = ?, model = ?, api_key = ?, updated_at = ?
             WHERE id = ? AND role = ?
            """,
            (label, base_url, model, api_key, utc_now(), provider_id, role),
        )
        return provider_id
    connection.execute(
        """
        INSERT INTO model_provider
            (role, label, base_url, model, api_key, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(role, label) DO UPDATE SET
            base_url = excluded.base_url,
            model = excluded.model,
            api_key = excluded.api_key,
            updated_at = excluded.updated_at
        """,
        (role, label, base_url, model, api_key, utc_now()),
    )
    row = connection.execute(
        "SELECT id FROM model_provider WHERE role = ? AND label = ?", (role, label)
    ).fetchone()
    return int(row["id"])


def list_providers(
    connection: sqlite3.Connection, role: str | None = None
) -> list[dict[str, Any]]:
    sql = f"SELECT {_PROVIDER_COLUMNS} FROM model_provider"
    params: tuple[Any, ...] = ()
    if role is not None:
        sql += " WHERE role = ?"
        params = (role,)
    return [dict(row) for row in connection.execute(sql + " ORDER BY role, label", params)]


def get_provider(connection: sqlite3.Connection, provider_id: int) -> dict[str, Any] | None:
    row = connection.execute(
        f"SELECT {_PROVIDER_COLUMNS} FROM model_provider WHERE id = ?", (provider_id,)
    ).fetchone()
    return dict(row) if row else None


def delete_provider(connection: sqlite3.Connection, provider_id: int) -> int:
    return connection.execute(
        "DELETE FROM model_provider WHERE id = ?", (provider_id,)
    ).rowcount
