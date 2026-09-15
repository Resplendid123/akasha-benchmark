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

ROLES = ("judge", "attribution")

# 这两个字段的主机名要过 _prefer_ipv4。
_HOST_URLS = {"base_url", "database_url"}


def _prefer_ipv4(url: str) -> str:
    """把主机名恰好是 ``localhost`` 的换成 ``127.0.0.1``。
    """
    parts = urlsplit(url)
    if parts.hostname != "localhost":
        return url
    port = f":{parts.port}" if parts.port else ""
    credentials = parts.netloc.rpartition("@")[0]
    netloc = f"{credentials}@127.0.0.1{port}" if credentials else f"127.0.0.1{port}"
    return urlunsplit(parts._replace(netloc=netloc))


def get_connection_row(connection: sqlite3.Connection) -> dict[str, Any]:
    row = connection.execute("SELECT * FROM akasha_connection WHERE id = 1").fetchone()
    return dict(row) if row is not None else {}


def sanitize_connection(payload: dict[str, Any]) -> dict[str, Any]:
    """只认已知字段并转好类型。没提交的字段不出现在结果里，因此保持原值。"""
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in CONNECTION_FIELDS:
            continue
        try:
            if key in _FLOATS:
                cleaned[key] = float(value)
            elif key in _HOST_URLS:
                cleaned[key] = _prefer_ipv4(str(value).strip())
            else:
                cleaned[key] = str(value)
        except (TypeError, ValueError) as exc:
            kind = "number" if key in _FLOATS else "string"
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
    concurrency: int = 1,
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
               SET label = ?, base_url = ?, model = ?, api_key = ?,
                   concurrency = ?, updated_at = ?
             WHERE id = ? AND role = ?
            """,
            (label, base_url, model, api_key, concurrency, utc_now(), provider_id, role),
        )
        return provider_id
    connection.execute(
        """
        INSERT INTO model_provider
            (role, label, base_url, model, api_key, concurrency, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(role, label) DO UPDATE SET
            base_url = excluded.base_url,
            model = excluded.model,
            api_key = excluded.api_key,
            concurrency = excluded.concurrency,
            updated_at = excluded.updated_at
        """,
        (role, label, base_url, model, api_key, concurrency, utc_now()),
    )
    row = connection.execute(
        "SELECT id FROM model_provider WHERE role = ? AND label = ?", (role, label)
    ).fetchone()
    return int(row["id"])


def list_providers(connection: sqlite3.Connection, role: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM model_provider"
    params: tuple[Any, ...] = ()
    if role is not None:
        sql += " WHERE role = ?"
        params = (role,)
    return [dict(row) for row in connection.execute(sql + " ORDER BY role, label", params)]


def get_provider(connection: sqlite3.Connection, provider_id: int) -> dict[str, Any] | None:
    row = connection.execute("SELECT * FROM model_provider WHERE id = ?", (provider_id,)).fetchone()
    return dict(row) if row else None


def delete_provider(connection: sqlite3.Connection, provider_id: int) -> int:
    return connection.execute("DELETE FROM model_provider WHERE id = ?", (provider_id,)).rowcount


# --- Akasha 模型配置组 ---


def list_config_groups(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    sql = "SELECT * FROM akasha_config_group ORDER BY label"
    return [dict(row) for row in connection.execute(sql)]


def get_config_group(connection: sqlite3.Connection, group_id: int) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT * FROM akasha_config_group WHERE id = ?", (group_id,)
    ).fetchone()
    return dict(row) if row else None


def selected_config_group(connection: sqlite3.Connection) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT * FROM akasha_config_group WHERE selected = 1 LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def upsert_config_group(
    connection: sqlite3.Connection,
    *,
    label: str,
    configs_json: str,
    group_id: int | None = None,
) -> int:
    """存一组配置。给了 ``group_id`` 就改那一行（可改 label），否则按 label 认行。"""
    if group_id is not None:
        connection.execute(
            "UPDATE akasha_config_group SET label = ?, configs_json = ?, updated_at = ? WHERE id = ?",
            (label, configs_json, utc_now(), group_id),
        )
        return group_id
    connection.execute(
        """
        INSERT INTO akasha_config_group (label, configs_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(label) DO UPDATE SET
            configs_json = excluded.configs_json,
            updated_at = excluded.updated_at
        """,
        (label, configs_json, utc_now()),
    )
    row = connection.execute(
        "SELECT id FROM akasha_config_group WHERE label = ?", (label,)
    ).fetchone()
    return int(row["id"])


def delete_config_group(connection: sqlite3.Connection, group_id: int) -> int:
    return connection.execute(
        "DELETE FROM akasha_config_group WHERE id = ?", (group_id,)
    ).rowcount


def set_selected_group(connection: sqlite3.Connection, group_id: int) -> None:
    """置本组为选中，清掉其它组的选中。"""
    connection.execute("UPDATE akasha_config_group SET selected = 0 WHERE selected = 1")
    connection.execute(
        "UPDATE akasha_config_group SET selected = 1 WHERE id = ?", (group_id,)
    )
