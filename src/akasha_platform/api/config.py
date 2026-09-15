"""配置层：Akasha 连接、它那边的模型配置、本地 judge / 归因端点。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request

from akasha_benchmark.akasha_client import AkashaClient, AkashaError
from akasha_benchmark.judge import JudgeClient, JudgeConfigError
from akasha_benchmark.judge.providers import resolve_provider
from akasha_benchmark import model_configs as model_configs_module
from akasha_benchmark.model_configs import FEATURES, drift
from akasha_benchmark.store import compile_store, config_store, dumps, loads

from ._common import config_of, db, writable

router = APIRouter(prefix="/api")


@router.get("/connection")
def get_connection(request: Request) -> dict[str, Any]:
    """那一份 Akasha 连接配置，连同各次编译的空间。密码原样回显。"""
    with db(request) as connection:
        row = config_store.get_connection_row(connection)
        compiles = [
            {
                "id": int(r["id"]),
                "run_id": r["run_id"],
                "workspace_id": r["workspace_id"],
                "space_id": r["space_id"],
            }
            for r in compile_store.list_compile_runs(connection)
            if r["space_id"]
        ]
    return {**{k: v for k, v in row.items() if k != "id"}, "compiles": compiles}


@router.put("/connection")
def put_connection(request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """改连接配置。只写提交了的字段，没出现的保持原值，传空串即清空。"""
    try:
        fields = config_store.sanitize_connection(payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    with writable(request) as connection:
        config_store.update_connection(connection, **fields)
    return {"updated": sorted(fields), "connection": get_connection(request)}


@router.post("/connection/test")
def test_connection(request: Request) -> dict[str, Any]:
    """登录 + 取当前用户 + 拉模型配置，并预检 owner 角色与各次编译的 workspace。

    owner 那一项不是可选检查：非 owner 会在授权闸门静默丢弃 chunk，
    症状看起来像召回质量差。
    """
    config = config_of(request)
    try:
        config.require_credentials()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    try:
        with AkashaClient(config) as client:
            client.login()
            me = client.current_user()
            model_configs = client.get_model_configs()
    except AkashaError as exc:
        raise HTTPException(502, f"Akasha 拒绝了请求：{exc}") from exc
    except OSError as exc:
        raise HTTPException(502, f"连不上 {config.base_url}：{exc}") from exc

    user = (me or {}).get("user") or {}
    workspace = (me or {}).get("workspace") or {}
    role = user.get("role")

    # 同一道判据 compile / query 在登录后也会走，这里先说出来。
    with db(request) as connection:
        blocked = []
        for row in compile_store.list_compile_runs(connection):
            reason = compile_store.workspace_mismatch(
                connection, int(row["id"]), workspace.get("id")
            )
            if reason:
                blocked.append({"id": int(row["id"]), "run_id": row["run_id"], "reason": reason})

    # 选中的本地组与远端逐项比对（不含 apiKey），供前端提示是否需要应用。
    with db(request) as connection:
        group = config_store.selected_config_group(connection)
    group_drift = None
    if group is not None:
        configs = loads(group["configs_json"], {})
        group_drift = {
            "id": int(group["id"]),
            "label": group["label"],
            "drift": drift(model_configs, model_configs_module.group_to_live(configs)),
        }

    return {
        "ok": True,
        "user": {"id": user.get("id"), "email": user.get("email"), "role": role},
        # workspace 由服务端解析，只读。
        "workspace": {"id": workspace.get("id"), "name": workspace.get("name")},
        "is_owner": role == "owner",
        # 这些编译在当前连接下用不了。
        "blocked_compiles": blocked,
        "owner_warning": (
            None
            if role == "owner"
            else f"这个账号的角色是 {role!r}，不是 owner。非 owner 会在授权闸门静默"
            "丢弃 chunk，症状看起来像召回质量差。编译前请提权。"
        ),
        "model_configs": model_configs,
        "group_drift": group_drift,
    }


@router.get("/model-configs")
def get_model_configs(request: Request) -> dict[str, Any]:
    """Akasha 的四项模型配置，连同各次编译的快照比对。"""
    config = config_of(request)
    try:
        config.require_credentials()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        with AkashaClient(config) as client:
            client.login()
            live = client.get_model_configs()
    except AkashaError as exc:
        raise HTTPException(502, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(502, f"连不上 {config.base_url}：{exc}") from exc

    with db(request) as connection:
        compiles = [
            {
                "id": int(row["id"]),
                "run_id": row["run_id"],
                "drift": drift(live, loads(row["model_configs_json"])),
            }
            for row in compile_store.list_compile_runs(connection)
            if row["model_configs_json"]
        ]
    return {"features": list(FEATURES), "live": live, "compiles": compiles}


@router.put("/model-configs/{feature}")
def put_model_config(
    request: Request, feature: str, payload: dict[str, Any] = Body(...)
) -> dict[str, Any]:
    """改 Akasha 的某一项模型配置。改的是那个部署，不是本平台。

    ``provider`` 不进表单（只有一个合法取值），在这里补上，免得 PUT 缺必填字段。
    """
    if feature not in FEATURES:
        raise HTTPException(422, f"未知配置项 {feature!r}；可用：{list(FEATURES)}")
    payload = {"provider": "openai-compatible", **payload}
    config = config_of(request)
    try:
        config.require_credentials()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        with AkashaClient(config) as client:
            client.login()
            result = client.put_model_config(feature, payload)
    except AkashaError as exc:
        raise HTTPException(502, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(502, f"连不上 {config.base_url}：{exc}") from exc

    requires_rebuild = feature in {"compiler", "embedding"}
    return {
        "feature": feature,
        "result": result,
        "requires_new_compile": requires_rebuild,
        "impact": "需要重新编译" if requires_rebuild else "不影响已有编译",
    }


@router.get("/providers")
def list_providers(request: Request, role: str | None = None) -> list[dict[str, Any]]:
    """judge / 归因端点。响应里没有 api_key，只有它是否已设置。"""
    if role is not None and role not in config_store.ROLES:
        raise HTTPException(422, f"role 必须是 {list(config_store.ROLES)} 之一")
    with db(request) as connection:
        rows = config_store.list_providers(connection, role)
    return [
        {
            **{k: v for k, v in row.items() if k != "api_key"},
            "api_key_set": bool((row["api_key"] or "").strip()),
        }
        for row in rows
    ]


@router.put("/providers/{role}")
def put_provider(
    request: Request, role: str, payload: dict[str, Any] = Body(...)
) -> dict[str, Any]:
    """存一个端点。带 ``id`` 是改那一条（可改 label），不带是按 label 认行。
    ``api_key`` 为空时保留现有密钥。"""
    if role not in config_store.ROLES:
        raise HTTPException(422, f"role 必须是 {list(config_store.ROLES)} 之一")
    label = str(payload.get("label") or "default").strip()
    base_url = str(payload.get("base_url") or "").strip()
    model = str(payload.get("model") or "").strip()
    if not base_url or not model:
        raise HTTPException(422, "base_url 与 model 必填")
    try:
        provider_id = None if payload.get("id") is None else int(payload["id"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, f"id 必须是整数，收到 {payload['id']!r}") from exc

    try:
        concurrency = max(1, int(payload.get("concurrency") or 1))
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, f"concurrency 必须是整数，收到 {payload.get('concurrency')!r}") from exc

    api_key = str(payload.get("api_key") or "")
    with writable(request) as connection:
        rows = config_store.list_providers(connection, role)
        by_id = {int(row["id"]): row for row in rows}
        if provider_id is not None and provider_id not in by_id:
            raise HTTPException(404, f"{role} 端点 #{provider_id} 不存在")
        # 改名撞上另一条时拒绝，否则那一条会被覆盖掉。
        clash = next((r for r in rows if r["label"] == label and int(r["id"]) != provider_id), None)
        if clash is not None and provider_id is not None:
            raise HTTPException(409, f"{role} 下已经有一个叫 {label!r} 的端点")

        existing = by_id.get(provider_id) if provider_id is not None else clash
        if not api_key:
            api_key = (existing or {}).get("api_key", "")
        try:
            provider_id = config_store.upsert_provider(
                connection,
                role=role,
                label=label,
                base_url=base_url,
                model=model,
                api_key=api_key,
                concurrency=concurrency,
                provider_id=provider_id,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
    return {
        "id": provider_id,
        "role": role,
        "label": label,
        "api_key_set": bool(api_key),
        "concurrency": concurrency,
    }


@router.post("/providers/{provider_id}/probe")
def probe_provider(request: Request, provider_id: int) -> dict[str, Any]:
    """真调一次这个端点，回模型说了什么或它为什么失败。

    密钥从库里取而不经前端。失败不抛 500：调不通是这个接口要报告的结果。
    """
    with db(request) as connection:
        record = config_store.get_provider(connection, provider_id)
        if record is None:
            raise HTTPException(404, f"端点 #{provider_id} 不存在")
        try:
            provider = resolve_provider(connection, provider_id, record["role"])
        except JudgeConfigError as exc:
            return {"ok": False, "failure": "config", "detail": str(exc)}

    with JudgeClient(provider) as client:
        # prompt 里必须出现 "json"：有些 provider 以此为 json_object 格式的前提，
        # 裸一句 hi 会被它们判 400，而那是探测本身的问题。
        reply = client.complete(
            "You reply with a single JSON object.",
            'hi — reply as JSON: {"reply": "<your greeting>"}',
        )

    return {
        "ok": reply.failure_kind is None,
        "failure": reply.failure_kind,
        "status": reply.status,
        "reply": (reply.content or "")[:400],
        # 失败时留一段原文，provider 的报错通常只在这里说得清楚。
        "detail": None if reply.failure_kind is None else (reply.raw or "")[:600],
        "provider": provider.redacted(),
    }


@router.delete("/providers/{provider_id}")
def delete_provider(request: Request, provider_id: int) -> dict[str, Any]:
    with writable(request) as connection:
        removed = config_store.delete_provider(connection, provider_id)
    if not removed:
        raise HTTPException(404, f"端点 #{provider_id} 不存在")
    return {"deleted": removed}


# --- Akasha 模型配置组 ---


def _group_view(row: dict[str, Any]) -> dict[str, Any]:
    """列表视图：每项只报 apiKeySet，不回明文密钥（明文只经 export）。"""
    configs = loads(row["configs_json"], {})
    features = {}
    for feature in FEATURES:
        entry = dict(configs.get(feature) or {})
        api_key = entry.pop("apiKey", "")
        features[feature] = {**entry, "apiKeySet": bool((api_key or "").strip())}
    return {
        "id": int(row["id"]),
        "label": row["label"],
        "selected": bool(row["selected"]),
        "configs": features,
        "updated_at": row["updated_at"],
    }


@router.get("/akasha-configs")
def list_akasha_configs(request: Request) -> dict[str, Any]:
    """本地保存的 Akasha 模型配置组。响应不含明文密钥。"""
    with db(request) as connection:
        groups = config_store.list_config_groups(connection)
    return {"features": list(FEATURES), "groups": [_group_view(row) for row in groups]}


@router.put("/akasha-configs")
def put_akasha_config(request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """存一组配置。带 ``id`` 是改那一条，否则按 label 认行。
    某项 ``apiKey`` 为空时保留该项的现有密钥。"""
    label = str(payload.get("label") or "").strip()
    if not label:
        raise HTTPException(422, "label 必填")
    submitted = payload.get("configs")
    if not isinstance(submitted, dict):
        raise HTTPException(422, "configs 必须是对象")
    try:
        group_id = None if payload.get("id") is None else int(payload["id"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, f"id 必须是整数，收到 {payload['id']!r}") from exc

    with writable(request) as connection:
        groups = config_store.list_config_groups(connection)
        by_id = {int(row["id"]): row for row in groups}
        if group_id is not None and group_id not in by_id:
            raise HTTPException(404, f"配置组 #{group_id} 不存在")
        clash = next((r for r in groups if r["label"] == label and int(r["id"]) != group_id), None)
        if clash is not None and group_id is not None:
            raise HTTPException(409, f"已经有一个叫 {label!r} 的配置组")
        existing = loads(by_id[group_id]["configs_json"], {}) if group_id is not None else {}

        configs: dict[str, Any] = {}
        for feature in FEATURES:
            entry = dict(submitted.get(feature) or {})
            if not (str(entry.get("apiKey") or "").strip()):
                # 密钥留空则保留该项现有值。
                entry["apiKey"] = (existing.get(feature) or {}).get("apiKey", "")
            configs[feature] = entry

        group_id = config_store.upsert_config_group(
            connection, label=label, configs_json=dumps(configs), group_id=group_id
        )
    return {"id": group_id, "label": label}


@router.delete("/akasha-configs/{group_id}")
def delete_akasha_config(request: Request, group_id: int) -> dict[str, Any]:
    with writable(request) as connection:
        removed = config_store.delete_config_group(connection, group_id)
    if not removed:
        raise HTTPException(404, f"配置组 #{group_id} 不存在")
    return {"deleted": removed}


@router.post("/akasha-configs/{group_id}/apply")
def apply_akasha_config(request: Request, group_id: int) -> dict[str, Any]:
    """把这一组的四项配置整组推送到远端 Akasha。"""
    with db(request) as connection:
        group = config_store.get_config_group(connection, group_id)
    if group is None:
        raise HTTPException(404, f"配置组 #{group_id} 不存在")
    configs = loads(group["configs_json"], {})

    config = config_of(request)
    try:
        config.require_credentials()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    applied: list[str] = []
    try:
        with AkashaClient(config) as client:
            client.login()
            for feature in FEATURES:
                entry = configs.get(feature) or {}
                body = {"provider": "openai-compatible", **entry}
                client.put_model_config(feature, body)
                applied.append(feature)
    except AkashaError as exc:
        raise HTTPException(502, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(502, f"连不上 {config.base_url}：{exc}") from exc

    # 应用即选中：这一组现在就是远端的真实配置，drift 基准也该是它。
    with writable(request) as connection:
        config_store.set_selected_group(connection, group_id)

    requires_rebuild = any(f in {"compiler", "embedding"} for f in applied)
    return {
        "applied": applied,
        "requires_new_compile": requires_rebuild,
        "impact": "需要重新编译" if requires_rebuild else "不影响已有编译",
    }


@router.get("/config/export")
def export_config(request: Request) -> dict[str, Any]:
    """导出全部配置，含明文密钥，可回填。"""
    with db(request) as connection:
        row = config_store.get_connection_row(connection)
        connection_data = {k: v for k, v in row.items() if k not in ("id", "updated_at")}
        providers = config_store.list_providers(connection)
        groups = config_store.list_config_groups(connection)
    return {
        "connection": connection_data,
        "providers": [
            {k: v for k, v in p.items() if k not in ("id", "updated_at")} for p in providers
        ],
        "akasha_configs": [
            {"label": g["label"], "configs": loads(g["configs_json"], {})} for g in groups
        ],
    }


@router.post("/config/import")
def import_config(request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """回填一份导出的配置。同 (role, label) / label 的会被覆盖。"""
    connection_data = payload.get("connection") or {}
    providers = payload.get("providers") or []
    akasha_configs = payload.get("akasha_configs") or []
    try:
        fields = config_store.sanitize_connection(connection_data)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    try:
        with writable(request) as connection:
            config_store.update_connection(connection, **fields)
            for p in providers:
                config_store.upsert_provider(
                    connection,
                    role=str(p.get("role") or ""),
                    label=str(p.get("label") or "default"),
                    base_url=str(p.get("base_url") or ""),
                    model=str(p.get("model") or ""),
                    api_key=str(p.get("api_key") or ""),
                    concurrency=max(1, int(p.get("concurrency") or 1)),
                )
            for g in akasha_configs:
                config_store.upsert_config_group(
                    connection,
                    label=str(g.get("label") or ""),
                    configs_json=dumps(g.get("configs") or {}),
                )
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "connection": sorted(fields),
        "providers": len(providers),
        "akasha_configs": len(akasha_configs),
    }
