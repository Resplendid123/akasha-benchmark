"""配置层：Akasha 连接、它那边的模型配置、本地 judge / 归因端点。

配置全部收在这一处。原先编译模型配置在编译层，那让「改一个模型要去哪」
取决于它属于哪一层 —— 而用户想的是「我要改配置」。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request

from akasha_benchmark.akasha_client import AkashaClient, AkashaError
from akasha_benchmark.model_configs import FEATURES, drift
from akasha_benchmark.store import config_store, loads, run_store

from ._common import config_of, db, writable

router = APIRouter(prefix="/api")


@router.get("/connection")
def get_connection(request: Request) -> dict[str, Any]:
    """那一份 Akasha 连接配置。密码原样回显 —— 库里存的是明文，UI 直接读写。"""
    with db(request) as connection:
        row = config_store.get_connection_row(connection)
        compiles = [
            {
                "id": int(r["id"]),
                "run_id": r["run_id"],
                "workspace_id": r["workspace_id"],
                "space_id": r["space_id"],
            }
            for r in run_store.list_compile_runs(connection)
            if r["space_id"]
        ]
    return {**{k: v for k, v in row.items() if k != "id"}, "compiles": compiles}


@router.put("/connection")
def put_connection(request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """改连接配置。**只写提交了的字段**，没出现的保持原值，显式传空串即清空。"""
    try:
        fields = config_store.sanitize_connection(payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    with writable(request) as connection:
        config_store.update_connection(connection, **fields)
    return {"updated": sorted(fields), "connection": get_connection(request)}


@router.post("/connection/test")
def test_connection(request: Request) -> dict[str, Any]:
    """登录 + 取当前用户 + 拉模型配置。

    OWNER 那一项不是可选检查：非 OWNER 会在第三道授权闸门**静默丢弃 chunk**，
    症状看起来像召回质量差 —— 在这里报出来比跑完一批编译再从指标里猜便宜得多。
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

    # 拿刚解析出的 workspace 去比已有的编译。这是一次**预检** —— 同一道判据
    # compile / query 在登录后也会走，但在这里先说出来，用户不必等起了任务才发现。
    with db(request) as connection:
        blocked = []
        for row in run_store.list_compile_runs(connection):
            reason = run_store.workspace_mismatch(
                connection, int(row["id"]), workspace.get("id")
            )
            if reason:
                blocked.append({"id": int(row["id"]), "run_id": row["run_id"], "reason": reason})

    return {
        "ok": True,
        "user": {"id": user.get("id"), "email": user.get("email"), "role": role},
        # workspace 由服务端解析（自建部署走 workspaceRepo.findFirst()），只读。
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
    }


@router.get("/model-configs")
def get_model_configs(request: Request) -> dict[str, Any]:
    """Akasha 的 compiler / embedding / answer / image 四项配置。

    同时给出各次编译的快照比对：「现在的配置与那次编译是否一致」一个响应里就能看出来。
    """
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
            for row in run_store.list_compile_runs(connection)
            if row["model_configs_json"]
        ]
    return {"features": list(FEATURES), "live": live, "compiles": compiles}


@router.put("/model-configs/{feature}")
def put_model_config(
    request: Request, feature: str, payload: dict[str, Any] = Body(...)
) -> dict[str, Any]:
    """改 Akasha 的某一项模型配置。**改的是那个部署，不是本平台。**

    ``provider`` 不进表单：Akasha 的 CHECK 约束只允许 ``openai-compatible``
    一个取值，让人填一个没有选择的字段只会填错。这里补上，免得 PUT 因为
    缺必填字段被拒。
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
    """judge / 归因端点。**响应里没有 api_key**，只有它是否已设置。"""
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
    """存一个端点。

    带 ``id`` 是改那一条（可以改 label）；不带是按 label 认行。``api_key`` 为空时
    保留现有密钥。
    """
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

    api_key = str(payload.get("api_key") or "")
    with writable(request) as connection:
        rows = config_store.list_providers(connection, role)
        by_id = {int(row["id"]): row for row in rows}
        if provider_id is not None and provider_id not in by_id:
            raise HTTPException(404, f"{role} 端点 #{provider_id} 不存在")
        # 改名撞上另一条：拒绝，否则那一条会被覆盖掉。
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
                provider_id=provider_id,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
    return {"id": provider_id, "role": role, "label": label, "api_key_set": bool(api_key)}


@router.delete("/providers/{provider_id}")
def delete_provider(request: Request, provider_id: int) -> dict[str, Any]:
    with writable(request) as connection:
        removed = config_store.delete_provider(connection, provider_id)
    if not removed:
        raise HTTPException(404, f"端点 #{provider_id} 不存在")
    return {"deleted": removed}
