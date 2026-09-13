"""配置路由：Akasha 连接、远端模型配置及本地评估/归因模型。"""

from __future__ import annotations

from typing import Any

from akasha_benchmark.akasha_client import AkashaClient, AkashaError
from akasha_benchmark.config import sanitize_updates
from akasha_benchmark.store import identity, repo
from fastapi import APIRouter, Body, HTTPException, Request

from .._common_types import MODEL_FEATURES
from ._common import config_of, db, strip_json, writable

router = APIRouter(prefix="/api")


# ------------------------------------------------------------ Akasha 连接


@router.get("/connection")
def get_connection(request: Request) -> dict[str, Any]:
    """那一份连接配置，以及上次测连接的结果。"""
    with db(request) as connection:
        row = repo.get_connection_row(connection)
        layers = repo.ingested_layers(connection)

    stored = dict(row) if row is not None else {}
    return {
        **{
            k: v
            for k, v in stored.items()
            if not k.endswith("_json")
        },
        # 上次测连接时那份 model_configs 的快照。
        "last_model_configs": repo.loads(stored.get("last_model_configs_json")),
        # 已入库的层，连同它们各自落在哪个 workspace。改 base_url / email 之前
        # 该看一眼：改到另一个部署会让它们跑不了。
        "ingested_layers": layers,
    }


@router.put("/connection")
def put_connection(request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """改连接配置。**只写** —— payload 里没出现的字段保持原值，显式传空串即清空。
    password / database_url 与 email / base_url 同款语义。

    workspace 漂移这类副作用提示不在这里出 —— 改完之后 ingest / query 在登录时
    会按相同的判据拦住，让那一步自己说话更直接。
    """
    try:
        fields = sanitize_updates(payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    with writable(request) as connection:
        repo.update_connection(connection, **fields)
        connection.commit()

    return {
        "updated": sorted(fields),
        "connection": get_connection(request),
    }


@router.post("/connection/test")
def test_connection(request: Request) -> dict[str, Any]:
    """登录 + 取当前用户 + 拉模型配置，结果存回库里。

    OWNER 那一项不是可选检查。非 OWNER 会在第三道授权闸门**静默丢弃 chunk**，
    症状看起来像召回质量差而不是一个错误 —— 在这里报出来，比在跑完一批编译
    之后从指标里猜出来便宜得多。
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
        _record(request, ok=False, role=None, model_configs=None)
        raise HTTPException(502, f"Akasha rejected the request: {exc}") from exc
    except OSError as exc:
        _record(request, ok=False, role=None, model_configs=None)
        raise HTTPException(502, f"cannot reach {config.base_url}: {exc}") from exc

    user = (me or {}).get("user") or {}
    workspace = (me or {}).get("workspace") or {}
    role = user.get("role")
    resolved = str(workspace.get("id") or "")
    _record(request, ok=True, role=role, model_configs=model_configs)

    # 拿刚解析出的 workspace 去比已入库的层。这是一次**预检** —— 同一道判据
    # ingest / query 在登录后也会走，但在这里先说出来，用户不必等起了任务才发现。
    with db(request) as connection:
        blocked = []
        for row in repo.ingested_layers(connection):
            message = repo.workspace_mismatch(connection, int(row["id"]), resolved)
            if message:
                blocked.append({"id": int(row["id"]), "label": row["label"], "reason": message})

    return {
        "ok": True,
        "user": {"id": user.get("id"), "email": user.get("email"), "role": role},
        # workspace 由服务端决定（自建部署走 workspaceRepo.findFirst()），
        # 所以这是只读信息，不是一个待填的配置项。
        "workspace": {"id": resolved, "name": workspace.get("name")},
        "is_owner": role == "owner",
        "owner_warning": (
            None
            if role == "owner"
            else (
                f"这个账号的角色是 {role!r}，不是 owner。非 owner 会在授权闸门静默"
                "丢弃 chunk，症状看起来像召回质量差。入库前请提权。"
            )
        ),
        "blocked_layers": blocked,
        "model_configs": model_configs,
    }


def _record(request: Request, **fields: Any) -> None:
    with writable(request) as connection:
        repo.record_connection_check(connection, **fields)
        connection.commit()


# --------------------------------------------------- Akasha 侧的模型配置


@router.get("/model-configs")
def get_model_configs(request: Request) -> dict[str, Any]:
    """Akasha 的 compiler / embedding / answer / image 四项配置。

    同时给出各索引层的快照比对结果，这样「现在的配置与这一层入库时是否一致」
    在一个响应里就能看出来。
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
        raise HTTPException(502, f"cannot reach {config.base_url}: {exc}") from exc

    with db(request) as connection:
        layers = [
            {
                "id": int(layer["id"]),
                "label": layer["label"],
                "ingested_at": layer["ingested_at"],
                "embedding_matches": identity.embedding_matches(
                    live, repo.loads(layer["model_configs_json"])
                ),
                "compiler_matches": identity.compiler_matches(
                    live, repo.loads(layer["model_configs_json"])
                ),
            }
            for layer in repo.list_index_layers(connection)
            if layer["model_configs_json"]
        ]

    return {
        "features": list(MODEL_FEATURES),
        "live": live,
        "index_layers": layers,

    }


@router.put("/model-configs/{feature}")
def put_model_config(
    request: Request, feature: str, payload: dict[str, Any] = Body(...)
) -> dict[str, Any]:
    """改 Akasha 的某一项模型配置。**改的是那个部署，不是本平台。**

    影响说明在响应里一起给，UI 负责在提交前弹确认。这里不做「二次确认」的
    状态机 —— 那种设计会让 API 变成有状态的，而真正需要拦住用户的地方是界面。
    """
    if feature not in MODEL_FEATURES:
        raise HTTPException(422, f"unknown feature {feature!r}; known: {list(MODEL_FEATURES)}")

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
        raise HTTPException(502, f"cannot reach {config.base_url}: {exc}") from exc

    requires_rebuild = feature in {"compiler", "embedding"}
    return {
        "feature": feature,
        "result": result,
        "requires_new_index_layer": requires_rebuild,
        "impact": "需重编译" if requires_rebuild else "无影响",
    }


# ------------------------------------------------------------ 模型 provider


@router.get("/providers")
@router.get("/providers/{role}")
def list_providers(request: Request, role: str | None = None) -> list[dict[str, Any]]:
    """judge / analysis 端点配置。**响应里没有 api_key**，只有它是否已设置。"""
    if role is not None and role not in {"judge", "analysis"}:
        raise HTTPException(422, "role must be judge or analysis")
    with db(request) as connection:
        rows = repo.list_model_providers(connection, role)
    return [
        {
            **{k: v for k, v in strip_json(row).items() if k not in {"api_key", "api_key_env"}},
            "params": repo.loads(row["params_json"], {}),
            "api_key_set": bool((row["api_key"] or "").strip()),
        }
        for row in rows
    ]


@router.put("/providers/{role}")
def put_provider(
    request: Request, role: str, payload: dict[str, Any] = Body(...)
) -> dict[str, Any]:
    """存一个 judge / analysis 端点。

    ``api_key`` 为空时保留现有密钥；连接密码的空值语义不同。
    """
    if role not in {"judge", "analysis"}:
        raise HTTPException(422, "role must be judge or analysis")
    label = str(payload.get("label") or "default").strip()
    base_url = str(payload.get("base_url") or "").strip()
    model = str(payload.get("model") or "").strip()
    if not base_url or not model:
        raise HTTPException(422, "base_url and model are required")

    api_key = str(payload.get("api_key") or "")

    with writable(request) as connection:
        existing = repo.get_model_provider(connection, role, label)
        if not api_key and existing:
            api_key = existing["api_key"]
        provider_id = repo.upsert_model_provider(
            connection,
            role=role,
            label=label,
            base_url=base_url,
            model=model,
            api_key=api_key,
            params={
                "temperature": float(payload.get("temperature", 0.0)),
                "max_tokens": int(payload.get("max_tokens", 1024)),
            },
        )
        connection.commit()
    return {"id": provider_id, "role": role, "label": label, "api_key_set": bool(api_key.strip())}


@router.delete("/providers/{provider_id}")
def delete_provider(request: Request, provider_id: int) -> dict[str, Any]:
    with writable(request) as connection:
        removed = repo.delete_model_provider(connection, provider_id)
        connection.commit()
    if not removed:
        raise HTTPException(404, f"no provider #{provider_id}")
    return {"deleted": removed}
