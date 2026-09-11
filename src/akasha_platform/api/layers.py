"""编译层与评测层。

分层依据是模型配置的重编译语义（§12.3）：compiler / embedding 改了必须重编译
（索引层，约 15 小时 / 1722 篇），answer 改了不用（查询层，10-14 秒每条），
指标与 k 是纯离线的（评测层）。

UI 上把索引层叫「编译层」—— 用户关心的是「这批文档编译成什么样了」，
而抽样与导入是达成它的手段。库里的表名不动，那是另一回事。
"""

from __future__ import annotations

from typing import Any

from akasha_benchmark.store import repo
from fastapi import APIRouter, Body, HTTPException, Query, Request

from .._common_types import DEFAULT_PAGE, MAX_PAGE
from ._common import db, strip_json, writable

router = APIRouter(prefix="/api")


def _ingest_identity(layer) -> dict[str, Any]:
    """这一层**入库时**跑在什么上。**不含密钥。**

    取自 ``connection_json``（ingest 写入的 redacted 快照）与 ``workspace_id``
    （从 users/me 解析）。它们记的是当时的值 —— 配置改过之后这里仍然是历史真相,
    而那正是 workspace 闸门要比对的东西。
    """
    snapshot = repo.loads(layer["connection_json"], {}) or {}
    return {
        "ingested_at": layer["ingested_at"],
        "base_url": snapshot.get("base_url"),
        "email": snapshot.get("email"),
        "workspace_id": layer["workspace_id"],
        "workspace_name": layer["workspace_name"],
        "akasha_user_role": layer["akasha_user_role"],
    }


@router.get("/layers")
def layers(request: Request) -> dict[str, Any]:
    """三层树。总览与各层的选择器都读它。"""
    with db(request) as connection:
        index_layers = []
        for layer in repo.list_index_layers(connection):
            layer_id = int(layer["id"])
            readiness = repo.index_layer_readiness(connection, layer_id)
            index_layers.append(
                {
                    **strip_json(layer),
                    "model_configs": repo.loads(layer["model_configs_json"]),
                    "datasets": repo.index_layer_datasets(connection, layer_id),
                    "page_map_counts": repo.page_map_counts(connection, layer_id),
                    "ready_for_query": readiness["ready"],
                    "not_ready_reasons": readiness["reasons"],
                    # 这一层入库时跑在什么上。配置改过之后它仍然是历史真相。
                    "ingest_identity": _ingest_identity(layer),
                    # 同抽样配置的其他层：「同子集换 embedding」的对照实验靠它找同伴。
                    "same_subset_layers": [
                        int(other["id"])
                        for other in repo.index_layers_by_subset_hash(
                            connection, layer["subset_hash"]
                        )
                        if int(other["id"]) != layer_id
                    ],
                    "query_layers": [
                        {
                            **strip_json(q),
                            "stats": repo.response_stats(connection, int(q["id"])),
                            "request_window": repo.request_window(connection, int(q["id"])),
                            "eval_layers": [
                                strip_json(e)
                                for e in repo.list_eval_layers(connection, int(q["id"]))
                            ],
                        }
                        for q in repo.list_query_layers(connection, layer_id)
                    ],
                }
            )
    return {"index_layers": index_layers}


@router.get("/layers/index/{layer_id}")
def index_layer(request: Request, layer_id: int) -> dict[str, Any]:
    with db(request) as connection:
        layer = repo.get_index_layer(connection, layer_id)
        if layer is None:
            raise HTTPException(404, f"no index layer #{layer_id}")
        return {
            **strip_json(layer),
            "model_configs": repo.loads(layer["model_configs_json"]),
            "connection": repo.loads(layer["connection_json"]),
            "datasets": repo.index_layer_datasets(connection, layer_id),
            "page_map_counts": repo.page_map_counts(connection, layer_id),
            "import_failures": repo.import_failures(connection, layer_id),
            "quality_gate": repo.latest_quality_gate(connection, layer_id),
            "compile_run": repo.latest_compile_run(connection, layer_id),
            "readiness": repo.index_layer_readiness(connection, layer_id),
            "ingest_identity": _ingest_identity(layer),
        }


@router.post("/layers/index/{layer_id}/discard-ingest")
def discard_ingest(
    request: Request, layer_id: int, payload: dict[str, Any] = Body(default={})
) -> dict[str, Any]:
    """清掉一层的入库产物，让它退回「已抽子集、未入库」。

    什么时候需要：连接配置改到了另一个 workspace（换账号 / 换部署），而这一层的
    ``page_map`` 与 ``space_id`` 只在原来那个里有意义。那时 ingest 与 query 都会
    被 workspace 闸门拦住，而唯一的出路是「承认之前那次入库不要了」。

    **远端的 space 不删** —— 我们不删别人的数据。它们留在那边，配置改回去
    还能复用。子集也不动：它是离线抽的，与连接无关。

    ``confirm`` 必须显式为真：重新入库要烧掉整批编译时间，不该被一次误点触发。
    """
    with writable(request) as connection:
        layer = repo.get_index_layer(connection, layer_id)
        if layer is None:
            raise HTTPException(404, f"no index layer #{layer_id}")
        if not layer["ingested_at"]:
            raise HTTPException(409, f"index layer #{layer_id} has no ingest to discard")

        total = sum(repo.page_map_counts(connection, layer_id).values())
        if not payload.get("confirm"):
            # 按实测的约 40 秒/篇估重新入库的代价。那是 Akasha 的 BullMQ worker
            # 吞吐，客户端调不动 —— 所以这个数字是真实的等待时间。
            hours = total * 40 / 3600
            raise HTTPException(
                409,
                f"index layer #{layer_id} has {total} page_map row(s) from its ingest on "
                f"{layer['ingested_at']}. Discarding drops page_map, space bindings, the "
                f"quality gate and the compile run; re-ingesting will take roughly "
                f"{hours:.1f}h of compilation. The subset is kept and remote spaces are "
                "left untouched. Pass confirm=true to proceed.",
            )

        discarded = repo.discard_ingest(connection, layer_id)
        connection.commit()
        readiness = repo.index_layer_readiness(connection, layer_id)

    return {
        "index_layer_id": layer_id,
        "discarded": discarded,
        "readiness": readiness,
        "note": (
            "已清掉这一层的入库产物，它退回「已抽子集、未入库」。"
            "子集保留；远端的 space 没有删。"
        ),
    }


@router.get("/layers/index/{layer_id}/docs")
def index_layer_docs(
    request: Request,
    layer_id: int,
    dataset: str | None = None,
    gold_only: bool = False,
    q: str | None = None,
    limit: int = Query(DEFAULT_PAGE, le=MAX_PAGE),
    offset: int = 0,
) -> dict[str, Any]:
    """本层已导入的文档，带 ``page_id`` —— **编译层 diff 视图的入口**。

    ``page_id`` 是走血缘链路的钥匙：有它才能查到这篇编成了哪些 artifact、
    参与召回的 chunk 是什么。没导入成功的文档 ``page_id`` 为 None，
    它们在这里照样列出（缺篇本身是要看的信息）。
    """
    with db(request) as connection:
        if repo.get_index_layer(connection, layer_id) is None:
            raise HTTPException(404, f"no index layer #{layer_id}")
        names = (
            [dataset]
            if dataset
            else [row["dataset"] for row in repo.index_layer_datasets(connection, layer_id)]
        )

        docs: list[dict[str, Any]] = []
        for name in names:
            page_map = repo.page_to_doc(connection, layer_id, name)
            doc_to_page = {doc: page for page, doc in page_map.items()}
            for doc in repo.subset_docs(connection, layer_id, name, with_text=False):
                if gold_only and not doc["is_gold"]:
                    continue
                docs.append(
                    {
                        "dataset": name,
                        "doc_id": doc["doc_id"],
                        "is_gold": bool(doc["is_gold"]),
                        "md_sha256": doc["md_sha256"],
                        "page_id": doc_to_page.get(doc["doc_id"]),
                    }
                )

    if q:
        needle = q.strip().lower()
        docs = [d for d in docs if needle in d["doc_id"].lower()]

    return {
        "index_layer_id": layer_id,
        "total": len(docs),
        "imported": sum(1 for d in docs if d["page_id"]),
        "offset": offset,
        "limit": limit,
        "docs": docs[offset : offset + limit],
    }


@router.get("/layers/index/{layer_id}/docs/{dataset}/{doc_id}")
def index_layer_doc(
    request: Request, layer_id: int, dataset: str, doc_id: str
) -> dict[str, Any]:
    """一篇文档导入 Akasha 的正文，以及它的 ``page_id``。

    正文取自库里的 ``subset_doc.md_text`` —— 那是**实际上传的那份**，
    不是从原始语料现渲染的。两者应当一致，而如果不一致，这里显示的是真相。
    """
    with db(request) as connection:
        docs = repo.subset_docs(connection, layer_id, dataset)
        match = next((d for d in docs if d["doc_id"] == doc_id), None)
        if match is None:
            raise HTTPException(404, f"no doc {doc_id!r} in layer #{layer_id}/{dataset}")
        page_map = repo.page_to_doc(connection, layer_id, dataset)
        page_id = next((p for p, d in page_map.items() if d == doc_id), None)

    return {
        "index_layer_id": layer_id,
        "dataset": dataset,
        "doc_id": doc_id,
        "is_gold": bool(match["is_gold"]),
        "md_text": match["md_text"],
        "md_sha256": match["md_sha256"],
        "page_id": page_id,
        "diff_url": f"/api/diff/{page_id}" if page_id else None,
        "note": (
            None
            if page_id
            else "这篇没有 page_id：导入失败或还没导。编译产物的 diff 因此看不了。"
        ),
    }


@router.get("/layers/query/{query_layer_id}/responses")
def query_responses(
    request: Request,
    query_layer_id: int,
    dataset: str | None = None,
    answer_mode: str | None = None,
    limit: int = Query(DEFAULT_PAGE, le=MAX_PAGE),
    offset: int = 0,
) -> dict[str, Any]:
    """一个查询层的生成结果。**评测层看「问题的生成结果响应」就是这里。**

    按 answerMode 给计数：``no_match`` / ``general`` 无条件返回空
    retrievedSources，它们的检索得分按定义为 0，混在一起读会把生成端拒答
    误当成检索失败。
    """
    with db(request) as connection:
        layer = repo.get_query_layer(connection, query_layer_id)
        if layer is None:
            raise HTTPException(404, f"no query layer #{query_layer_id}")
        names = [dataset] if dataset else repo.response_datasets(connection, query_layer_id)

        rows: list[dict[str, Any]] = []
        for name in names:
            for row in repo.responses_of(connection, query_layer_id, name):
                body = row.get("response") or {}
                mode = body.get("answerMode") if isinstance(body, dict) else None
                if answer_mode and mode != answer_mode:
                    continue
                rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "dataset": name,
                        "question": row["question"],
                        "answer_mode": mode,
                        "http_status": row["http_status"],
                        "error": row["error"],
                        "latency_ms": row["latency_ms"],
                        "requested_at": row["requested_at"],
                        "answer": (
                            (body.get("answer") or "")[:600] if isinstance(body, dict) else None
                        ),
                        "retrieved_count": len(
                            body.get("retrievedSources") or [] if isinstance(body, dict) else []
                        ),
                        "citation_count": len(
                            body.get("citations") or [] if isinstance(body, dict) else []
                        ),
                    }
                )

        counts: dict[str, int] = {}
        for row in rows:
            key = row["answer_mode"] or "missing"
            counts[key] = counts.get(key, 0) + 1

    return {
        "query_layer_id": query_layer_id,
        "label": layer["label"],
        "total": len(rows),
        "count_by_answer_mode": counts,
        "offset": offset,
        "limit": limit,
        "responses": rows[offset : offset + limit],
    }


@router.get("/layers/query/{query_layer_id}/responses/{sample_id}")
def query_response(request: Request, query_layer_id: int, sample_id: str) -> dict[str, Any]:
    """单条完整响应体。存的是完整的，不是当下用得到的那几个字段。"""
    with db(request) as connection:
        row = repo.response_of(connection, query_layer_id, sample_id)
        if row is None:
            raise HTTPException(404, f"no response for {sample_id!r} in layer #{query_layer_id}")
        audit = next(
            (
                a
                for a in repo.audit_records(connection, query_layer_id)
                if a["sample_id"] == sample_id
            ),
            None,
        )
    return {
        **row,
        # retrievalDiagnostics 不在 HTTP 响应里，只在审计表 —— 抄进库的那份在这。
        "audit": audit,
    }


@router.get("/layers/eval/{eval_layer_id}")
def eval_layer(request: Request, eval_layer_id: int) -> dict[str, Any]:
    """一个评测层的全部汇总，按数据集与 scope 组织。"""
    with db(request) as connection:
        layer = repo.get_eval_layer(connection, eval_layer_id)
        if layer is None:
            raise HTTPException(404, f"no eval layer #{eval_layer_id}")
        query_layer = repo.get_query_layer(connection, int(layer["query_layer_id"]))

        summaries: dict[str, dict[str, dict[str, float]]] = {}
        for row in repo.metric_summaries(connection, eval_layer_id):
            summaries.setdefault(row["dataset"], {}).setdefault(row["scope"], {})[
                row["metric"]
            ] = row["value"]

        return {
            **strip_json(layer),
            "ks": repo.loads(layer["ks_json"], []),
            # 这一轮勾了哪些指标。报告页的列以它为准。
            "metrics": repo.loads(layer["metrics_json"], []),
            "query_layer": strip_json(query_layer or {}),
            "datasets": [
                {
                    **strip_json(row),
                    "missing_responses": repo.loads(row["missing_responses_json"], []),
                    "unmapped_page_ids": repo.loads(row["unmapped_page_ids_json"], []),
                    # 算不了的指标连同原因一起给前端，**不伪造 0 分**。
                    "omitted_metrics": repo.loads(row["omitted_metrics_json"], []),
                    "answer_mode_distribution": repo.loads(
                        row["answer_mode_distribution_json"], {}
                    ),
                    "stratified": repo.loads(row["stratified_json"]),
                    "scopes": summaries.get(row["dataset"], {}),
                }
                for row in repo.dataset_evals(connection, eval_layer_id)
            ],
            "judge": repo.judge_summary(connection, eval_layer_id, "faithfulness"),
            "badcase_causes": repo.badcase_cause_counts(connection, eval_layer_id),
        }
