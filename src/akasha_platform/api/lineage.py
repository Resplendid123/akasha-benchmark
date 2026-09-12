"""血缘与原文/编译 diff —— 归因层的链路视图。

六跳链路：page -> artifact -> chunk -> 图边 -> 原文。**必须一屏走完**,
否则归因就得像那次一样手写 SQL。

原文 vs 编译产物的并排 diff 是**一等视图**，不是附属功能：那次 recall@5 = 0.5
的根因只有把两者并排才看得见 —— 问题问的是 "who won Grammy and Emmy award",
而编译产物里这两个词已经没了，于是三条召回路径同时断。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from .. import diff as diff_mod
from ..lineage import BadPageId, LineageUnavailable
from ._common import config_of, db, reader
from .samples import sample_detail_of

router = APIRouter(prefix="/api")


@router.get("/lineage/{page_id}")
def lineage(request: Request, page_id: str) -> dict[str, Any]:
    """一个 page 的完整链路：artifact -> chunk -> 图边 -> 原文。"""
    try:
        result = reader(request).lineage(page_id)
    except BadPageId as exc:
        raise HTTPException(400, str(exc)) from exc
    except LineageUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return {**result, "queried": _queried_note(request)}


@router.get("/diff/{page_id}")
def raw_vs_compiled(request: Request, page_id: str, question: str = "") -> dict[str, Any]:
    """原文 vs 编译产物并排。**编译层的「查看文档经过的变化」就是这里。**"""
    try:
        result = reader(request).lineage(page_id)
    except BadPageId as exc:
        raise HTTPException(400, str(exc)) from exc
    except LineageUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return {**diff_mod.build(result, question), "queried": _queried_note(request)}


def _queried_note(request: Request) -> dict[str, Any]:
    """这次查的是哪个部署的库。

    空结果与「连错了库」在响应上长得一样，所以这一项不是装饰 —— 它是区分
    「这篇确实没编译产物」和「配置指向了别处」的唯一线索。
    """
    return {"base_url": config_of(request).base_url}


@router.get("/layers/eval/{eval_layer_id}/samples/{sample_id}/lineage")
def sample_lineage(request: Request, eval_layer_id: int, sample_id: str) -> dict[str, Any]:
    """一条样本的全部 gold 文档的血缘 + diff。

    这是「失败样本 -> gold doc_id -> page_id -> artifact -> chunk -> 图边 -> 原文」
    那条路径的成品：给定样本，直接把每篇 gold 走完。
    """
    with db(request) as connection:
        detail = sample_detail_of(connection, eval_layer_id, sample_id)
    question = detail.get("question") or ""
    chain_reader = reader(request)

    per_doc: list[dict[str, Any]] = []
    for doc_id, page_id in (detail.get("gold_pages") or {}).items():
        if not page_id:
            per_doc.append({"doc_id": doc_id, "page_id": None, "error": "not in page_map"})
            continue
        try:
            chain = chain_reader.lineage(page_id)
        except BadPageId as exc:
            raise HTTPException(400, str(exc)) from exc
        except LineageUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        per_doc.append(
            {
                "doc_id": doc_id,
                "page_id": page_id,
                "lineage": chain,
                "diff": diff_mod.build(chain, question),
            }
        )

    return {
        "sample_id": sample_id,
        "question": question,
        "answer_mode": detail.get("answer_mode"),
        "metrics": detail.get("metrics"),
        "gold": per_doc,
    }
