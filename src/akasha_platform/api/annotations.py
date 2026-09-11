"""标注与 judge-human 一致率。

``level='sample'`` 的标注**跨 run 继承** —— 样本层是资产，另两层是笔记（决策 14）。
``author_kind`` 区分 human 与 model，两者同表，所以一致率是一个 GROUP BY
就能算出来的免费产物（§12.5）。它是判断「这个 LLM 归因能不能信」的唯一办法。
"""

from __future__ import annotations

from typing import Any

from akasha_benchmark.store import repo
from fastapi import APIRouter, Body, HTTPException, Request

from ._common import db, writable

router = APIRouter(prefix="/api")


@router.post("/annotations")
def add_annotation(request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    level = payload.get("level")
    if level not in {"sample", "query_layer", "eval_layer"}:
        raise HTTPException(422, "level must be sample, query_layer or eval_layer")
    author_kind = payload.get("author_kind", "human")
    if author_kind not in {"human", "model"}:
        raise HTTPException(422, "author_kind must be human or model")
    target_id = str(payload.get("target_id") or "").strip()
    if not target_id:
        raise HTTPException(422, "target_id is required")

    with writable(request) as connection:
        annotation_id = repo.add_annotation(
            connection,
            level=level,
            target_id=target_id,
            author_kind=author_kind,
            author=str(payload.get("author") or "anonymous"),
            labels=list(payload.get("labels") or []),
            note=payload.get("note"),
            # §12.6：记 source 与 confidence，这样任何指标结果都能追溯到
            # 「它依赖的 gold 有多少是人确认过的」。
            source=str(payload.get("source") or author_kind),
            confidence=payload.get("confidence"),
        )
        connection.commit()
        return {"id": annotation_id}


@router.get("/annotations/agreement")
def agreement(request: Request, level: str = "sample") -> dict[str, Any]:
    """judge/归因 与人工的一致率。"""
    with db(request) as connection:
        rows = repo.label_agreement(connection, level)
    matched = sum(r["exact_match"] for r in rows)
    return {
        "level": level,
        "compared": len(rows),
        "exact_matches": matched,
        "agreement": matched / len(rows) if rows else None,
        "rows": rows,
        "note": "只统计 human 与 model 都标过的目标；单边标注无从比较。",
    }


@router.get("/annotations/{level}/{target_id}")
def annotations(request: Request, level: str, target_id: str) -> list[dict[str, Any]]:
    with db(request) as connection:
        return [
            {**{k: v for k, v in a.items() if k != "labels_json"},
             "labels": repo.loads(a["labels_json"], [])}
            for a in repo.annotations_for(connection, level, target_id)
        ]


@router.delete("/annotations/{annotation_id}")
def delete_annotation(request: Request, annotation_id: int) -> dict[str, Any]:
    with writable(request) as connection:
        removed = repo.delete_annotation(connection, annotation_id)
        connection.commit()
    if not removed:
        raise HTTPException(404, f"no annotation #{annotation_id}")
    return {"deleted": removed}
