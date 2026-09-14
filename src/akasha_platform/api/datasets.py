"""数据集层与归一化层。

两层看的是同一批数据的两个形态，所以放一个路由里：

* 数据集层 —— **原始样例**，直接读 ``dataset/*.json``，不经库也不经适配器。
  这是「归一化之前长什么样」的唯一入口。
* 归一化层 —— 库里的 sample / corpus_doc，以及适配器状态。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from akasha_benchmark.datasets import all_adapters, get_adapter
from akasha_benchmark.datasets.resolver import DEFAULT_DATASET_DIR
from akasha_benchmark.io_utils import load_json
from akasha_benchmark.metrics import registry
from akasha_benchmark.stages import download
from akasha_benchmark.store import data_store

from ._common import db, writable

router = APIRouter(prefix="/api")

DEFAULT_PAGE = 20
MAX_PAGE = 200


@router.get("/datasets")
def datasets(request: Request) -> dict[str, Any]:
    """四组数据集的原始文件状态与归一化状态。

    适配器声明身份规则（哪个字段当 doc_id、gold 怎么解析）与它拥有哪些标注
    （provides）—— 后者决定哪些指标算得出来。分派只按数据集名，
    绝不按「row 里有没有某个字段」来猜：那样数据换版会静默走错分支，
    而症状是一个看着合理的指标，不是一个报错。
    """
    with db(request) as connection:
        normalized = {row["name"]: row for row in data_store.list_datasets(connection)}

    files: dict[str, list[dict[str, Any]]] = {}
    for entry in download.file_status():
        files.setdefault(str(entry["dataset"]), []).append(entry)

    entries = []
    for adapter in all_adapters():
        record = normalized.get(adapter.name)
        present = all(f["present"] and not f["error"] for f in files[adapter.name])
        entries.append(
            {
                "name": adapter.name,
                "adapter": type(adapter).__name__,
                "provides": sorted(d.value for d in adapter.provides),
                "identity_rules": adapter.identity_rules(),
                "expected_qa_rows": adapter.expected_qa_rows(),
                "files": files[adapter.name],
                "files_ready": present,
                "normalized": record is not None,
                "normalized_at": (record or {}).get("normalized_at"),
                "qa_rows": (record or {}).get("qa_rows"),
                "corpus_rows": (record or {}).get("corpus_rows"),
            }
        )
    return {"datasets": entries, "dataset_dir": str(DEFAULT_DATASET_DIR)}


@router.delete("/datasets/{name}")
def delete_dataset(request: Request, name: str) -> dict[str, Any]:
    """清理一个数据集的归一化产物。原始文件保留。"""
    with writable(request) as connection:
        try:
            deleted = data_store.delete_dataset(connection, name)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
    if not deleted:
        raise HTTPException(404, f"{name} 还没归一化")
    return {"deleted": deleted, "dataset": name}


@router.get("/datasets/{name}/raw")
def raw_samples(
    request: Request,
    name: str,
    kind: str = "qa",
    limit: int = Query(DEFAULT_PAGE, le=MAX_PAGE),
    offset: int = 0,
) -> dict[str, Any]:
    """**原始样例**，直接读数据文件。``kind`` 取 ``qa`` 或 ``corpus``。

    刻意不走适配器：这一栏要回答的是「上游给的是什么」，而适配器的产物
    已经是解释过一轮的结果。两者并排才看得出归一化做了什么。
    """
    if kind not in ("qa", "corpus"):
        raise HTTPException(422, "kind 必须是 qa 或 corpus")
    try:
        adapter = get_adapter(name)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    filename = adapter.qa_filename if kind == "qa" else adapter.corpus_filename
    path = DEFAULT_DATASET_DIR / filename
    if not path.is_file():
        raise HTTPException(404, f"{path.name} 缺失，请先在数据集页下载。")
    rows = load_json(path)
    if not isinstance(rows, list):
        raise HTTPException(500, f"{path.name}: 期望一个 JSON 数组")

    return {
        "dataset": adapter.name,
        "kind": kind,
        "source_file": filename,
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        # 原样返回，不裁字段 —— 裁了就看不到上游还有哪些字段没用上。
        "rows": rows[offset : offset + limit],
    }


@router.get("/datasets/{name}/samples")
def normalized_samples(
    request: Request,
    name: str,
    q: str | None = None,
    limit: int = Query(DEFAULT_PAGE, le=MAX_PAGE),
    offset: int = 0,
) -> dict[str, Any]:
    """归一化后的样本。``q`` 按问题文本或 sample_id 过滤。"""
    with db(request) as connection:
        if data_store.get_dataset(connection, name) is None:
            raise HTTPException(404, f"{name} 还没归一化")
        rows = data_store.samples_of(connection, name)

    if q:
        needle = q.strip().lower()
        rows = [
            r
            for r in rows
            if needle in r["question"].lower() or needle in r["sample_id"].lower()
        ]
    return {
        "dataset": name,
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "samples": rows[offset : offset + limit],
    }


@router.get("/datasets/{name}/samples/{sample_id}")
def normalized_sample(request: Request, name: str, sample_id: str) -> dict[str, Any]:
    """一条样本连同它的 gold 文档正文。

    gold 一起给，因为「这条样本的 gold 到底是什么」是判断标注质量的入口 ——
    归因层判 gold_annotation_suspect 时看的就是这个。
    """
    with db(request) as connection:
        sample = data_store.get_sample(connection, sample_id)
        if sample is None or sample["dataset"] != name:
            raise HTTPException(404, f"{name} 里没有样本 {sample_id!r}")
        gold, missing = [], []
        for doc_id in sample["gold_doc_ids"]:
            doc = data_store.corpus_doc(connection, name, doc_id)
            if doc is None:
                missing.append(doc_id)
            else:
                gold.append({**doc, "text": doc["text"][:4000]})
    return {
        **sample,
        "gold_docs": gold,
        # gold 指向语料里不存在的 doc_id 是身份规则出错的信号，必须报出来。
        "missing_gold_doc_ids": missing,
    }


@router.get("/datasets/{name}/corpus")
def normalized_corpus(
    request: Request,
    name: str,
    q: str | None = None,
    limit: int = Query(10, le=50),
    offset: int = 0,
) -> dict[str, Any]:
    """归一化后的语料。正文裁到可读长度。"""
    with db(request) as connection:
        if data_store.get_dataset(connection, name) is None:
            raise HTTPException(404, f"{name} 还没归一化")
        rows = data_store.corpus_of(connection, name)

    if q:
        needle = q.strip().lower()
        rows = [
            r for r in rows if needle in r["title"].lower() or needle in r["doc_id"].lower()
        ]
    return {
        "dataset": name,
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "docs": [
            {**r, "text": (r["text"] or "")[:2000], "truncated": len(r["text"] or "") > 2000}
            for r in rows[offset : offset + limit]
        ],
    }


@router.get("/metrics")
def metrics(request: Request, datasets: str | None = None) -> dict[str, Any]:
    """指标声明，以及给定数据集组合下哪些能勾。

    判据是数据集声明的 ``provides`` 与指标声明的 ``requires`` 做集合比对，
    不是数据集名字 —— 所以新增指标不必碰任何枚举。

    交集是「对所选每一组都算得出来」，并集是「至少一组能算」。勾了只对部分组
    成立的指标不会报错：缺依赖的那组会省略它，而不是伪造 0 分。
    """
    definitions = [
        {
            "name": d.name,
            "family": d.family,
            "requires": sorted(r.value for r in d.requires),
            "kind": d.kind,
            "higher_is_better": d.higher_is_better,
            "per_k": d.per_k,
            "description": d.description,
        }
        for d in registry.METRIC_DEFINITIONS
    ]

    names = [n.strip() for n in (datasets or "").split(",") if n.strip()]
    per_dataset: dict[str, list[str]] = {}
    for name in names:
        try:
            adapter = get_adapter(name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        per_dataset[adapter.name] = [d.name for d in registry.available(adapter.provides)]

    sets = [set(v) for v in per_dataset.values()]
    return {
        "definitions": definitions,
        "per_dataset": per_dataset,
        # UI 的默认勾选范围。
        "computable_for_all": sorted(set.intersection(*sets)) if sets else [],
        "computable_for_some": sorted(set.union(*sets)) if sets else [],
    }
