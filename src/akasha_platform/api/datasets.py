"""数据集层与归一化层。同一批数据的两个形态，所以放一个路由里：

* 数据集层 —— 原始样例，直接读 ``dataset/*.json``，不经库也不经适配器。
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


def _contains_text(value: Any, needle: str) -> bool:
    """递归搜索原始 JSON 的值；字段名不属于数据内容，不参与匹配。"""
    if isinstance(value, dict):
        return any(_contains_text(item, needle) for item in value.values())
    if isinstance(value, list):
        return any(_contains_text(item, needle) for item in value)
    return value is not None and needle in str(value).lower()


@router.get("/datasets")
def datasets(request: Request) -> dict[str, Any]:
    """各组数据集的原始文件状态与归一化状态，连同各自的身份规则与 provides。"""
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
                "subset_strategy": adapter.subset_strategy.value,
                # 假就是本地数据集：下载按钮跳过它，缺文件要手动放进 dataset/。
                "downloadable": adapter.downloadable,
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
    q: str | None = None,
    limit: int = Query(DEFAULT_PAGE, ge=1, le=MAX_PAGE),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """原始样例，直接读数据文件，``kind`` 取 ``qa`` 或 ``corpus``。

    不走适配器：这一栏答的是「上游给的是什么」，与归一化产物并排才看得出差别。
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
    needle = (q or "").strip().lower()
    if needle:
        rows = [row for row in rows if _contains_text(row, needle)]

    return {
        "dataset": adapter.name,
        "kind": kind,
        "source_file": filename,
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        # 原样返回，不裁字段，否则看不到上游还有哪些字段没用上。
        "rows": rows[offset : offset + limit],
    }


@router.get("/datasets/{name}/samples")
def normalized_samples(
    request: Request,
    name: str,
    q: str | None = None,
    limit: int = Query(DEFAULT_PAGE, ge=1, le=MAX_PAGE),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """归一化后的样本。``q`` 按问题文本、sample_id 或参考答案过滤。"""
    with db(request) as connection:
        if data_store.get_dataset(connection, name) is None:
            raise HTTPException(404, f"{name} 还没归一化")
        total, rows = data_store.sample_page(
            connection,
            name,
            search=(q or "").strip() or None,
            limit=limit,
            offset=offset,
        )
    return {
        "dataset": name,
        "total": total,
        "offset": offset,
        "limit": limit,
        "samples": rows,
    }


@router.get("/datasets/{name}/samples/{sample_id}")
def normalized_sample(request: Request, name: str, sample_id: str) -> dict[str, Any]:
    """一条样本连同它的 gold 文档正文，供判断标注质量。"""
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
        # 非空即身份规则出错：gold 指向了语料里不存在的 doc_id。
        "missing_gold_doc_ids": missing,
    }


@router.get("/datasets/{name}/corpus")
def normalized_corpus(
    request: Request,
    name: str,
    q: str | None = None,
    limit: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """归一化后的语料。正文按数据库内容原样返回。"""
    with db(request) as connection:
        if data_store.get_dataset(connection, name) is None:
            raise HTTPException(404, f"{name} 还没归一化")
        total, rows = data_store.corpus_page(
            connection,
            name,
            search=(q or "").strip() or None,
            limit=limit,
            offset=offset,
        )
    return {
        "dataset": name,
        "total": total,
        "offset": offset,
        "limit": limit,
        "docs": rows,
    }


@router.get("/metrics")
def metrics(request: Request, datasets: str | None = None) -> dict[str, Any]:
    """指标声明，以及给定数据集组合下哪些能勾。

    ``computable_for_all`` 是交集（每一组都算得出来），
    ``computable_for_some`` 是并集。勾了只对部分组成立的指标不报错，
    缺依赖的那组省略它。
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
