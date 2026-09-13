"""数据集层与归一化层。

两层看的是同一批数据的两个形态，所以放一个路由里：

* 数据集层 —— **原始样例**，直接读 ``dataset/*.json``，不经库。
  这是「归一化之前长什么样」的唯一入口。
* 归一化层 —— 库里的 ``sample`` / ``corpus_doc``，以及适配器实现状态。
  没有适配器的源无法入库，那一项要在这里说清楚。
"""

from __future__ import annotations

from typing import Any

from akasha_benchmark.datasets import DATASET_NAMES, all_adapters, get_adapter
from akasha_benchmark.datasets.resolver import DEFAULT_DATASET_DIR
from akasha_benchmark.io_utils import load_json
from akasha_benchmark.metrics import registry
from akasha_benchmark.store import repo
from fastapi import APIRouter, HTTPException, Query, Request

from .._common_types import DEFAULT_PAGE, MAX_PAGE
from ._common import db, strip_json, writable

router = APIRouter(prefix="/api")


@router.delete("/datasets/{name}")
def delete_dataset(request: Request, name: str) -> dict[str, Any]:
    with writable(request) as connection:
        try:
            deleted = repo.delete_dataset(connection, name)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if not deleted:
            raise HTTPException(404, f"{name} has not been normalized yet")
    return {"deleted": deleted, "dataset": name}


@router.get("/datasets")
def datasets(request: Request) -> list[dict[str, Any]]:
    """库里的归一化记录。没归一化过的数据集不在这里 —— 见 ``/adapters``。"""
    with db(request) as connection:
        rows = repo.list_datasets(connection)
    return [
        {
            **strip_json(row),
            "provides": repo.loads(row["provides_json"], []),
            "identity_rules": repo.loads(row["identity_rules_json"], {}),
            "dedup_stats": repo.loads(row["dedup_stats_json"], {}),
            "gold_count_distribution": repo.loads(row["gold_count_distribution_json"], {}),
        }
        for row in rows
    ]


@router.get("/adapters")
def adapters(request: Request) -> dict[str, Any]:
    """适配器实现状态 —— 归一化层的闸门视图。

    **没有适配器就无法入库。** 这不是「暂时缺个功能」：原始数据的身份规则
    （哪个字段当 doc_id、gold 怎么解析）必须逐组核对，猜不出来 —— 按字段
    存在性去猜的话，数据换个版本就会静默走错分支，而症状是一个看着挺合理的
    指标，不是一个报错。

    「其他源」不是一份预设清单，而是**扫 ``dataset/`` 里没有适配器认领的文件**。
    从实际扫出来才有意义：预设清单只会列出写代码时想到的那几个。
    """
    with db(request) as connection:
        normalized = {row["name"]: row for row in repo.list_datasets(connection)}

    claimed: set[str] = set()
    entries: list[dict[str, Any]] = []
    for adapter in all_adapters():
        qa_path = DEFAULT_DATASET_DIR / adapter.qa_filename
        corpus_path = DEFAULT_DATASET_DIR / adapter.corpus_filename
        claimed.update({adapter.qa_filename, adapter.corpus_filename})
        record = normalized.get(adapter.name)
        files_present = qa_path.is_file() and corpus_path.is_file()
        entries.append(
            {
                "name": adapter.name,
                "aliases": list(adapter.aliases),
                "adapter": type(adapter).__name__,
                "adapter_version": adapter.version,
                "implemented": True,
                "provides": sorted(d.value for d in adapter.provides),
                "files_present": files_present,
                "qa_file": adapter.qa_filename,
                "corpus_file": adapter.corpus_filename,
                "expected_qa_rows": adapter.expected_qa_rows(),
                "normalized": record is not None,
                "normalized_at": (record or {}).get("normalized_at"),
                "qa_rows": (record or {}).get("qa_rows"),
                "corpus_rows": (record or {}).get("corpus_rows"),
                "blocked_reason": (
                    None
                    if files_present
                    else "原始文件缺失，请在「数据集」页点击下载数据集"
                ),
            }
        )

    markdown_dir = DEFAULT_DATASET_DIR / "markdown"
    entries.append(
        {
            "name": "markdown-docs",
            "aliases": [],
            "adapter": "待接入",
            "adapter_version": "—",
            "implemented": False,
            "provides": [],
            "files_present": markdown_dir.is_dir() and any(markdown_dir.rglob("*.md")),
            "qa_file": "待定",
            "corpus_file": "markdown/**/*.md",
            "expected_qa_rows": None,
            "normalized": False,
            "normalized_at": None,
            "qa_rows": None,
            "corpus_rows": None,
            "blocked_reason": "已预留 Markdown 文档包适配器；待确定 QA 数据格式后接入。",
        }
    )

    unclaimed = (
        sorted(
            path.name
            for path in DEFAULT_DATASET_DIR.glob("*.json")
            if path.name not in claimed
        )
        if DEFAULT_DATASET_DIR.is_dir()
        else []
    )

    return {
        "adapters": entries,
        # dataset/ 里没有适配器认领的文件。它们**无法入库**。
        "unclaimed_files": unclaimed,
        "dataset_dir": str(DEFAULT_DATASET_DIR),
        "note": (
            "没有适配器的源无法入库。适配器要声明身份规则（哪个字段当 doc_id、"
            "gold 怎么解析）与它拥有哪些标注（provides）—— 后者决定哪些指标算得出来。"
            "分派只按数据集名字，绝不按「row 里有没有某个字段」来猜。"
        ),
    }


@router.get("/datasets/{name}/raw")
def raw_samples(
    request: Request,
    name: str,
    limit: int = Query(DEFAULT_PAGE, le=MAX_PAGE),
    offset: int = 0,
) -> dict[str, Any]:
    """**原始样例**，直接读数据文件，不经库也不经适配器。

    刻意不走适配器：这一栏要回答的是「上游给的是什么」，而适配器的产物
    已经是解释过一轮的结果。两者并排才看得出归一化做了什么。
    """
    try:
        adapter = get_adapter(name)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    qa_path = DEFAULT_DATASET_DIR / adapter.qa_filename
    if not qa_path.is_file():
        raise HTTPException(
            404,
            f"{qa_path.name} 缺失，请在「数据集」页点击下载数据集。",
        )

    rows = load_json(qa_path)
    if not isinstance(rows, list):
        raise HTTPException(500, f"{qa_path.name}: expected a JSON array")

    window = rows[offset : offset + limit]
    return {
        "dataset": adapter.name,
        "source_file": adapter.qa_filename,
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        # 原样返回，不裁字段 —— 裁了就看不到「上游还有哪些字段没用上」。
        "rows": window,
        "note": (
            "这是原始文件里的行，未经适配器解释。归一化后的形态见 "
            f"/api/datasets/{adapter.name}/samples。"
        ),
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
        record = repo.get_dataset(connection, name)
        if record is None:
            raise HTTPException(
                404, f"{name} has not been normalized yet; run the normalize stage first"
            )
        rows = repo.samples_of(connection, name)

    if q:
        needle = q.strip().lower()
        rows = [
            row
            for row in rows
            if needle in row["question"].lower() or needle in row["sample_id"].lower()
        ]

    return {
        "dataset": name,
        "adapter": record["adapter"],
        "adapter_version": record["adapter_version"],
        "provides": repo.loads(record["provides_json"], []),
        "identity_rules": repo.loads(record["identity_rules_json"], {}),
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "samples": rows[offset : offset + limit],
    }


@router.get("/datasets/{name}/corpus")
def normalized_corpus(
    request: Request,
    name: str,
    q: str | None = None,
    limit: int = Query(20, le=100),
    offset: int = 0,
) -> dict[str, Any]:
    """归一化后的语料。正文裁到可读长度，全文走编译层的 diff 视图。"""
    with db(request) as connection:
        if repo.get_dataset(connection, name) is None:
            raise HTTPException(404, f"{name} has not been normalized yet")
        rows = repo.corpus_of(connection, name)

    if q:
        needle = q.strip().lower()
        rows = [
            row
            for row in rows
            if needle in row["title"].lower() or needle in row["doc_id"].lower()
        ]

    window = [
        {**row, "text": (row["text"] or "")[:2000], "text_truncated": len(row["text"] or "") > 2000}
        for row in rows[offset : offset + limit]
    ]
    return {
        "dataset": name,
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "docs": window,
    }


@router.get("/datasets/{name}/samples/{sample_id}")
def normalized_sample(request: Request, name: str, sample_id: str) -> dict[str, Any]:
    """一条归一化样本，连同它的 gold 文档正文。

    gold 一起给，因为「这条样本的 gold 到底是什么」是判断标注质量的入口 ——
    归因层判 ``gold_annotation_suspect`` 时要看的就是这个。
    """
    with db(request) as connection:
        matches = [s for s in repo.samples_of(connection, name) if s["sample_id"] == sample_id]
        if not matches:
            raise HTTPException(404, f"no sample {sample_id!r} in {name}")
        sample = matches[0]
        gold = [
            {
                **doc,
                "text": (doc["text"] or "")[:4000],
                "text_truncated": len(doc["text"] or "") > 4000,
            }
            for doc_id in sample["gold_doc_ids"]
            if (doc := repo.corpus_doc(connection, name, doc_id)) is not None
        ]
        missing = [
            doc_id
            for doc_id in sample["gold_doc_ids"]
            if repo.corpus_doc(connection, name, doc_id) is None
        ]

    return {
        **sample,
        "gold_docs": gold,
        # gold 指向语料里不存在的 doc_id 是身份规则出错的信号，必须报出来。
        "missing_gold_doc_ids": missing,
    }


@router.get("/metrics/definitions")
def metric_definitions() -> list[dict[str, Any]]:
    """指标声明。前端靠 ``requires`` 知道某个数据集该不该显示某一列，
    靠 ``higher_is_better`` 决定排序方向（``truncation_loss`` 是越低越好）。"""
    return [
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


@router.get("/metrics/available")
def available_metrics(request: Request, datasets: str | None = None) -> dict[str, Any]:
    """给定数据集组合，哪些指标可勾选。**评测层的勾选范围由它决定。**

    判据是数据集声明的 ``provides`` 与指标声明的 ``requires`` 做集合比对，
    不是数据集名字。所以新增指标不必碰任何枚举。

    交集与并集都给：交集是「对所选的每一组都算得出来」，并集是「至少一组能算」。
    UI 默认用交集勾选，用并集提示「这几项只对部分组有效」。
    """
    names = [n.strip() for n in (datasets or "").split(",") if n.strip()] or list(DATASET_NAMES)
    per_dataset: dict[str, list[str]] = {}
    provides_map: dict[str, list[str]] = {}
    for name in names:
        try:
            adapter = get_adapter(name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        per_dataset[adapter.name] = [d.name for d in registry.available(adapter.provides)]
        provides_map[adapter.name] = sorted(d.value for d in adapter.provides)

    sets = [set(v) for v in per_dataset.values()]
    intersection = sorted(set.intersection(*sets)) if sets else []
    union = sorted(set.union(*sets)) if sets else []

    return {
        "datasets": names,
        "provides": provides_map,
        "per_dataset": per_dataset,
        # 对所选的每一组都算得出来。UI 的默认勾选范围。
        "computable_for_all": intersection,
        # 至少一组能算。勾了这里面、不在交集里的项，报告会对缺依赖的组省略。
        "computable_for_some": union,
        "partial": sorted(set(union) - set(intersection)),
        "note": (
            "勾了只对部分组成立的指标不会报错：缺依赖的那一组会省略它并写明原因，"
            "而不是伪造 0 分。"
        ),
    }
