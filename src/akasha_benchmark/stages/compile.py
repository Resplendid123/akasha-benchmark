"""编译层：抽子集 -> 建空间 -> 导入 -> 编译 -> 质量闸门。

抽样顺序是先 QA 后 corpus，否则大部分 gold 会落在子集外：

    1. 固定种子抽 N 条 QA
    2. 这些 QA 的 gold 全集作为语料必选集
    3. 从剩余语料随机补负样本

narrativeqa 没有 gold 标注，改成整篇取文档，再取属于这些文档的问题。

一次编译一个空间，``run_id`` 固化这次的配置与模型快照。导入逐条提交，
所以暂停后继续时跳过已导入的文档。
"""

from __future__ import annotations

import random
import sqlite3
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from ..akasha_client import ACTIVE_RUN_STATUSES, AkashaClient, AkashaError
from ..config import AkashaConfig, load_config
from ..datasets import DATASET_NAMES, CorpusDoc, DataDependency, get_adapter
from ..store import compile_store, data_store, dumps, transaction
from ..task import TaskContext

DEFAULT_QA_LIMIT = 20
DEFAULT_NEGATIVES_RATIO = 1.0
DEFAULT_NARRATIVEQA_DOCS = 2


def default_seed() -> int:
    """当天日期，形如 20260908。同一天起的编译抽同一批。"""
    return int(datetime.now(UTC).strftime("%Y%m%d"))


def _largest_remainder(weights: dict[str, int], total: int) -> dict[str, int]:
    """按比例分配名额，各层之和精确等于 total。"""
    pool = sum(weights.values())
    if pool == 0:
        return dict.fromkeys(weights, 0)
    exact = {k: total * v / pool for k, v in weights.items()}
    floors = {k: int(v) for k, v in exact.items()}
    order = sorted(weights, key=lambda k: (-(exact[k] - floors[k]), k))
    for key in order[: total - sum(floors.values())]:
        floors[key] += 1
    return floors


def _stratified(
    samples: list[dict[str, Any]], limit: int, key: str, rng: random.Random
) -> list[dict[str, Any]]:
    """按 ``metadata[key]`` 分层抽样。musique 不分层的话几乎全是 2hop。"""
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        strata[str(sample["metadata"][key])].append(sample)

    quotas = _largest_remainder({k: len(v) for k, v in strata.items()}, limit)
    picked: list[dict[str, Any]] = []
    for name in sorted(strata):
        # 先排序再抽，结果因此不依赖查询返回顺序。
        available = sorted(strata[name], key=lambda s: s["sample_id"])
        picked.extend(rng.sample(available, min(quotas[name], len(available))))

    if len(picked) < limit:
        chosen = {s["sample_id"] for s in picked}
        rest = sorted(
            (s for s in samples if s["sample_id"] not in chosen), key=lambda s: s["sample_id"]
        )
        picked.extend(rng.sample(rest, min(limit - len(picked), len(rest))))
    return sorted(picked, key=lambda s: s["sample_id"])


def _safe_doc_id(doc_id: str) -> str:
    """拒掉不能当导入 filename 用的 doc_id。"""
    if not doc_id or doc_id in {".", ".."} or set(doc_id) & set('/\\:*?"<>|'):
        raise ValueError(f"doc_id {doc_id!r} 不能作为文件名")
    return doc_id


def build_subset(
    connection: sqlite3.Connection,
    compile_id: int,
    dataset: str,
    *,
    seed: int,
    qa_limit: int,
    negatives_ratio: float,
    narrativeqa_docs: int = DEFAULT_NARRATIVEQA_DOCS,
) -> dict[str, Any]:
    """抽一个数据集的子集写库，返回统计。"""
    adapter = get_adapter(dataset)
    if data_store.get_dataset(connection, adapter.name) is None:
        raise ValueError(f"{adapter.name} 还没归一化")

    samples = data_store.samples_of(connection, adapter.name)
    corpus = data_store.corpus_of(connection, adapter.name)
    by_id = {doc["doc_id"]: doc for doc in corpus}
    if not samples:
        raise ValueError(f"{adapter.name} 没有样本")

    # 随机源是 (数据集, seed) 而不含 run_id，同 seed 因此抽出同一批文档。
    rng = random.Random(f"{adapter.name}:{seed}")

    if adapter.has(DataDependency.GOLD_DOCS):
        if "hop_prefix" in samples[0]["metadata"]:
            strategy = "stratified_by_hop"
            picked = _stratified(samples, min(qa_limit, len(samples)), "hop_prefix", rng)
        else:
            strategy = "uniform_qa_then_gold_corpus"
            ordered = sorted(samples, key=lambda s: s["sample_id"])
            picked = sorted(
                rng.sample(ordered, min(qa_limit, len(ordered))), key=lambda s: s["sample_id"]
            )
        gold_ids = sorted({d for s in picked for d in s["gold_doc_ids"]})
        pool = sorted(set(by_id) - set(gold_ids))
        wanted = round(len(gold_ids) * negatives_ratio)
        negatives = sorted(rng.sample(pool, min(wanted, len(pool))))
        doc_ids = sorted(set(gold_ids) | set(negatives))
    else:
        # narrativeqa：整篇取文档，优先取 chunk 最少的（chunk 数决定编译成本）。
        strategy = "whole_documents"
        chunks: dict[str, list[str]] = defaultdict(list)
        for doc in corpus:
            chunks[doc["doc_id"].rsplit("_", 1)[0]].append(doc["doc_id"])
        ranked = sorted(chunks, key=lambda d: (len(chunks[d]), d))
        chosen = sorted(ranked[:narrativeqa_docs])
        doc_ids = sorted(
            (c for d in chosen for c in chunks[d]),
            key=lambda c: (c.rsplit("_", 1)[0], int(c.rsplit("_", 1)[1])),
        )
        picked = sorted(
            (s for s in samples if s["metadata"]["document_id"] in set(chosen)),
            key=lambda s: s["sample_id"],
        )
        if qa_limit and len(picked) > qa_limit:
            picked = sorted(rng.sample(picked, qa_limit), key=lambda s: s["sample_id"])
        gold_ids, negatives = [], []

    # 验收：每条样本的 gold 都要在子集语料内，否则 Recall 的上限不是 1。
    subset_ids = set(doc_ids)
    uncovered = {
        s["sample_id"]: sorted(set(s["gold_doc_ids"]) - subset_ids)
        for s in picked
        if set(s["gold_doc_ids"]) - subset_ids
    }
    if uncovered:
        raise RuntimeError(
            f"{adapter.name}: {len(uncovered)} 条样本的 gold 落在子集语料之外，"
            f"例如 {next(iter(uncovered.items()))}"
        )

    gold_set = set(gold_ids)
    docs = [{"doc_id": _safe_doc_id(doc_id), "is_gold": doc_id in gold_set} for doc_id in doc_ids]
    with transaction(connection):
        compile_store.replace_compile_subset(
            connection, compile_id, adapter.name, [s["sample_id"] for s in picked], docs
        )

    return {
        "dataset": adapter.name,
        "strategy": strategy,
        "samples": len(picked),
        "docs": len(doc_ids),
        "gold": len(gold_ids),
        "negatives": len(negatives),
    }


def markdown_of(connection: sqlite3.Connection, dataset: str, doc_id: str) -> str:
    """渲染导入 Akasha 的正文。heading 承担 title，文件名承担 doc_id。"""
    doc = data_store.corpus_doc(connection, dataset, doc_id)
    if doc is None:
        raise ValueError(f"{dataset}/{doc_id} 不在语料里")
    return CorpusDoc(doc_id=doc_id, title=doc["title"], text=doc["text"]).to_markdown()


def _wait_for_compile(
    ctx: TaskContext,
    client: AkashaClient,
    space_id: str,
    config: AkashaConfig,
    *,
    expect_runs: int = 0,
) -> dict[str, Any]:
    """轮询到全部编译 Run 终态，返回 ``{status_counts, no_runs, timed_out}``。

    空的 ``statusCounts`` 与「全部终态」分开报：两者的 ``active`` 都是 0，
    但前者是压根没编译。``expect_runs`` 为 0 且看不到 Run 时立即返回。
    """
    deadline = time.monotonic() + config.poll_timeout_seconds
    while True:
        ctx.checkpoint()
        summary = client.run_diagnostics_summary([space_id])
        counts: dict[str, int] = summary.get("statusCounts") or {}
        active = sum(n for name, n in counts.items() if name in ACTIVE_RUN_STATUSES)
        if counts and active == 0:
            return {"status_counts": counts, "no_runs": False, "timed_out": False}
        if not counts and expect_runs == 0:
            return {"status_counts": counts, "no_runs": True, "timed_out": False}
        if time.monotonic() > deadline:
            return {"status_counts": counts, "no_runs": not counts, "timed_out": True}
        ctx.log(f"编译中：{counts}" if counts else "等待编译 Run 出现")
        time.sleep(config.poll_interval_seconds)


def _page_failures(client: AkashaClient, space_id: str) -> list[str]:
    """按 errorCode 聚合失败页面，逐条描述成一行。"""
    try:
        log = client.page_log([space_id])
    except AkashaError as exc:
        return [f"读逐页日志失败：HTTP {exc.status}"]

    entries = log.get("items") or []
    failed = [e for e in entries if e.get("status") == "failed"]
    if not failed:
        return []

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for entry in failed:
        key = (
            str(entry.get("errorCode") or "unknown"),
            str(entry.get("errorSummary") or ""),
        )
        grouped[key].append(str(entry.get("title") or entry.get("sourcePageId") or "?"))

    lines = []
    for (code, summary), titles in sorted(grouped.items()):
        sample = "、".join(titles[:3]) + ("…" if len(titles) > 3 else "")
        lines.append(f"{len(titles)} 篇 {code}：{summary}（例如 {sample}）")
    return lines


def _compile_pace(client: AkashaClient, space_id: str) -> dict[str, Any] | None:
    """每篇编译耗时的估算，取 Run 的墙钟时长除以页数。

    是估算而不是实测：单篇的起止时间拿不到，并发度大于 1 时这个值偏高。
    """
    try:
        report = client.run_diagnostics([space_id])
    except AkashaError:
        return None

    runs = report.get("items") or []
    if not runs:
        return None
    total_ms = sum(int(r.get("runDurationMs") or 0) for r in runs)
    pages = sum(int((r.get("progress") or {}).get("text", {}).get("expected") or 0) for r in runs)
    if not pages or not total_ms:
        return None
    return {
        "runs": len(runs),
        "pages": pages,
        "total_ms": total_ms,
        "per_page_ms": round(total_ms / pages),
    }


def _quality_gate(client: AkashaClient, space_id: str) -> dict[str, Any]:
    """入库完整性闸门。要求四项计数都拿到值且都为 0，取不到值不算通过。"""
    report = client.quality_diagnostics([space_id])
    summary = report.get("summary") or {}
    gates = {
        name: summary.get(name)
        for name in (
            "missingChunkPageCount",
            "missingEmbeddingPageCount",
            "missingSourcePageCount",
            "stalePageCount",
        )
    }
    passed = all(isinstance(v, int) and v == 0 for v in gates.values())
    return {"passed": passed, "gates": gates}


def run(ctx: TaskContext) -> None:
    params = ctx.params
    datasets = list(params.get("datasets") or [])
    if not datasets:
        raise ValueError("请至少选择一个数据集")
    unknown = sorted(set(datasets) - set(DATASET_NAMES))
    if unknown:
        raise ValueError(f"未知数据集：{unknown}")

    seed = int(params["seed"]) if params.get("seed") is not None else default_seed()
    qa_limit = int(params.get("qa_limit") or DEFAULT_QA_LIMIT)
    negatives_ratio = float(params.get("negatives_ratio", DEFAULT_NEGATIVES_RATIO))
    if qa_limit < 1:
        raise ValueError("每个数据集的 QA 数必须大于 0")
    if negatives_ratio < 0:
        raise ValueError("负样本比例不能为负")

    config = load_config(ctx.db)
    config.require_credentials()

    compile_id = ctx.target("compile")
    run_id = str(params.get("run_id") or "").strip() or f"run{uuid.uuid4().hex[:10]}"
    ctx.freeze(
        run_id=run_id,
        datasets=datasets,
        seed=seed,
        qa_limit=qa_limit,
        negatives_ratio=negatives_ratio,
    )
    if compile_id is None:
        if compile_store.compile_run_by_run_id(ctx.db, run_id):
            raise ValueError(f"编译名称 {run_id!r} 已存在，请换个名称或继续原任务")
        compile_id = compile_store.create_compile_run(
            ctx.db,
            run_id=run_id,
            datasets=datasets,
            seed=seed,
            qa_limit=qa_limit,
            negatives_ratio=negatives_ratio,
        )
        ctx.bind("compile", compile_id)
    ctx.log(f"编译 {run_id}（#{compile_id}），数据集 {', '.join(datasets)}")

    _execute(ctx, compile_id, run_id, datasets, seed, qa_limit, negatives_ratio, config)


def _execute(
    ctx: TaskContext,
    compile_id: int,
    run_id: str,
    datasets: list[str],
    seed: int,
    qa_limit: int,
    negatives_ratio: float,
    config: AkashaConfig,
) -> None:
    # 已抽过子集就不重抽：重抽会让已导入文档的 page_id 指向不在子集里的文档，
    # 而那种错配不报错，只会让每个检索指标都算错。
    if not compile_store.compile_docs(ctx.db, compile_id):
        for dataset in datasets:
            ctx.checkpoint()
            # 编译三段共用一把刻度（抽子集 0、导入 1、等编译 2、完成 3），
            # 免得各段用各自的总量让进度条来回跳。细节放 note 里。
            ctx.progress(0, 3, f"抽子集 {dataset}")
            stats = build_subset(
                ctx.db,
                compile_id,
                dataset,
                seed=seed,
                qa_limit=qa_limit,
                negatives_ratio=negatives_ratio,
            )
            ctx.log(
                f"{dataset}: QA {stats['samples']}，语料 {stats['docs']} "
                f"(gold {stats['gold']} / 负样本 {stats['negatives']})，策略 {stats['strategy']}"
            )
    else:
        ctx.log("子集已存在，跳过抽样")

    with AkashaClient(config) as client:
        client.login()
        me = client.current_user()
        user = (me or {}).get("user") or {}
        workspace = (me or {}).get("workspace") or {}
        # 非 owner 会在授权闸门静默丢弃 chunk，症状看起来像召回质量差。
        if user.get("role") != "owner":
            raise RuntimeError(
                f"当前账号角色是 {user.get('role')!r}，不是 owner。"
                "非 owner 会在授权闸门静默丢弃 chunk，入库前请提权。"
            )

        record = compile_store.get_compile_run(ctx.db, compile_id) or {}
        space_id = record.get("space_id")
        if not space_id:
            # 随机 slug，把这次编译的空间与用户自己的空间分开。
            slug = f"bench{uuid.uuid4().hex[:16]}"
            space = client.create_space(
                name=f"bench {run_id}"[:100],
                slug=slug,
                description=f"Akasha-Benchmark {run_id}. 自动生成，可安全删除。",
            )
            space_id = space.get("id")
            if not space_id:
                raise RuntimeError(f"创建空间未返回 id：{space!r}")
            compile_store.update_compile_run(
                ctx.db,
                compile_id,
                space_id=space_id,
                space_name=slug,
                workspace_id=workspace.get("id"),
                model_configs_json=dumps(client.get_model_configs()),
            )
            ctx.db.commit()
            ctx.log(f"创建空间 {slug}（{space_id}）")
        else:
            # 在发出任何写入之前拦住：换了账号或部署之后，已记下的 page_id 全部失效。
            mismatch = compile_store.workspace_mismatch(ctx.db, compile_id, workspace.get("id"))
            if mismatch:
                raise RuntimeError(mismatch)
            ctx.log(f"复用本次编译的空间 {space_id}")

        _import_docs(ctx, client, compile_id, space_id)

        # 等编译的总量拿不到，用「导入完 = 2/3」这一档，收尾时推到 3/3。
        ctx.progress(2, 3, "等待编译")
        result = client.compile_spaces([space_id])
        accepted = int(result.get("acceptedRunCount") or 0)
        coalesced = int(result.get("coalescedRunCount") or 0)
        ctx.log(f"已请求编译：accepted={accepted} coalesced={coalesced}")
        wait = _wait_for_compile(ctx, client, space_id, config, expect_runs=accepted + coalesced)
        if wait["no_runs"]:
            # 一个 Run 都没有：page 停在「已上传、未编译」，没有源文本也没有 chunk。
            raise RuntimeError(
                f"编译没有启动：Akasha 一个编译 Run 都没有（accepted={accepted} "
                f"coalesced={coalesced}）。{len(compile_store.compile_docs(ctx.db, compile_id))} "
                "篇语料已上传但没被编译，请检查 Akasha 的编译 worker 是否在跑。"
            )
        if wait["timed_out"]:
            raise RuntimeError(
                f"编译轮询超时（{config.poll_timeout_seconds}s），"
                f"状态 {wait['status_counts']}。此时查询会得到偏低但不报错的指标。"
            )
        ctx.log(f"编译终态：{wait['status_counts']}")

        pace = _compile_pace(client, space_id)
        if pace:
            compile_store.update_compile_run(ctx.db, compile_id, pace_json=dumps(pace))
            ctx.log(
                f"编译节奏（估算）：{pace['pages']} 篇用 "
                f"{pace['total_ms'] / 1000:.1f}s，约 {pace['per_page_ms'] / 1000:.1f}s/篇"
            )

        quality = _quality_gate(client, space_id)
        compile_store.update_compile_run(ctx.db, compile_id, quality_json=dumps(quality))
        ctx.db.commit()
        ctx.log(f"质量闸门 {quality['gates']} -> {'通过' if quality['passed'] else '未通过'}")
        if not quality["passed"]:
            # 闸门报的是后果，原因要从逐页日志取。
            reasons = _page_failures(client, space_id)
            for line in reasons:
                ctx.log(f"编译失败原因：{line}", "error")
            detail = f"；{reasons[0]}" if reasons else ""
            raise RuntimeError(f"编译质量闸门未通过：半成品索引产出的指标没有意义{detail}")

    ctx.progress(3, 3, "编译完成")
    ctx.log("编译完成，可以进入查询层")


def _import_docs(ctx: TaskContext, client: AkashaClient, compile_id: int, space_id: str) -> None:
    """串行导入未导入的文档，逐条提交。失败的记下来并继续下一篇。"""
    pending = compile_store.compile_docs(ctx.db, compile_id, pending_only=True)
    total = len(compile_store.compile_docs(ctx.db, compile_id))
    done = total - len(pending)
    if not pending:
        ctx.log(f"{total} 篇语料均已导入，跳过")
        return

    ctx.log(f"导入语料：待导入 {len(pending)} / 共 {total}")
    failures = 0
    for doc in pending:
        ctx.checkpoint()
        markdown = markdown_of(ctx.db, doc["dataset"], doc["doc_id"])
        try:
            page = client.import_page_text(f"{doc['doc_id']}.md", markdown, space_id)
            page_id = (page or {}).get("id")
            error = None if page_id else f"响应里没有 page id: {page!r}"[:300]
        except AkashaError as exc:
            page_id, error = None, f"HTTP {exc.status}: {exc.body[:200]}"

        compile_store.record_page(
            ctx.db, compile_id, doc["dataset"], doc["doc_id"], page_id=page_id, error=error
        )
        ctx.db.commit()
        done += 1
        if error:
            failures += 1
            ctx.log(f"{doc['dataset']}/{doc['doc_id']}: {error}", "warn")
        if done % 10 == 0 or done == total:
            ctx.progress(1, 3, f"导入语料 {done}/{total}")

    if failures:
        raise RuntimeError(f"{failures} 篇语料导入失败，page_map 不完整")
