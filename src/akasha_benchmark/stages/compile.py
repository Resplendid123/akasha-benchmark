"""编译层：抽子集 -> 建空间 -> 导入 -> 编译 -> 质量闸门。

抽样顺序是先 QA 后 corpus，否则大部分 gold 会落在子集外：

    1. 固定种子抽 N 条 QA
    2. 这些 QA 的 gold 全集作为语料必选集
    3. 从剩余语料随机补负样本

没有 gold 标注的两组走别的路，各自由适配器的 ``subset_strategy`` 声明：
narrativeqa 整篇取文档再取属于这些文档的问题；itfaq 没有指回文档的字段，语料整份导入。

一次编译一个空间，``run_id`` 固化这次的配置与模型快照。导入逐条提交，
所以暂停后继续时跳过已导入的文档。
"""

from __future__ import annotations

import queue
import random
import re
import sqlite3
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import httpx

from ..akasha_client import ACTIVE_RUN_STATUSES, AkashaClient, AkashaError
from ..config import AkashaConfig, load_config
from ..datasets import (
    DATASET_NAMES,
    CorpusDoc,
    DataDependency,
    SubsetStrategy,
    get_adapter,
)
from ..store import compile_store, config_store, data_store, dumps, transaction
from ..task import Paused, TaskContext

DEFAULT_QA_LIMIT = 20
DEFAULT_NEGATIVES_RATIO = 1.0
DEFAULT_NARRATIVEQA_DOCS = 2
DEFAULT_IMPORT_CONCURRENCY = 4
MAX_IMPORT_CONCURRENCY = 16
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MARKDOWN_HEADING = re.compile(r"^\s*#+\s*.*$", re.MULTILINE)


def default_seed() -> int:
    """当天日期，形如 20260915。同一天起的编译抽同一批。"""
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


def _has_indexable_text(text: str) -> bool:
    """判断正文是否除标题、图片外仍有可编译文本。"""
    body = _MARKDOWN_IMAGE.sub("", text or "")
    body = _MARKDOWN_HEADING.sub("", body)
    return bool(body.strip())


def build_subset(
    connection: sqlite3.Connection,
    compile_id: int,
    dataset: str,
    *,
    seed: int,
    qa_limit: int,
    negatives_ratio: float,
    full_corpus: bool = False,
    narrativeqa_docs: int = DEFAULT_NARRATIVEQA_DOCS,
) -> dict[str, Any]:
    """抽一个数据集的子集写库，返回统计。"""
    adapter = get_adapter(dataset)
    if data_store.get_dataset(connection, adapter.name) is None:
        raise ValueError(f"{adapter.name} 还没归一化")

    samples = data_store.samples_of(connection, adapter.name)
    corpus = data_store.corpus_of(connection, adapter.name)
    by_id = {doc["doc_id"]: doc for doc in corpus}
    usable_doc_ids = {
        doc_id for doc_id, doc in by_id.items() if _has_indexable_text(doc["text"])
    }
    if not samples:
        raise ValueError(f"{adapter.name} 没有样本")

    # 随机源是 (数据集, seed) 而不含 run_id，同 seed 因此抽出同一批文档。
    rng = random.Random(f"{adapter.name}:{seed}")

    strategy = adapter.subset_strategy
    gold_strategies = (SubsetStrategy.QA_THEN_GOLD, SubsetStrategy.STRATIFIED_HOP)
    # 声明要 gold 而数据集没有标注时报错，不静默产出空 gold 集。
    if strategy in gold_strategies and not adapter.has(DataDependency.GOLD_DOCS):
        raise ValueError(f"{adapter.name}: 策略 {strategy.value!r} 需要 gold 标注，但本组没有")

    if full_corpus:
        # 显式全量语料：QA 仍受 qa_limit 控制，语料不再按 gold/负样本抽样。
        doc_ids = sorted(usable_doc_ids)
        picked = sorted(
            rng.sample(
                sorted(samples, key=lambda s: s["sample_id"]), min(qa_limit, len(samples))
            ),
            key=lambda s: s["sample_id"],
        )
        gold_ids = sorted({d for s in picked for d in s["gold_doc_ids"]})
        negatives = (
            sorted(set(doc_ids) - set(gold_ids))
            if adapter.has(DataDependency.GOLD_DOCS)
            else []
        )
    elif strategy is SubsetStrategy.FULL_CORPUS:
        # itfaq：没有指回文档的字段，抽语料就无法保证被抽到的问题还答得上。
        # 语料整份导入，所以 negatives_ratio 对这一组不起作用。
        doc_ids = sorted(usable_doc_ids)
        picked = sorted(
            rng.sample(
                sorted(samples, key=lambda s: s["sample_id"]), min(qa_limit, len(samples))
            ),
            key=lambda s: s["sample_id"],
        )
        gold_ids, negatives = [], []
    elif strategy is SubsetStrategy.WHOLE_DOCS:
        # narrativeqa：整篇取文档，优先取 chunk 最少的（chunk 数决定编译成本）。
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
    else:
        if strategy is SubsetStrategy.STRATIFIED_HOP:
            picked = _stratified(samples, min(qa_limit, len(samples)), "hop_prefix", rng)
        else:
            ordered = sorted(samples, key=lambda s: s["sample_id"])
            picked = sorted(
                rng.sample(ordered, min(qa_limit, len(ordered))), key=lambda s: s["sample_id"]
            )
        gold_ids = sorted({d for s in picked for d in s["gold_doc_ids"]})
        pool = sorted(usable_doc_ids - set(gold_ids))
        wanted = round(len(gold_ids) * negatives_ratio)
        negatives = sorted(rng.sample(pool, min(wanted, len(pool))))
        doc_ids = sorted(set(gold_ids) | set(negatives))

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
        "strategy": strategy.value,
        "samples": len(picked),
        "docs": len(doc_ids),
        "empty_docs": len(by_id) - len(usable_doc_ids),
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
    coalesced_runs: int = 0,
    expected_run_ids: set[str] | None = None,
    baseline_run_ids: set[str] | None = None,
    baseline_sequence: int = 0,
) -> dict[str, Any]:
    """轮询到全部编译 Run 终态，返回 ``{status_counts, no_runs, timed_out}``。

    空的 ``statusCounts`` 与「全部终态」分开报：两者的 ``active`` 都是 0，
    但前者是压根没编译。``expect_runs`` 为 0 且看不到 Run 时立即返回。
    """
    deadline = time.monotonic() + config.poll_timeout_seconds
    baseline_run_ids = baseline_run_ids or set()
    expected_run_ids = expected_run_ids or set()
    while True:
        ctx.checkpoint()
        runs: list[dict[str, Any]] = []
        try:
            runs = list(client.run_diagnostics([space_id], limit=50).get("items") or [])
            diagnostics_ok = True
        except AkashaError:
            runs = []
            diagnostics_ok = False
        current = [
            run
            for run in runs
            if str(run.get("runId") or run.get("id") or "") in expected_run_ids
            or (
                (str(run.get("runId") or run.get("id")) not in baseline_run_ids)
                if run.get("runId") or run.get("id")
                else int(run.get("spaceJobSequence") or 0) > baseline_sequence
            )
        ]
        # coalesced 请求复用已有 Run；此时取当前空间最新的一条，而不是把历史 Run 合计。
        if (
            not current
            and coalesced_runs
            and runs
            and any(run.get("runId") or run.get("id") or run.get("spaceJobSequence") for run in runs)
        ):
            current = [max(runs, key=lambda r: int(r.get("spaceJobSequence") or 0))]

        if current:
            counts: dict[str, int] = {}
            for run in current:
                status = str(run.get("status") or "unknown")
                counts[status] = counts.get(status, 0) + 1
            progress = {
                key: sum(int((run.get("progress") or {}).get("text", {}).get(key) or 0) for run in current)
                for key in ("expected", "succeeded", "failed", "skipped")
            }
            active = sum(n for name, n in counts.items() if name in ACTIVE_RUN_STATUSES)
        elif expect_runs and diagnostics_ok and any(
            run.get("runId") or run.get("id") or run.get("spaceJobSequence") for run in runs
        ):
            # 已知空间历史 Run，但本次 accepted Run 尚未出现在诊断列表中；继续等，
            # 不能用包含历史 Run 的 summary 提前结束。
            counts = {}
            progress = {}
            active = 1
        else:
            # 兼容旧服务/测试替身没有 Run 明细的情况；生产服务走上面的按 Run 过滤。
            summary = client.run_diagnostics_summary([space_id])
            counts = summary.get("statusCounts") or {}
            progress = {}
            active = sum(n for name, n in counts.items() if name in ACTIVE_RUN_STATUSES)

        if current and progress.get("expected"):
            ctx.progress(
                progress["succeeded"] + progress["skipped"] + progress["failed"],
                progress["expected"],
                "编译语料",
            )
        elif counts:
            ctx.progress(0, None, "等待远端编译")
        if counts and active == 0:
            return {
                "status_counts": counts,
                "no_runs": False,
                "timed_out": False,
                "runs": current,
                "progress": progress,
            }
        if not counts and expect_runs == 0:
            return {"status_counts": counts, "no_runs": True, "timed_out": False}
        if time.monotonic() > deadline:
            return {"status_counts": counts, "no_runs": not counts, "timed_out": True}
        if not counts:
            ctx.progress(0, None, "等待远端编译")
        time.sleep(config.poll_interval_seconds)


def _page_failures(client: AkashaClient, space_id: str) -> list[str]:
    """按 errorCode 聚合失败页面，逐条描述成一行。"""
    try:
        log = client.page_log([space_id])
    except AkashaError as exc:
        return [f"读逐页日志失败：HTTP {exc.status}"]

    entries = log.get("items") or []
    failed = [e for e in entries if e.get("status") in {"failed", "skipped"}]
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


def _compile_pace(
    client: AkashaClient, space_id: str, runs: list[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    """每篇编译耗时的估算，取 Run 的墙钟时长除以页数。

    是估算而不是实测：单篇的起止时间拿不到，并发度大于 1 时这个值偏高。
    """
    if runs is None:
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


def _progress_of(runs: list[dict[str, Any]]) -> dict[str, int]:
    return {
        key: sum(int((run.get("progress") or {}).get("text", {}).get(key) or 0) for run in runs)
        for key in ("expected", "succeeded", "failed", "skipped")
    }


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
    full_corpus = bool(params.get("full_corpus", False))
    import_concurrency = int(params.get("import_concurrency") or DEFAULT_IMPORT_CONCURRENCY)
    if qa_limit < 1:
        raise ValueError("每个数据集的 QA 数必须大于 0")
    if negatives_ratio < 0:
        raise ValueError("负样本比例不能为负")
    if not 1 <= import_concurrency <= MAX_IMPORT_CONCURRENCY:
        raise ValueError(f"导入并发必须在 1 到 {MAX_IMPORT_CONCURRENCY} 之间")

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
        full_corpus=full_corpus,
        import_concurrency=import_concurrency,
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

    _execute(
        ctx,
        compile_id,
        run_id,
        datasets,
        seed,
        qa_limit,
        negatives_ratio,
        full_corpus,
        import_concurrency,
        config,
    )


def _execute(
    ctx: TaskContext,
    compile_id: int,
    run_id: str,
    datasets: list[str],
    seed: int,
    qa_limit: int,
    negatives_ratio: float,
    full_corpus: bool,
    import_concurrency: int,
    config: AkashaConfig,
) -> None:
    # 已抽过子集就不重抽：重抽会让已导入文档的 page_id 指向不在子集里的文档，
    # 而那种错配不报错，只会让每个检索指标都算错。
    if not compile_store.compile_docs(ctx.db, compile_id):
        for dataset in datasets:
            ctx.checkpoint()
            ctx.progress(0, None, f"抽子集 {dataset}")
            stats = build_subset(
                ctx.db,
                compile_id,
                dataset,
                seed=seed,
                qa_limit=qa_limit,
                negatives_ratio=negatives_ratio,
                full_corpus=full_corpus,
            )
            ctx.log(
                f"{dataset}: QA {stats['samples']}，语料 {stats['docs']} "
                f"(gold {stats['gold']} / 负样本 {stats['negatives']})，策略 {stats['strategy']}"
                + (f"，跳过无正文 {stats['empty_docs']} 篇" if stats["empty_docs"] else "")
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
            group = config_store.selected_config_group(ctx.db)
            compile_store.update_compile_run(
                ctx.db,
                compile_id,
                space_id=space_id,
                space_name=slug,
                workspace_id=workspace.get("id"),
                config_group=group["label"] if group else None,
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

        _import_docs(ctx, client, compile_id, space_id, import_concurrency)

        total_docs = len(compile_store.compile_docs(ctx.db, compile_id))
        ctx.progress(0, total_docs, "等待编译")
        try:
            before = client.run_diagnostics([space_id], limit=50).get("items") or []
        except AkashaError:
            before = []
        baseline_run_ids = {
            str(run.get("runId") or run.get("id")) for run in before if run.get("runId") or run.get("id")
        }
        baseline_sequence = max((int(run.get("spaceJobSequence") or 0) for run in before), default=0)
        previous_run_ids = [str(value) for value in ctx.params.get("remote_compile_run_ids") or []]
        previous_runs = [
            run
            for run in before
            if str(run.get("runId") or run.get("id") or "") in set(previous_run_ids)
        ]
        active_previous = [
            run
            for run in previous_runs
            if str(run.get("status") or "") in ACTIVE_RUN_STATUSES
        ]

        ctx.checkpoint()
        if active_previous:
            remote_run_ids = {
                str(run.get("runId") or run.get("id")) for run in active_previous
            }
            accepted, coalesced = len(remote_run_ids), 0
            ctx.log(f"接管仍在运行的远端编译：{', '.join(sorted(remote_run_ids))}")
            wait = _wait_for_compile(
                ctx,
                client,
                space_id,
                config,
                expect_runs=len(remote_run_ids),
                expected_run_ids=remote_run_ids,
            )
        elif previous_run_ids:
            failed_page_ids = client.retryable_run_page_ids(previous_run_ids)
            if failed_page_ids:
                result = client.retry_pages(failed_page_ids)
                remote_run_ids = {str(value) for value in result.get("jobIds") or []}
                if not remote_run_ids:
                    raise RuntimeError(
                        f"Akasha 已接收 {len(failed_page_ids)} 篇失败页面，但没有返回重试 Run ID"
                    )
                ctx.freeze(remote_compile_run_ids=sorted(remote_run_ids))
                ctx.log(
                    f"批量重试 {len(failed_page_ids)} 篇失败页面，"
                    f"远端 Run {', '.join(sorted(remote_run_ids))}"
                )
                if ctx.pause_requested:
                    for remote_run_id in remote_run_ids:
                        client.cancel_compile_run(
                            remote_run_id,
                            "Akasha-Benchmark task paused during retry submission",
                        )
                    ctx.checkpoint()
                accepted, coalesced = len(remote_run_ids), 0
                wait = _wait_for_compile(
                    ctx,
                    client,
                    space_id,
                    config,
                    expect_runs=len(remote_run_ids),
                    expected_run_ids=remote_run_ids,
                )
            elif previous_runs and all(
                str(run.get("status") or "") == "cancelled"
                and not int((run.get("progress") or {}).get("text", {}).get("expected") or 0)
                for run in previous_runs
            ):
                # Run 在初始化逐页记录前就被取消；retry-pages 只接受已有编译记录
                # 的页面，这种情况只能重新提交首次全空间编译。
                result = client.compile_spaces([space_id])
                accepted = int(result.get("acceptedRunCount") or 0)
                coalesced = int(result.get("coalescedRunCount") or 0)
                remote_run_ids = {
                    str(run["runId"])
                    for run in result.get("runs") or []
                    if isinstance(run, dict) and run.get("runId")
                }
                if remote_run_ids:
                    ctx.freeze(remote_compile_run_ids=sorted(remote_run_ids))
                ctx.log("上次 Run 在逐页初始化前被取消，重新提交首次编译", "warn")
                wait = _wait_for_compile(
                    ctx,
                    client,
                    space_id,
                    config,
                    expect_runs=accepted + coalesced,
                    coalesced_runs=coalesced,
                    expected_run_ids=remote_run_ids,
                    baseline_run_ids=baseline_run_ids,
                    baseline_sequence=baseline_sequence,
                )
            else:
                # 远端 Run 已经终态且没有失败页：后端重启可能只漏了质量检查。
                ctx.log("已保存的远端编译没有失败页面，只刷新质量闸门")
                wait = {
                    "status_counts": {
                        str(run.get("status") or "unknown"): sum(
                            1
                            for item in previous_runs
                            if str(item.get("status") or "unknown")
                            == str(run.get("status") or "unknown")
                        )
                        for run in previous_runs
                    },
                    "no_runs": False,
                    "timed_out": False,
                    "runs": previous_runs,
                    "progress": _progress_of(previous_runs),
                }
                accepted = coalesced = 0
        else:
            result = client.compile_spaces([space_id])
            accepted = int(result.get("acceptedRunCount") or 0)
            coalesced = int(result.get("coalescedRunCount") or 0)
            remote_run_ids = {
                str(run["runId"])
                for run in result.get("runs") or []
                if isinstance(run, dict) and run.get("runId")
            }
            if remote_run_ids:
                ctx.freeze(remote_compile_run_ids=sorted(remote_run_ids))
            if ctx.pause_requested:
                for remote_run_id in remote_run_ids:
                    client.cancel_compile_run(
                        remote_run_id, "Akasha-Benchmark task paused during compile submission"
                    )
                ctx.checkpoint()
            ctx.log(f"已请求编译：accepted={accepted} coalesced={coalesced}")
            wait = _wait_for_compile(
                ctx,
                client,
                space_id,
                config,
                expect_runs=accepted + coalesced,
                coalesced_runs=coalesced,
                expected_run_ids=remote_run_ids,
                baseline_run_ids=baseline_run_ids,
                baseline_sequence=baseline_sequence,
            )
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

        pace = _compile_pace(client, space_id, wait.get("runs") or None)
        if pace:
            compile_store.update_compile_run(ctx.db, compile_id, pace_json=dumps(pace))
            ctx.log(
                f"编译节奏（估算）：{pace['pages']} 篇用 "
                f"{pace['total_ms'] / 1000:.1f}s，约 {pace['per_page_ms'] / 1000:.1f}s/篇"
            )

        quality = _quality_gate(client, space_id)
        # 质量接口只报告空间里缺了多少产物，不报告是否仍有可用产物。保存本次
        # Run 的逐页结果，让查询层能区分「全军覆没」与「仅少数页面失败」。
        quality["progress"] = wait.get("progress") or {}
        compile_store.update_compile_run(ctx.db, compile_id, quality_json=dumps(quality))
        ctx.db.commit()
        ctx.log(f"质量闸门 {quality['gates']} -> {'通过' if quality['passed'] else '未通过'}")
        if not quality["passed"]:
            # 闸门报的是后果，原因要从逐页日志取。
            reasons = _page_failures(client, space_id)
            for line in reasons:
                ctx.log(f"编译失败原因：{line}", "error")
            detail = f"；{reasons[0]}" if reasons else ""
            raise RuntimeError(
                f"编译质量闸门未通过：整批结果不完整；成功页面仍可供查询层使用{detail}"
            )

    total_docs = len(compile_store.compile_docs(ctx.db, compile_id))
    ctx.progress(total_docs, total_docs, "编译完成")
    ctx.log("编译完成，可以进入查询层")


def _import_one(
    client: AkashaClient, doc: dict[str, Any], markdown: str, space_id: str
) -> dict[str, Any]:
    """导入一篇文档；预期的传输失败转成可由主线程落库的结果。"""
    try:
        page = client.import_page_text(f"{doc['doc_id']}.md", markdown, space_id)
        page_id = page.get("id") if isinstance(page, dict) else None
        error = None if page_id else f"响应里没有 page id: {page!r}"[:300]
    except (AkashaError, httpx.RequestError, OSError) as exc:
        page_id, error = None, f"{type(exc).__name__}: {exc}"[:300]
    return {"doc": doc, "page_id": page_id, "error": error}


def _import_docs(
    ctx: TaskContext,
    client: AkashaClient,
    compile_id: int,
    space_id: str,
    concurrency: int,
) -> None:
    """并发导入未导入文档；失败项在末尾重试一次，仍失败则暂停。

    worker 不接触 SQLite。正文在主线程读取，结果也由主线程逐条提交，因此暂停或
    进程退出后，继续任务只会选中 ``page_id`` 仍为空的文档。
    """
    pending = compile_store.compile_docs(ctx.db, compile_id, pending_only=True)
    total = len(compile_store.compile_docs(ctx.db, compile_id))
    done = total - len(pending)
    if not pending:
        ctx.log(f"{total} 篇语料均已导入，跳过")
        return

    worker_count = min(concurrency, len(pending))
    ctx.log(f"导入语料：待导入 {len(pending)} / 共 {total}，并发 {worker_count}")

    # httpx client 与实例级限流状态不跨 worker 共享；登录会复用共享 JWT。
    extras = [AkashaClient(client.config) for _ in range(worker_count - 1)]
    try:
        for spare in extras:
            spare.login()
        clients: queue.Queue[AkashaClient] = queue.Queue()
        for worker in (client, *extras):
            clients.put(worker)

        def task(item: tuple[dict[str, Any], str]) -> dict[str, Any]:
            doc, markdown = item
            borrowed = clients.get()
            try:
                return _import_one(borrowed, doc, markdown, space_id)
            finally:
                clients.put(borrowed)

        def run_pass(
            executor: ThreadPoolExecutor, docs: list[dict[str, Any]], *, retry: bool
        ) -> list[dict[str, Any]]:
            nonlocal done
            failures: list[dict[str, Any]] = []
            for start in range(0, len(docs), worker_count):
                ctx.checkpoint()
                batch = docs[start : start + worker_count]
                work = [
                    (doc, markdown_of(ctx.db, doc["dataset"], doc["doc_id"]))
                    for doc in batch
                ]
                for row in executor.map(task, work):
                    doc = row["doc"]
                    compile_store.record_page(
                        ctx.db,
                        compile_id,
                        doc["dataset"],
                        doc["doc_id"],
                        page_id=row["page_id"],
                        error=row["error"],
                    )
                    ctx.db.commit()
                    if row["error"]:
                        failures.append(doc)
                        prefix = "重试仍失败" if retry else "导入失败"
                        ctx.log(
                            f"{prefix} {doc['dataset']}/{doc['doc_id']}: {row['error']}",
                            "warn",
                        )
                    else:
                        done += 1
                ctx.progress(done, total, "导入语料")
            return failures

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            failures = run_pass(executor, pending, retry=False)
            if failures:
                ctx.checkpoint()
                ctx.log(f"首轮有 {len(failures)} 篇失败，移到末尾重试一次", "warn")
                failures = run_pass(executor, failures, retry=True)
    finally:
        for spare in extras:
            spare.close()

    if failures:
        raise Paused(
            f"{len(failures)} 篇语料重试后仍未导入，任务已暂停；"
            "继续任务将从未导入文档接着执行"
        )
