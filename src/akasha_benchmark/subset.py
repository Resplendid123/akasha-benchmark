"""子集：建一个**索引层**，给每个数据集抽出可独立评测的子集。

抽样顺序必须是**先 QA 后 corpus**。随机抽 100 篇 corpus 的话，
大部分 gold 文档会落在子集外，Recall 会因为跟检索器毫无关系的原因被钉在 0 附近。

    1. 固定种子抽 N 条 QA
    2. 这些 QA 的 gold doc_id 全集作为 corpus 必选集
    3. 从剩余 corpus 随机补负样本，凑到目标规模

narrativeqa 走另一条路：它没有 gold 标注，而且 293 个问题只覆盖 10 篇文档，
抽 100 条问题会把 4111 个 chunk 里的绝大多数都牵进来。所以它改成
整篇整篇地取文档（连同该文档的全部 chunk），再取属于这些文档的问题。

产出进库（``subset_sample`` / ``subset_doc``），**md 正文也进库** —— 入库阶段
从库里取正文上传。这一步不依赖 Akasha 在线。

    uv run python -m akasha_benchmark.subset --label run002
    uv run python -m akasha_benchmark.subset --label run002 --dataset hotpotqa --qa-limit 100
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import run_args
from .datasets import DATASET_NAMES, DataDependency, get_adapter, subset_dir
from .io_utils import atomic_write_json, atomic_write_jsonl, atomic_write_text, sha256_text, utc_now
from .store import connect, repo

DEFAULT_QA_LIMIT = 100
DEFAULT_NEGATIVES_RATIO = 1.0
DEFAULT_NARRATIVEQA_DOCS = 2


def _largest_remainder(weights: dict[str, int], total: int) -> dict[str, int]:
    """按 ``weights`` 的比例把 ``total`` 分配到各层。

    用最大余额法，保证各层之和精确等于 ``total``，而且小层也能分到名额
    （musique 的 4hop2 只占总体 2.7%，直接取整会被抹成 0）。
    """
    pool = sum(weights.values())
    if pool == 0:
        return {k: 0 for k in weights}
    exact = {k: total * v / pool for k, v in weights.items()}
    floors = {k: int(v) for k, v in exact.items()}
    remainder = total - sum(floors.values())
    order = sorted(weights, key=lambda k: (-(exact[k] - floors[k]), k))
    for key in order[:remainder]:
        floors[key] += 1
    return floors


def _stratified_sample(
    samples: list[dict[str, Any]], limit: int, key: str, rng: random.Random
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """按 ``metadata[key]`` 的每个取值分层，层内按比例抽样。"""
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        strata[str(sample["metadata"][key])].append(sample)

    quotas = _largest_remainder({k: len(v) for k, v in strata.items()}, limit)
    picked: list[dict[str, Any]] = []
    for name in sorted(strata):
        # 先按 sample_id 排序再抽，结果不依赖查询的返回顺序。
        available = sorted(strata[name], key=lambda s: s["sample_id"])
        picked.extend(rng.sample(available, min(quotas[name], len(available))))

    # 某一层样本不够时从其余样本补齐，保证总数仍达到 limit。
    if len(picked) < limit:
        chosen = {s["sample_id"] for s in picked}
        rest = sorted(
            (s for s in samples if s["sample_id"] not in chosen), key=lambda s: s["sample_id"]
        )
        picked.extend(rng.sample(rest, min(limit - len(picked), len(rest))))

    picked.sort(key=lambda s: s["sample_id"])
    return picked, quotas


def _safe_doc_id(doc_id: str) -> str:
    """doc_id 会当文件名用（导入时的 filename、可选导出时的 md 文件名），
    所以任何可能跳出目录的取值都要拒掉。它来自数据文件，属于外部输入。
    """
    if (
        not doc_id
        or doc_id in {".", ".."}
        or set(doc_id) & set('/\\:*?"<>|')
        or doc_id != doc_id.strip()
    ):
        raise ValueError(f"doc_id {doc_id!r} is not safe to use as a filename")
    return doc_id


def _markdown(title: str, text: str) -> str:
    """渲染成 Akasha 导入用的 Markdown。

    Akasha 优先取首个 heading 当 title 并从正文移除，所以 heading 负责 title、
    文件名负责 doc_id。两者独立，即使 musique 有重复 title 也不影响身份追踪。
    """
    return f"# {title}\n\n{text}\n"


def build_subset(
    connection: sqlite3.Connection,
    layer_id: int,
    dataset: str,
    *,
    seed: int,
    qa_limit: int = DEFAULT_QA_LIMIT,
    negatives_ratio: float = DEFAULT_NEGATIVES_RATIO,
    narrativeqa_docs: int = DEFAULT_NARRATIVEQA_DOCS,
) -> dict[str, Any]:
    """抽一个数据集的子集写进库，返回统计。"""
    adapter = get_adapter(dataset)
    record = repo.get_dataset(connection, adapter.name)
    if record is None:
        raise FileNotFoundError(
            f"{adapter.name}: not in the database; run "
            "`python -m akasha_benchmark.normalize` first"
        )
    samples = repo.samples_of(connection, adapter.name)
    corpus = repo.corpus_of(connection, adapter.name)
    by_id = {doc["doc_id"]: doc for doc in corpus}

    # 随机源是 (数据集名, seed)，**刻意不含 label**。
    #
    # 带上 label 的话，抽样结果就依赖一个 subset_hash 覆盖不到的东西 ——
    # 两个层会拿到同样的哈希却是完全不同的子集（实测：同 seed 不同 label,
    # hotpotqa 400 篇里只重叠 30 篇）。那让 subset_hash 变成一个会说谎的字段。
    #
    # 更要紧的是它会毁掉 对照实验：「同子集、换 embedding」需要
    # 两个层拿到**同一批**文档，而 label 必须唯一，于是永远凑不出来。
    # 想换一批样本就换 seed —— 那本来就是 seed 的职责，不需要第二个旋钮。
    rng = random.Random(f"{adapter.name}:{seed}")

    quotas: dict[str, int] = {}
    negatives: list[str] = []

    if adapter.has(DataDependency.GOLD_DOCS):
        # musique 按跳数分层，否则随机 100 条几乎全是 2hop，测不出多跳深度的影响。
        if "hop_prefix" in samples[0]["metadata"]:
            strategy = "stratified_by_hop_prefix"
            picked, quotas = _stratified_sample(samples, qa_limit, "hop_prefix", rng)
        else:
            strategy = "uniform_qa_then_gold_corpus"
            ordered = sorted(samples, key=lambda s: s["sample_id"])
            picked = sorted(
                rng.sample(ordered, min(qa_limit, len(ordered))), key=lambda s: s["sample_id"]
            )

        gold_ids = sorted({d for s in picked for d in s["gold_doc_ids"]})
        pool = sorted(set(by_id) - set(gold_ids))
        want = int(round(len(gold_ids) * negatives_ratio))
        negatives = sorted(rng.sample(pool, min(want, len(pool))))
        doc_ids = sorted(set(gold_ids) | set(negatives))
    else:
        # narrativeqa：整篇取文档、连同全部 chunk，再取属于这些文档的问题。
        strategy = "whole_documents"
        chunks_by_doc: dict[str, list[str]] = defaultdict(list)
        for doc in corpus:
            chunks_by_doc[doc["doc_id"].rsplit("_", 1)[0]].append(doc["doc_id"])
        # 优先取 chunk 最少的文档：chunk 数直接决定编译成本，而这份数据的价值
        # 在答案质量，不在检索深度，没必要为它烧编译预算。
        ranked = sorted(chunks_by_doc, key=lambda d: (len(chunks_by_doc[d]), d))
        chosen_docs = sorted(ranked[:narrativeqa_docs])
        doc_ids = sorted(
            (c for d in chosen_docs for c in chunks_by_doc[d]),
            key=lambda c: (c.rsplit("_", 1)[0], int(c.rsplit("_", 1)[1])),
        )
        picked = sorted(
            (s for s in samples if s["metadata"]["document_id"] in set(chosen_docs)),
            key=lambda s: int(s["dataset_sample_id"]),
        )
        gold_ids = []
        quotas = {d: len(chunks_by_doc[d]) for d in chosen_docs}
        if qa_limit and len(picked) > qa_limit:
            picked = sorted(
                rng.sample(picked, qa_limit), key=lambda s: int(s["dataset_sample_id"])
            )

    # 验收标准：每条 sample 的所有 gold 都必须在子集 corpus 内。
    subset_ids = set(doc_ids)
    uncovered = {
        s["sample_id"]: [d for d in s["gold_doc_ids"] if d not in subset_ids]
        for s in picked
        if any(d not in subset_ids for d in s["gold_doc_ids"])
    }
    if uncovered:
        raise RuntimeError(
            f"{adapter.name}: {len(uncovered)} samples have gold outside the subset corpus, "
            f"e.g. {next(iter(uncovered.items()))}"
        )

    gold_set = set(gold_ids)
    docs = []
    for doc_id in doc_ids:
        doc = by_id[doc_id]
        markdown = _markdown(doc["title"], doc["text"])
        docs.append(
            {
                "doc_id": _safe_doc_id(doc_id),
                "md_text": markdown,
                "md_sha256": sha256_text(markdown),
                "is_gold": doc_id in gold_set,
            }
        )

    repo.upsert_index_layer_dataset(
        connection,
        layer_id,
        adapter.name,
        strategy=strategy,
        qa_count=len(picked),
        corpus_count=len(doc_ids),
        gold_doc_count=len(gold_ids),
        negative_doc_count=len(negatives),
        strata=quotas,
        # 上游哈希链：这一层是从哪份归一化产物抽出来的。
        normalized_qa_sha256=record["qa_sha256"],
        normalized_corpus_sha256=record["corpus_sha256"],
    )
    # replace_subset 会先删旧行：重抽样意味着上一次的 md 全部作废，留着的话
    # 入库会把残留一起导进 Akasha，语料规模就悄悄变大了。
    repo.replace_subset(
        connection, layer_id, adapter.name, [s["sample_id"] for s in picked], docs
    )
    # 每次改动文档集都跟着重算哈希，这样「哈希与内容一致」在任何一次
    # build_subset 之后都成立，而不是只在 main() 跑完之后成立。
    # 幂等且便宜（一次排序查询），所以逐数据集调没有问题。
    repo.recompute_subset_hash(connection, layer_id)
    connection.commit()

    gold_per_sample = [len(s["gold_doc_ids"]) for s in picked]
    return {
        "dataset": adapter.name,
        "strategy": strategy,
        "qa_count": len(picked),
        "corpus_count": len(doc_ids),
        "gold_doc_count": len(gold_ids),
        "negative_doc_count": len(negatives),
        "gold_coverage": 1.0 if picked else 0.0,
        "strata": quotas,
        "gold_per_sample": {
            "min": min(gold_per_sample, default=0),
            "max": max(gold_per_sample, default=0),
            "mean": (
                round(sum(gold_per_sample) / len(gold_per_sample), 4) if gold_per_sample else 0.0
            ),
        },
        "hop_distribution": dict(
            sorted(Counter(str(s["metadata"].get("hop_count", "n/a")) for s in picked).items())
        ),
        "type_distribution": dict(
            sorted(Counter(str(s["metadata"].get("type", "n/a")) for s in picked).items())
        ),
    }


def export_subset(
    connection: sqlite3.Connection, layer_id: int, dataset: str, data_dir: Path | None = None
) -> Path:
    """把库里的子集导出成 md + jsonl。**可选**，不是任何阶段的输入。"""
    layer = repo.get_index_layer(connection, layer_id) or {}
    out_dir = subset_dir(str(layer.get("label") or layer_id), dataset, data_dir)
    corpus_dir = out_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    docs = repo.subset_docs(connection, layer_id, dataset)
    hashes: dict[str, str] = {}
    for doc in docs:
        atomic_write_text(corpus_dir / f"{doc['doc_id']}.md", doc["md_text"])
        hashes[doc["doc_id"]] = doc["md_sha256"]

    # 不属于本次子集的残留 md 删掉，避免下游误以为它们还在集合里。
    for existing in corpus_dir.glob("*.md"):
        if existing.stem not in hashes:
            existing.unlink()

    atomic_write_jsonl(
        out_dir / "samples.jsonl",
        (
            {
                "dataset": s["dataset"],
                "sample_id": s["sample_id"],
                "dataset_sample_id": s["dataset_sample_id"],
                "question": s["question"],
                "answers": list(s["answers"]),
                "gold_doc_ids": list(s["gold_doc_ids"]),
                "metadata": s["metadata"],
            }
            for s in repo.subset_samples(connection, layer_id, dataset)
        ),
    )
    atomic_write_json(out_dir / "corpus_hashes.json", hashes)
    return out_dir


class LayerAlreadyIngested(RuntimeError):
    """这一层已经入库过，重抽子集会让它的 page_map 全部失效。"""


def ensure_layer(
    connection: sqlite3.Connection,
    *,
    label: str,
    seed: int,
    qa_limit: int,
    negatives_ratio: float,
    narrativeqa_docs: int,
    datasets: list[str],
) -> tuple[int, bool]:
    """取或建索引层，返回 ``(id, 是否新建)``。

    同 label 复用同一层，这样重跑 subset 是「重抽这一层」而不是「多出一层」。
    ``config_hash`` 此时留空 —— 它要吃 compiler 与 embedding，入库时才补齐。

    **已入库的层拒绝重抽。** ``replace_subset`` 删 ``subset_doc`` 但不动
    ``page_map``，所以重抽之后两者会指向不同的文档集 —— 而条数往往仍然相等
    （同一个 qa_limit 抽出来的语料规模差不多），于是那种状态看起来是正常的。
    接下来查询打在装着旧文档的 Space 上、指标按新 gold 算，每条检索数都是 0,
    看起来像检索烂到极点。要改抽样就换个 label 建新层，或者先清掉入库产物。
    """
    existing = repo.index_layer_by_label(connection, label)
    if existing and existing["ingested_at"]:
        layer_id = int(existing["id"])
        mapped = sum(repo.page_map_counts(connection, layer_id).values())
        raise LayerAlreadyIngested(
            f"index layer {label!r} (#{layer_id}) was ingested on "
            f"{existing['ingested_at']} and has {mapped} page_map row(s). Re-sampling "
            "would leave those pages pointing at documents that are no longer in the "
            "subset, and the mismatch does not raise — it just makes every retrieval "
            "metric wrong. Use a new label for different sampling parameters, or "
            "discard this layer's ingest first."
        )
    if existing:
        return int(existing["id"]), False
    # subset_hash 先留空串：它是内容寻址的，要等文档抽完才算得出来
    # （见 repo.recompute_subset_hash）。
    layer_id = repo.create_index_layer(
        connection,
        label=label,
        subset_hash="",
        seed=seed,
        qa_limit=qa_limit,
        negatives_ratio=negatives_ratio,
        narrativeqa_docs=narrativeqa_docs,
    )
    connection.commit()
    return layer_id, True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # required=True 改成合并后校验：参数可能来自库里的 run_config，
    # argparse 会在读库之前就拒绝。见 run_args.require。
    parser.add_argument("--label", default=None, help="索引层标签，例如 run002")
    parser.add_argument("--dataset", action="append", choices=list(DATASET_NAMES))
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--qa-limit", type=int, default=DEFAULT_QA_LIMIT)
    parser.add_argument(
        "--negatives-ratio",
        type=float,
        default=DEFAULT_NEGATIVES_RATIO,
        help="每篇 gold 配多少篇负样本（1.0 即 gold 全集 + 等量负样本）",
    )
    parser.add_argument("--narrativeqa-docs", type=int, default=DEFAULT_NARRATIVEQA_DOCS)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None, help="仅 --export 用")
    parser.add_argument("--export", action="store_true", help="额外导出 md/jsonl（可选）")
    run_args.add_argument(parser)
    args = run_args.apply(parser.parse_args(argv), stage="subset")
    run_args.require(args.label, "label", "subset")

    datasets = args.dataset or list(DATASET_NAMES)
    connection = connect(args.db)
    try:
        try:
            layer_id, created = _ensure(connection, args, datasets)
        except LayerAlreadyIngested as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            return 1
        print(
            f"index layer #{layer_id} ({args.label}) "
            f"{'created' if created else 'reused'}, seed={args.seed}"
        )

        failures = 0
        total_corpus = 0
        for name in datasets:
            try:
                result = build_subset(
                    connection,
                    layer_id,
                    name,
                    seed=args.seed,
                    qa_limit=args.qa_limit,
                    negatives_ratio=args.negatives_ratio,
                    narrativeqa_docs=args.narrativeqa_docs,
                )
            except Exception as exc:  # noqa: BLE001 - 报告后继续做下一个数据集
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            total_corpus += result["corpus_count"]
            print(
                f"ok   {name:<16} qa={result['qa_count']:<4} "
                f"corpus={result['corpus_count']:<5} "
                f"(gold={result['gold_doc_count']} neg={result['negative_doc_count']}) "
                f"strategy={result['strategy']}"
            )
            if args.export:
                print(f"     exported -> {export_subset(connection, layer_id, name, args.data_dir)}")

        # 全部数据集抽完之后，按实际文档集重算哈希。
        digest = repo.recompute_subset_hash(connection, layer_id)
        repo.update_index_layer(connection, layer_id, subset_built_at=utc_now())
        connection.commit()
        print(f"subset_hash={digest}（按实际文档集内容寻址）")

        if not failures:
            print(f"\ntotal corpus documents={total_corpus} -> ~{total_corpus * 2} compile LLM calls")
        else:
            print(f"\n{failures} dataset(s) failed", file=sys.stderr)
        return 1 if failures else 0
    finally:
        connection.close()


def _ensure(connection, args, datasets: list[str]) -> tuple[int, bool]:
    return ensure_layer(
        connection,
        label=args.label,
        seed=args.seed,
        qa_limit=args.qa_limit,
        negatives_ratio=args.negatives_ratio,
        narrativeqa_docs=args.narrativeqa_docs,
        datasets=datasets,
    )


if __name__ == "__main__":
    sys.exit(main())
