"""子集：在同一个 run_id 下，给每个数据集抽出可独立评测的子集。

抽样顺序必须是**先 QA 后 corpus**。随机抽 100 篇 corpus 的话，
大部分 gold 文档会落在子集外，Recall 会因为跟检索器毫无关系的原因被钉在 0 附近。

    1. 固定种子抽 N 条 QA
    2. 这些 QA 的 gold doc_id 全集作为 corpus 必选集
    3. 从剩余 corpus 随机补负样本，凑到目标规模

narrativeqa 走另一条路：它没有 gold 标注，而且 293 个问题只覆盖 10 篇文档，
抽 100 条问题会把 4111 个 chunk 里的绝大多数都牵进来。所以它改成
整篇整篇地取文档（连同该文档的全部 chunk），再取属于这些文档的问题。

    uv run python -m akasha_benchmark.subset --run-id run001
    uv run python -m akasha_benchmark.subset --run-id run001 --dataset hotpotqa --qa-limit 100
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .datasets import (
    DATASET_NAMES,
    Capability,
    CanonicalSample,
    CorpusDoc,
    get_adapter,
    normalized_dir,
    subset_dir,
)
from .io_utils import atomic_write_json, atomic_write_jsonl, atomic_write_text, load_json, read_jsonl, sha256_text, utc_now

DEFAULT_QA_LIMIT = 100
DEFAULT_NEGATIVES_RATIO = 1.0
DEFAULT_NARRATIVEQA_DOCS = 2


def _load_normalized(
    dataset: str, data_dir: Path | None
) -> tuple[list[CanonicalSample], list[CorpusDoc], dict]:
    """读归一化产出。顺带把上游 manifest 带出来，好把 sha256 记进本阶段 manifest。"""
    src = normalized_dir(dataset, data_dir)
    manifest_path = src / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{dataset}: {manifest_path} missing; run `python -m akasha_benchmark.normalize` first"
        )
    samples = [CanonicalSample.model_validate(row) for row in read_jsonl(src / "samples.jsonl")]
    corpus = [CorpusDoc.model_validate(row) for row in read_jsonl(src / "corpus.jsonl")]
    return samples, corpus, load_json(manifest_path)


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
    samples: list[CanonicalSample], limit: int, key: str, rng: random.Random
) -> tuple[list[CanonicalSample], dict[str, int]]:
    """按 ``metadata[key]`` 的每个取值分层，层内按比例抽样。"""
    strata: dict[str, list[CanonicalSample]] = defaultdict(list)
    for sample in samples:
        strata[str(sample.metadata[key])].append(sample)

    quotas = _largest_remainder({k: len(v) for k, v in strata.items()}, limit)
    picked: list[CanonicalSample] = []
    for name in sorted(strata):
        # 先按 sample_id 排序再抽，这样结果不依赖字典/文件的遍历顺序。
        available = sorted(strata[name], key=lambda s: s.sample_id)
        quota = min(quotas[name], len(available))
        picked.extend(rng.sample(available, quota))

    # 某一层样本不够时从其余样本补齐，保证总数仍达到 limit。
    if len(picked) < limit:
        chosen = {s.sample_id for s in picked}
        rest = sorted((s for s in samples if s.sample_id not in chosen), key=lambda s: s.sample_id)
        picked.extend(rng.sample(rest, min(limit - len(picked), len(rest))))

    picked.sort(key=lambda s: s.sample_id)
    return picked, quotas


def _safe_doc_id(doc_id: str) -> str:
    """doc_id 会直接当文件名，所以任何可能跳出目录的取值都要拒掉。

    doc_id 来自数据文件，属于外部输入，不能假定它是干净的。
    """
    if not doc_id or doc_id in {".", ".."} or set(doc_id) & set('/\\:*?"<>|') or doc_id != doc_id.strip():
        raise ValueError(f"doc_id {doc_id!r} is not safe to use as a filename")
    return doc_id


def build_subset(
    dataset: str,
    run_id: str,
    seed: int,
    qa_limit: int = DEFAULT_QA_LIMIT,
    negatives_ratio: float = DEFAULT_NEGATIVES_RATIO,
    narrativeqa_docs: int = DEFAULT_NARRATIVEQA_DOCS,
    data_dir: Path | None = None,
) -> dict:
    """抽一个数据集的子集，返回写入的 manifest。"""
    adapter = get_adapter(dataset)
    samples, corpus, upstream = _load_normalized(adapter.name, data_dir)
    by_id = {doc.doc_id: doc for doc in corpus}
    # 种子里带上 run_id 和数据集名，这样同一个 seed 下各数据集互不相关，
    # 但同一组 (run_id, dataset, seed) 永远可复现。
    rng = random.Random(f"{run_id}:{adapter.name}:{seed}")

    strategy: str
    quotas: dict[str, int] = {}
    negatives: list[str] = []

    if adapter.supports(Capability.EVIDENCE_RECALL):
        # musique 按跳数分层，否则随机 100 条几乎全是 2hop，测不出多跳深度的影响。
        if "hop_prefix" in samples[0].metadata:
            strategy = "stratified_by_hop_prefix"
            picked, quotas = _stratified_sample(samples, qa_limit, "hop_prefix", rng)
        else:
            strategy = "uniform_qa_then_gold_corpus"
            ordered = sorted(samples, key=lambda s: s.sample_id)
            picked = sorted(rng.sample(ordered, min(qa_limit, len(ordered))), key=lambda s: s.sample_id)

        gold_ids = sorted({d for s in picked for d in s.gold_doc_ids})
        pool = sorted(set(by_id) - set(gold_ids))
        want = int(round(len(gold_ids) * negatives_ratio))
        negatives = sorted(rng.sample(pool, min(want, len(pool))))
        doc_ids = sorted(set(gold_ids) | set(negatives))
    else:
        # narrativeqa：整篇取文档、连同全部 chunk，再取属于这些文档的问题。
        strategy = "whole_documents"
        chunks_by_doc: dict[str, list[str]] = defaultdict(list)
        for doc in corpus:
            chunks_by_doc[doc.doc_id.rsplit("_", 1)[0]].append(doc.doc_id)
        # 优先取 chunk 最少的文档：chunk 数直接决定编译成本，而这份数据的价值
        # 在答案质量和跨文档实体合并，不在检索深度，没必要为它烧编译预算。
        ranked = sorted(chunks_by_doc, key=lambda d: (len(chunks_by_doc[d]), d))
        chosen_docs = sorted(ranked[:narrativeqa_docs])
        doc_ids = sorted(
            (c for d in chosen_docs for c in chunks_by_doc[d]),
            key=lambda c: (c.rsplit("_", 1)[0], int(c.rsplit("_", 1)[1])),
        )
        picked = sorted(
            (s for s in samples if s.metadata["document_id"] in set(chosen_docs)),
            key=lambda s: int(s.dataset_sample_id),
        )
        gold_ids = []
        quotas = {d: len(chunks_by_doc[d]) for d in chosen_docs}
        if qa_limit and len(picked) > qa_limit:
            picked = sorted(rng.sample(picked, qa_limit), key=lambda s: int(s.dataset_sample_id))

    # 验收标准：每条 sample 的所有 gold 都必须在子集 corpus 内。
    subset_ids = set(doc_ids)
    uncovered = {
        s.sample_id: [d for d in s.gold_doc_ids if d not in subset_ids]
        for s in picked
        if any(d not in subset_ids for d in s.gold_doc_ids)
    }
    if uncovered:
        raise RuntimeError(
            f"{adapter.name}: {len(uncovered)} samples have gold outside the subset corpus, "
            f"e.g. {next(iter(uncovered.items()))}"
        )

    out_dir = subset_dir(run_id, adapter.name, data_dir)
    corpus_dir = out_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    md_hashes: dict[str, str] = {}
    for doc_id in doc_ids:
        doc = by_id[doc_id]
        markdown = doc.to_markdown()
        atomic_write_text(corpus_dir / f"{_safe_doc_id(doc_id)}.md", markdown)
        md_hashes[doc_id] = sha256_text(markdown)

    # 上一次用不同抽样跑出来的残留 md，会在入库时被一起导进库，
    # 所以把不属于本次子集的文件删掉。
    removed = 0
    for existing in corpus_dir.glob("*.md"):
        if existing.stem not in subset_ids:
            existing.unlink()
            removed += 1

    atomic_write_jsonl(out_dir / "samples.jsonl", (s.model_dump(mode="json") for s in picked))
    atomic_write_json(out_dir / "corpus_hashes.json", md_hashes)

    gold_per_sample = [len(s.gold_doc_ids) for s in picked]
    manifest = {
        "stage": "subset",
        "run_id": run_id,
        "dataset": adapter.name,
        "generated_at": utc_now(),
        "seed": seed,
        "rng_stream": f"{run_id}:{adapter.name}:{seed}",
        "strategy": strategy,
        "qa_limit": qa_limit,
        "negatives_ratio": negatives_ratio,
        "capabilities": sorted(c.value for c in adapter.capabilities),
        "qa_count": len(picked),
        "corpus_count": len(doc_ids),
        "gold_doc_count": len(gold_ids),
        "negative_doc_count": len(negatives),
        "gold_coverage": 1.0 if picked else 0.0,
        "stale_md_removed": removed,
        "strata": quotas,
        "gold_per_sample": {
            "min": min(gold_per_sample, default=0),
            "max": max(gold_per_sample, default=0),
            "mean": round(sum(gold_per_sample) / len(gold_per_sample), 4) if gold_per_sample else 0.0,
        },
        "hop_distribution": dict(
            sorted(Counter(str(s.metadata.get("hop_count", "n/a")) for s in picked).items())
        ),
        "type_distribution": dict(
            sorted(Counter(str(s.metadata.get("type", "n/a")) for s in picked).items())
        ),
        "upstream": {
            "normalized_samples_sha256": upstream["outputs"]["samples"]["sha256"],
            "normalized_corpus_sha256": upstream["outputs"]["corpus"]["sha256"],
            "identity_rules": upstream["identity_rules"],
        },
        "corpus_md_sha256": md_hashes,
    }
    atomic_write_json(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
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
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    failures = 0
    total_corpus = 0
    for name in args.dataset or list(DATASET_NAMES):
        try:
            manifest = build_subset(
                name,
                run_id=args.run_id,
                seed=args.seed,
                qa_limit=args.qa_limit,
                negatives_ratio=args.negatives_ratio,
                narrativeqa_docs=args.narrativeqa_docs,
                data_dir=args.data_dir,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        total_corpus += manifest["corpus_count"]
        print(
            f"ok   {name:<16} qa={manifest['qa_count']:<4} "
            f"corpus={manifest['corpus_count']:<5} "
            f"(gold={manifest['gold_doc_count']} neg={manifest['negative_doc_count']}) "
            f"strategy={manifest['strategy']}"
        )

    if not failures:
        print(f"\ntotal corpus documents={total_corpus} -> ~{total_corpus * 2} compile LLM calls")
    else:
        print(f"\n{failures} dataset(s) failed", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
