"""轮次 1：把四组数据归一化成 samples.jsonl + corpus.jsonl。

全量，不抽样。抽样是轮次 2 的事，而且它需要一个已经校验过的规范化底座才能抽。

    uv run python -m akasha_benchmark.normalize
    uv run python -m akasha_benchmark.normalize --dataset hotpotqa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .datasets import (
    CORPUS_ID_RULES,
    DATASET_NAMES,
    SAMPLE_ID_RULES,
    CanonicalSample,
    load_corpus,
    normalized_dir,
    resolve,
)
from .io_utils import atomic_write_json, atomic_write_jsonl, load_json, sha256_file, utc_now


def normalize_dataset(
    name: str, dataset_dir: Path | None = None, data_dir: Path | None = None
) -> dict:
    """归一化一个数据集，返回写入的 manifest。"""
    resolved = resolve(name, dataset_dir)
    adapter = resolved.adapter

    # 先建 corpus 索引：适配器解析 gold 时要靠它反查 doc_id。
    corpus = load_corpus(adapter.name, resolved.corpus_path)
    rows = load_json(resolved.qa_path)
    if not isinstance(rows, list):
        raise ValueError(f"{resolved.qa_path}: expected a JSON array")

    # 行数和实测快照不一致，说明数据换版了，身份规则需要重新确认。
    expected = adapter.expected_qa_rows()
    if expected is not None and len(rows) != expected:
        raise ValueError(
            f"{adapter.name}: QA file has {len(rows)} rows, adapter expects {expected}. "
            "The snapshot changed; re-verify the identity rules before continuing."
        )

    samples: list[CanonicalSample] = []
    seen_ids: dict[str, int] = {}
    for row_index, row in enumerate(rows):
        sample = adapter.parse_row(row, row_index, corpus)
        # 重复 ID 直接报错，不静默保留后一条。
        if sample.sample_id in seen_ids:
            raise ValueError(
                f"{adapter.name}: duplicate sample_id {sample.sample_id!r} at rows "
                f"{seen_ids[sample.sample_id]} and {row_index}"
            )
        seen_ids[sample.sample_id] = row_index
        samples.append(sample)

    out_dir = normalized_dir(adapter.name, data_dir)
    sample_count = atomic_write_jsonl(
        out_dir / "samples.jsonl", (s.model_dump(mode="json") for s in samples)
    )
    corpus_count = atomic_write_jsonl(
        out_dir / "corpus.jsonl", (d.model_dump(mode="json") for d in corpus.docs)
    )

    gold_dist: dict[int, int] = {}
    for sample in samples:
        gold_dist[len(sample.gold_doc_ids)] = gold_dist.get(len(sample.gold_doc_ids), 0) + 1

    manifest = {
        "round": 1,
        "dataset": adapter.name,
        "adapter": type(adapter).__name__,
        "adapter_version": adapter.version,
        "generated_at": utc_now(),
        "capabilities": sorted(c.value for c in adapter.capabilities),
        "identity_rules": {
            "sample_id": SAMPLE_ID_RULES[adapter.name],
            "corpus_doc_id": CORPUS_ID_RULES[adapter.name],
        },
        "sources": {
            "qa": {
                "path": str(resolved.qa_path.resolve()),
                "sha256": sha256_file(resolved.qa_path),
                "rows": len(rows),
            },
            "corpus": {
                "path": str(resolved.corpus_path.resolve()),
                "sha256": sha256_file(resolved.corpus_path),
                "rows": len(corpus.docs),
            },
        },
        "outputs": {
            "samples": {"rows": sample_count, "sha256": sha256_file(out_dir / "samples.jsonl")},
            "corpus": {"rows": corpus_count, "sha256": sha256_file(out_dir / "corpus.jsonl")},
        },
        # 只报告不执行：musique 的重复 title 是不同段落，去重会丢 gold。
        "corpus_dedup_stats": corpus.dedup_stats(),
        "dedup_applied": False,
        # 注意这是**去重后**的 gold 篇数分布，与 PLAN.md 0.4 的去重前数字不同。
        "gold_count_distribution": {str(k): v for k, v in sorted(gold_dist.items())},
        # 轮次 5 按 sha256(query) join 审计表，重复 question 会让那一行没法连。
        "unique_question_texts": len({s.question for s in samples}),
    }
    atomic_write_json(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(DATASET_NAMES),
        help="要归一化的数据集，可重复。默认四组全做。",
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    targets = args.dataset or list(DATASET_NAMES)
    failures = 0
    for name in targets:
        try:
            manifest = normalize_dataset(name, args.dataset_dir, args.data_dir)
        except Exception as exc:  # noqa: BLE001 - 报告后继续做下一个数据集
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        out = manifest["outputs"]
        print(
            f"ok   {name:<16} samples={out['samples']['rows']:<5} "
            f"corpus={out['corpus']['rows']:<6} "
            f"gold_dist={manifest['gold_count_distribution']} "
            f"caps={','.join(manifest['capabilities'])}"
        )

    if failures:
        print(f"\n{failures} dataset(s) failed", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
