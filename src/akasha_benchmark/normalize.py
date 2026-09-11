"""归一化：把四组原始数据整成库里的 ``sample`` + ``corpus_doc``。

**库是事实来源**（PLAN.md §12 决策 2）。这一步把原始文件读进库，之后所有阶段
都从库里取样本与语料；``--export`` 可以另外落一份 jsonl，那是可选导出，
不是任何阶段的输入。

这一步不依赖 Akasha 在线，改造后仍然如此（§12.2 的硬性要求）。

    uv run python -m akasha_benchmark.normalize
    uv run python -m akasha_benchmark.normalize --dataset hotpotqa --export
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

from .datasets import (
    CORPUS_ID_RULES,
    DATASET_NAMES,
    SAMPLE_ID_RULES,
    CanonicalSample,
    load_corpus,
    normalized_dir,
    repo_relative,
    resolve,
)
from . import run_args
from .io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    sha256_file,
    sha256_text,
    utc_now,
)
from .metrics import registry
from .store import connect, repo


def normalize_dataset(
    connection: sqlite3.Connection, name: str, dataset_dir: Path | None = None
) -> dict[str, Any]:
    """归一化一个数据集进库，返回本次的统计。"""
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

    gold_dist: dict[int, int] = {}
    for sample in samples:
        gold_dist[len(sample.gold_doc_ids)] = gold_dist.get(len(sample.gold_doc_ids), 0) + 1

    # 上游哈希链的起点：原始文件的 sha256 进库，下游各层逐级往下带。
    repo.upsert_dataset(
        connection,
        name=adapter.name,
        adapter=type(adapter).__name__,
        adapter_version=adapter.version,
        provides=sorted(d.value for d in adapter.provides),
        identity_rules={
            "sample_id": SAMPLE_ID_RULES[adapter.name],
            "corpus_doc_id": CORPUS_ID_RULES[adapter.name],
        },
        qa_path=repo_relative(resolved.qa_path),
        qa_sha256=sha256_file(resolved.qa_path),
        qa_rows=len(rows),
        corpus_path=repo_relative(resolved.corpus_path),
        corpus_sha256=sha256_file(resolved.corpus_path),
        corpus_rows=len(corpus.docs),
        # 只报告不执行去重：musique 的重复 title 是不同段落，去重会丢 gold。
        dedup_stats=corpus.dedup_stats(),
        # 这是**去重后**的 gold 篇数分布（§0.2），与原始标注条数不同 ——
        # hotpotqa 的 supporting_facts 是 (title, 句子下标) 对，同一篇会出现多次。
        gold_count_distribution={str(k): v for k, v in sorted(gold_dist.items())},
        # 审计归因按 sha256(query) join 审计表，重复 question 会让那一行没法连。
        unique_question_texts=len({s.question for s in samples}),
    )
    sample_count = repo.replace_samples(
        connection, adapter.name, (s.model_dump(mode="json") for s in samples)
    )
    connection.commit()

    corpus_count = repo.replace_corpus(
        connection,
        adapter.name,
        (
            {**doc.model_dump(mode="json"), "text_sha256": sha256_text(doc.text)}
            for doc in corpus.docs
        ),
    )
    connection.commit()

    return {
        "dataset": adapter.name,
        "samples": sample_count,
        "corpus": corpus_count,
        "provides": sorted(d.value for d in adapter.provides),
        "gold_count_distribution": {str(k): v for k, v in sorted(gold_dist.items())},
    }


def export_dataset(
    connection: sqlite3.Connection, dataset: str, data_dir: Path | None = None
) -> Path:
    """把库里的归一化结果导出成 jsonl。**可选**，不是任何阶段的输入。

    留这个口子是为了「拿一份能 diff、能带走的快照」。权威始终在库里，
    所以 manifest 里记的是 ``exported_at`` 而不是 ``generated_at`` ——
    这份文件是库的投影，不是产它的那一步。
    """
    out_dir = normalized_dir(dataset, data_dir)
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
            for s in repo.samples_of(connection, dataset)
        ),
    )
    atomic_write_jsonl(
        out_dir / "corpus.jsonl",
        (
            {"doc_id": d["doc_id"], "title": d["title"], "text": d["text"]}
            for d in repo.corpus_of(connection, dataset)
        ),
    )
    record = repo.get_dataset(connection, dataset) or {}
    atomic_write_json(
        out_dir / "manifest.json",
        {
            "stage": "normalize",
            "note": "exported from the database; the database is the source of truth",
            "dataset": dataset,
            "exported_at": utc_now(),
            "adapter": record.get("adapter"),
            "adapter_version": record.get("adapter_version"),
            "provides": repo.loads(record.get("provides_json"), []),
            "identity_rules": repo.loads(record.get("identity_rules_json"), {}),
            "sources": {
                "qa": {
                    "path": record.get("qa_path"),
                    "sha256": record.get("qa_sha256"),
                    "rows": record.get("qa_rows"),
                },
                "corpus": {
                    "path": record.get("corpus_path"),
                    "sha256": record.get("corpus_sha256"),
                    "rows": record.get("corpus_rows"),
                },
            },
            "corpus_dedup_stats": repo.loads(record.get("dedup_stats_json"), {}),
            "dedup_applied": False,
            "gold_count_distribution": repo.loads(
                record.get("gold_count_distribution_json"), {}
            ),
            "unique_question_texts": record.get("unique_question_texts"),
        },
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(DATASET_NAMES),
        help="要归一化的数据集，可重复。默认四组全做。",
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None, help="仅 --export 用")
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument(
        "--export", action="store_true", help="额外导出 jsonl 快照（可选，非阶段输入）"
    )
    run_args.add_argument(parser)
    args = run_args.apply(parser.parse_args(argv), stage="normalize")

    connection = connect(args.db)
    try:
        # 指标声明同步进表，供前端读取；计算依据始终是代码里的声明。
        repo.sync_metric_definitions(connection, registry.as_rows())
        connection.commit()

        failures = 0
        for name in args.dataset or list(DATASET_NAMES):
            try:
                result = normalize_dataset(connection, name, args.dataset_dir)
            except Exception as exc:  # noqa: BLE001 - 报告后继续做下一个数据集
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            print(
                f"ok   {name:<16} samples={result['samples']:<5} "
                f"corpus={result['corpus']:<6} "
                f"gold_dist={result['gold_count_distribution']} "
                f"provides={','.join(result['provides'])}"
            )
            if args.export:
                print(f"     exported -> {export_dataset(connection, name, args.data_dir)}")

        if failures:
            print(f"\n{failures} dataset(s) failed", file=sys.stderr)
        return 1 if failures else 0
    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())
