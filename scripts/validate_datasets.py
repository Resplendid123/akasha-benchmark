"""归一化产物的验收检查。

逐行过全量数据，并把归一化产物与原始文件对比。


每个数据集检查：
  * 每一行原始数据能否通过适配器
  * dataset_sample_id 是否缺失或重复
  * gold 篇数分布；「声明了 GOLD_DOCS 却抽不出 gold」的行
  * 每个 gold doc_id 是否都能在 corpus 中找到
  * 重复的 question 文本
  * corpus 的 (title, text) 唯一性
  * 归一化产物与重新推导的结果是否逐行一致
  * manifest 里的 sha256 与磁盘上的文件是否还对得上

    uv run python scripts/validate_datasets.py
    uv run python scripts/validate_datasets.py --dataset musique
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from akasha_benchmark.datasets import (  # noqa: E402
    DATASET_NAMES,
    DataDependency,
    load_corpus,
    resolve,
)
from akasha_benchmark.io_utils import load_json, sha256_file  # noqa: E402
from akasha_benchmark.store import connect, repo  # noqa: E402


class Report:
    """一个数据集的检查结果。errors 非空即不通过；notes 是实测统计，供人核对。"""

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.errors: list[str] = []
        self.notes: list[str] = []

    def fail(self, message: str) -> None:
        self.errors.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(dataset: str, dataset_dir: Path | None, connection: sqlite3.Connection) -> Report:
    """检查一个数据集。发现问题时尽量继续走完，好一次性报出全部错误。

    ``connection`` 是只读连接：这一步只核对，不写任何东西。
    """
    report = Report(dataset)
    resolved = resolve(dataset, dataset_dir)
    adapter = resolved.adapter

    # --- corpus：行身份与 (title, text) 唯一性（不满足时 load_corpus 直接抛异常）---
    corpus = load_corpus(adapter.name, resolved.corpus_path)
    stats = corpus.dedup_stats()
    report.note(
        f"corpus rows={stats['rows']} unique_titles={stats['unique_titles']} "
        f"unique_pairs={stats['unique_title_text_pairs']} "
        f"rows_in_dup_title_groups={stats['rows_in_duplicate_title_groups']}"
    )
    if stats["unique_title_text_pairs"] != stats["rows"]:
        report.fail(
            f"(title, text) is not unique: {stats['rows']} rows but "
            f"{stats['unique_title_text_pairs']} distinct pairs"
        )

    # --- 每一行原始数据都过一遍适配器 ---
    rows = load_json(resolved.qa_path)
    declares_recall = adapter.has(DataDependency.GOLD_DOCS)
    rederived = []
    seen: dict[str, int] = {}
    gold_dist: Counter[int] = Counter()
    empty_gold: list[int] = []

    for row_index, row in enumerate(rows):
        try:
            sample = adapter.parse_row(row, row_index, corpus)
        except Exception as exc:  # noqa: BLE001 - 记下来继续走完，别只报第一行
            report.fail(f"row {row_index}: {type(exc).__name__}: {exc}")
            continue

        if not sample.dataset_sample_id:
            report.fail(f"row {row_index}: empty dataset_sample_id")
        if sample.sample_id in seen:
            report.fail(
                f"row {row_index}: duplicate sample_id {sample.sample_id!r} "
                f"(first seen at row {seen[sample.sample_id]})"
            )
        seen[sample.sample_id] = row_index

        gold_dist[len(sample.gold_doc_ids)] += 1
        if declares_recall and not sample.gold_doc_ids:
            empty_gold.append(row_index)

        unknown = [d for d in sample.gold_doc_ids if d not in corpus.by_id]
        if unknown:
            report.fail(f"row {row_index}: gold doc_ids absent from corpus: {unknown}")
        if len(set(sample.gold_doc_ids)) != len(sample.gold_doc_ids):
            report.fail(f"row {row_index}: gold_doc_ids contains duplicates (must be a set)")

        rederived.append(sample)

    report.note(f"gold_count_distribution={dict(sorted(gold_dist.items()))}")
    if empty_gold:
        report.fail(
            f"{len(empty_gold)} rows declare GOLD_DOCS but yield no gold "
            f"(first: {empty_gold[:5]})"
        )

    expected = adapter.expected_qa_rows()
    if expected is not None and len(rows) != expected:
        report.fail(f"QA rows={len(rows)}, adapter expects {expected}")

    questions = Counter(s.question for s in rederived)
    duplicates = {q: c for q, c in questions.items() if c > 1}
    report.note(
        f"questions={len(rederived)} unique={len(questions)} duplicate_texts={len(duplicates)}"
    )
    if duplicates:
        # 不算致命错误。但审计归因按 sha256(query) join 审计表，
        # 这些行必须从那个 join 里排除，所以要在这里点出来。
        sample_q = next(iter(duplicates))
        report.note(
            f"WARNING duplicate question text blocks the audit-table join for "
            f"{sum(duplicates.values())} rows, e.g. {sample_q[:70]!r}"
        )

    # --- 库里的产物必须与重新推导的结果一致 ---
    #
    # 库是事实来源，所以这里比对的是 sample / corpus_doc 表，不是 jsonl 文件。
    # 校验的性质没变：**逐行重新推导一遍再比**，这样「归一化跑过之后原始数据
    # 又变了」或者「适配器改过但没重跑」都会在这里暴露，而不是等到指标算出来
    # 才发现数字对不上。
    record = repo.get_dataset(connection, adapter.name)
    if record is None:
        report.fail(
            f"{adapter.name} is not in the database; run "
            "`python -m akasha_benchmark.normalize` first"
        )
        return report

    stored = repo.samples_of(connection, adapter.name)
    if len(stored) != len(rederived):
        report.fail(f"database has {len(stored)} samples, re-derived {len(rederived)}")
    else:
        by_id = {s.sample_id: s for s in rederived}
        for row in stored:
            fresh = by_id.get(row["sample_id"])
            if fresh is None:
                report.fail(f"database has sample_id {row['sample_id']!r} that no longer derives")
                break
            fresh_row = fresh.model_dump(mode="json")
            differing = sorted(
                key
                for key in ("question", "dataset_sample_id")
                if row[key] != fresh_row[key]
            )
            if tuple(row["answers"]) != tuple(fresh_row["answers"]):
                differing.append("answers")
            if tuple(row["gold_doc_ids"]) != tuple(fresh_row["gold_doc_ids"]):
                differing.append("gold_doc_ids")
            if row["metadata"] != fresh_row["metadata"]:
                differing.append("metadata")
            if differing:
                report.fail(
                    f"database drifted from source at sample_id="
                    f"{row['sample_id']!r}, fields={sorted(differing)}"
                )
                break

    stored_corpus_ids = [r["doc_id"] for r in repo.corpus_of(connection, adapter.name)]
    if sorted(stored_corpus_ids) != sorted(d.doc_id for d in corpus.docs):
        report.fail("database corpus doc_ids differ from the re-derived corpus identity")
    report.note(f"corpus rows in database={len(stored_corpus_ids)}")

    # 上游哈希链的起点：库里记的原始文件 sha256 必须仍与磁盘上的一致。
    # 不一致意味着 normalize 之后原始数据又换过，而这一层的下游全部过期。
    for label, path, column in (
        ("qa", resolved.qa_path, "qa_sha256"),
        ("corpus", resolved.corpus_path, "corpus_sha256"),
    ):
        actual = sha256_file(path)
        if record[column] != actual:
            report.fail(
                f"source {label} changed since normalization: "
                f"database {record[column][:12]} != actual {actual[:12]}"
            )

    provides = sorted(d.value for d in adapter.provides)
    stored_provides = repo.loads(record["provides_json"], [])
    if stored_provides != provides:
        report.fail(f"database provides {stored_provides} != adapter {provides}")
    report.note(f"provides={','.join(provides)}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=list(DATASET_NAMES))
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    args = parser.parse_args(argv)

    # 只读连接：这一步只核对产物，不写任何东西。库不存在时给一句清楚的提示,
    # 而不是让 read_only 的 FileNotFoundError 从循环深处冒出来。
    try:
        connection = connect(args.db, read_only=True)
    except FileNotFoundError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1

    try:
        failed = 0
        for name in args.dataset or list(DATASET_NAMES):
            try:
                report = validate(name, args.dataset_dir, connection)
            except Exception as exc:  # noqa: BLE001 - 直接抛出来的硬失败也是一种结果
                failed += 1
                print(f"\n=== {name}\n  ERROR {type(exc).__name__}: {exc}")
                continue

            print(f"\n=== {name}: {'PASS' if report.ok else 'FAIL'}")
            for note in report.notes:
                print(f"  {note}")
            for error in report.errors[:20]:
                print(f"  ERROR {error}")
            if len(report.errors) > 20:
                print(f"  ... and {len(report.errors) - 20} more errors")
            failed += not report.ok

        print(f"\n{'all datasets pass' if not failed else f'{failed} dataset(s) FAILED'}")
        return 1 if failed else 0
    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())
