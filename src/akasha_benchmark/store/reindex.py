"""从磁盘产物重建库里的产物索引。

存在的理由只有一个：``page_map`` 里那些 ``page_id`` 是真实 Akasha 实例里的行,
重新挣一遍要烧约 15 小时编译（1722 篇 × 约 40 秒）。§12.9 的血缘视图要靠它们
跳到 ``knowledge_pages``，所以这批 id 必须能搬进新库。

**只动产物表。** ``annotation`` / ``judge_verdict`` / ``judge_provider`` 一律不碰 ——
它们没有上游可重算，而它们与产物表同库，一个粗心的 ``DELETE FROM`` 就没了
（§12.7）。这条不是靠注释保证的，是靠 :data:`REBUILDABLE_TABLES` 白名单 +
:func:`_assert_protected_untouched` 的前后计数比对保证的。

    uv run python -m akasha_benchmark.store.reindex --run-id run001
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

from ..datasets import DATASET_NAMES, get_adapter
from ..datasets.resolver import DEFAULT_DATA_DIR, normalized_dir, subset_dir
from ..io_utils import load_json, read_jsonl, sha256_text
from ..metrics import registry
from . import identity, repo
from .db import DEFAULT_BATCH, connect

# reindex 允许清空重建的表。不在这个集合里的表，reindex 一行都不准删。
REBUILDABLE_TABLES = frozenset(
    {
        "dataset",
        "sample",
        "corpus_doc",
        "index_layer",
        "index_layer_dataset",
        "subset_sample",
        "subset_doc",
        "page_map",
        "import_failure",
        "compile_run",
        "quality_gate",
        "query_layer",
        "query_response",
        "eval_layer",
        "sample_eval",
        "sample_metric",
        "metric_summary",
        "dataset_eval",
        "audit_record",
    }
)

# 这些表 reindex 前后的行数必须一模一样。判据放在代码里而不是注释里。
PROTECTED_TABLES = ("annotation", "judge_verdict", "judge_provider")


def _counts(connection: sqlite3.Connection, tables: tuple[str, ...]) -> dict[str, int]:
    return {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in tables
    }


def _assert_protected_untouched(
    connection: sqlite3.Connection, before: dict[str, int]
) -> None:
    after = _counts(connection, PROTECTED_TABLES)
    if after != before:
        raise RuntimeError(
            f"reindex changed protected tables: {before} -> {after}. "
            "Annotations and judge verdicts have no upstream to rebuild from."
        )


def reindex_normalized(
    connection: sqlite3.Connection, dataset: str, data_dir: Path | None
) -> dict[str, Any]:
    """导入一个数据集的归一化产物。"""
    source = normalized_dir(dataset, data_dir)
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} missing; run the normalize stage first")
    manifest = load_json(manifest_path)
    adapter = get_adapter(dataset)

    repo.upsert_dataset(
        connection,
        name=dataset,
        adapter=manifest["adapter"],
        adapter_version=str(manifest["adapter_version"]),
        # 以适配器当前的声明为准，不照抄磁盘上的字段：磁盘产物可能早于
        # capability 反转，而权威是代码里的声明。
        provides=sorted(d.value for d in adapter.provides),
        identity_rules=manifest["identity_rules"],
        qa_path=manifest["sources"]["qa"]["path"],
        qa_sha256=manifest["sources"]["qa"]["sha256"],
        qa_rows=manifest["sources"]["qa"]["rows"],
        corpus_path=manifest["sources"]["corpus"]["path"],
        corpus_sha256=manifest["sources"]["corpus"]["sha256"],
        corpus_rows=manifest["sources"]["corpus"]["rows"],
        dedup_stats=manifest["corpus_dedup_stats"],
        gold_count_distribution=manifest["gold_count_distribution"],
        unique_question_texts=manifest["unique_question_texts"],
    )
    samples = repo.replace_samples(connection, dataset, read_jsonl(source / "samples.jsonl"))
    corpus = repo.replace_corpus(
        connection,
        dataset,
        (
            {**doc, "text_sha256": sha256_text(doc["text"])}
            for doc in read_jsonl(source / "corpus.jsonl")
        ),
    )
    connection.commit()
    return {"dataset": dataset, "samples": samples, "corpus": corpus, "manifest": manifest}


def reindex_subset(
    connection: sqlite3.Connection,
    layer_id: int,
    dataset: str,
    run_id: str,
    data_dir: Path | None,
) -> dict[str, Any]:
    """导入一个数据集的子集，md 正文进库。

    md 正文入库是「库当权威」的关键一步：ingest 之后从库里取正文上传，
    磁盘上那 1722 个 .md 就降级成可选导出了。总量 3.8MB，最大单篇 3.5KB。
    """
    source = subset_dir(run_id, dataset, data_dir)
    manifest = load_json(source / "manifest.json")
    hashes: dict[str, str] = manifest["corpus_md_sha256"]
    gold_ids = {
        doc_id
        for row in read_jsonl(source / "samples.jsonl")
        for doc_id in row.get("gold_doc_ids") or []
    }

    docs: list[dict[str, Any]] = []
    corpus_dir = source / "corpus"
    for doc_id, expected in sorted(hashes.items()):
        md_path = corpus_dir / f"{doc_id}.md"
        if not md_path.is_file():
            raise FileNotFoundError(f"{md_path} missing but listed in the subset manifest")
        text = md_path.read_text(encoding="utf-8")
        actual = sha256_text(text)
        # 与抽样时记的 sha256 不符，说明子集产物被改过。这一道在 reindex 就要拦:
        # 让一份对不上的语料进库，后面每个指标都失去可追溯性。
        if actual != expected:
            raise RuntimeError(
                f"{dataset}/{doc_id}.md changed since the subset was built "
                f"(manifest {expected[:12]} != disk {actual[:12]})"
            )
        docs.append(
            {"doc_id": doc_id, "md_text": text, "md_sha256": actual, "is_gold": doc_id in gold_ids}
        )

    sample_ids = [row["sample_id"] for row in read_jsonl(source / "samples.jsonl")]
    # 上游哈希取**库里 dataset 行的当前值**，不取子集 manifest 的 upstream 字段。
    #
    # 那两个字段记的是 samples.jsonl / corpus.jsonl 这两个**产物**的 sha256，
    # 而库里 dataset.qa_sha256 记的是**原始输入**文件的 sha256 —— 拿它们互相比对
    # 是在比两种不同的东西，``stale_upstream`` 会因此永远判为断链。
    #
    # 库当权威之后没有 samples.jsonl 可哈希了，链条的正确口径就是原始输入：
    # dataset.qa_sha256 变了 == normalize 换过数据快照 == 这一层的子集过期。
    # reindex 刚在同一次运行里导入过这份归一化数据，所以此刻取当前值是对的。
    record = repo.get_dataset(connection, dataset) or {}
    repo.upsert_index_layer_dataset(
        connection,
        layer_id,
        dataset,
        strategy=manifest["strategy"],
        qa_count=manifest["qa_count"],
        corpus_count=manifest["corpus_count"],
        gold_doc_count=manifest["gold_doc_count"],
        negative_doc_count=manifest["negative_doc_count"],
        strata=manifest.get("strata") or {},
        normalized_qa_sha256=record["qa_sha256"],
        normalized_corpus_sha256=record["corpus_sha256"],
    )
    repo.replace_subset(connection, layer_id, dataset, sample_ids, docs)
    connection.commit()
    return {"dataset": dataset, "samples": len(sample_ids), "docs": len(docs)}


def reindex_ingest(
    connection: sqlite3.Connection, layer_id: int, run_id: str, data_dir: Path | None
) -> dict[str, Any]:
    """导入 page_map 与入库 manifest。**这是 reindex 存在的主要理由。**"""
    source = (data_dir or DEFAULT_DATA_DIR) / "ingest" / run_id
    manifest_path = source / "manifest.json"
    page_map_path = source / "page_map.jsonl"
    if not manifest_path.is_file() or not page_map_path.is_file():
        return {"page_map_rows": 0, "note": "no ingest artifacts on disk"}

    manifest = load_json(manifest_path)
    workspace = manifest.get("workspace") or {}
    user = manifest.get("user") or {}
    repo.update_index_layer(
        connection,
        layer_id,
        model_configs_json=repo.dumps(manifest.get("model_configs")),
        workspace_id=workspace.get("id"),
        workspace_name=workspace.get("name"),
        akasha_user_id=user.get("id"),
        akasha_user_role=user.get("role"),
        connection_json=repo.dumps(manifest.get("connection")),
        ingested_at=manifest.get("generated_at"),
    )
    for name, space in (manifest.get("spaces") or {}).items():
        repo.set_space(
            connection,
            layer_id,
            name,
            space_id=space["id"],
            space_slug=space.get("slug") or "",
            space_reused=bool(space.get("reused")),
        )

    rows = 0
    connection.execute("DELETE FROM page_map WHERE index_layer_id = ?", (layer_id,))
    for row in read_jsonl(page_map_path):
        repo.record_page(
            connection,
            layer_id,
            row["dataset"],
            doc_id=row["doc_id"],
            page_id=row["page_id"],
            space_id=row.get("space_id") or "",
            title=row.get("title"),
            md_sha256=row.get("md_sha256") or "",
        )
        rows += 1
        if rows % DEFAULT_BATCH == 0:
            connection.commit()
    connection.commit()

    _reindex_gates(connection, layer_id, manifest)
    return {"page_map_rows": rows}


def _reindex_gates(
    connection: sqlite3.Connection, layer_id: int, manifest: dict[str, Any]
) -> None:
    """把编译与质量闸门的记录搬进库。

    ``quality`` 为 null 意味着那趟入库跑的是 ``--skip-compile`` —— 它**不等于**
    质量验收通过（§6.4）。所以这里不写 quality_gate 行，``quality_passed``
    留 NULL，让「没跑过」与「跑过且为 0」在库里也能区分。
    """
    compile_result = manifest.get("compile")
    runs = manifest.get("runs")
    if compile_result or runs:
        repo.record_compile_run(
            connection,
            layer_id,
            accepted_run_count=(compile_result or {}).get("acceptedRunCount"),
            coalesced_run_count=(compile_result or {}).get("coalescedRunCount"),
            status_counts=(runs or {}).get("status_counts"),
            terminal=(runs or {}).get("terminal"),
            timed_out=bool((runs or {}).get("timed_out")),
            requested_at=manifest.get("generated_at") or "",
            finished_at=manifest.get("generated_at"),
        )

    quality = manifest.get("quality")
    if quality:
        passed, _ = repo.record_quality_gate(
            connection, layer_id, gates=quality.get("gates") or {}, report=quality.get("report")
        )
        repo.update_index_layer(connection, layer_id, quality_passed=int(passed))
    connection.commit()


def reindex_responses(
    connection: sqlite3.Connection, layer_id: int, run_id: str, data_dir: Path | None
) -> dict[str, Any]:
    """导入响应，建出对应的查询层。"""
    source = (data_dir or DEFAULT_DATA_DIR) / "responses" / run_id
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        return {"query_layer_id": None, "responses": 0, "note": "no response artifacts on disk"}

    manifest = load_json(manifest_path)
    index_layer = repo.get_index_layer(connection, layer_id) or {}
    index_hash = index_layer.get("config_hash") or ""
    model_configs = manifest.get("model_configs")
    label = f"{run_id}-query"

    existing = repo.query_layer_by_label(connection, label)
    if existing:
        query_layer_id = int(existing["id"])
        connection.execute(
            "DELETE FROM query_response WHERE query_layer_id = ?", (query_layer_id,)
        )
    else:
        query_layer_id = repo.create_query_layer(
            connection,
            index_layer_id=layer_id,
            label=label,
            config_hash=identity.query_layer_hash(
                index_config_hash=index_hash,
                score_threshold=manifest.get("score_threshold"),
                model_configs=model_configs,
            ),
            score_threshold=manifest.get("score_threshold"),
            concurrency=manifest.get("concurrency") or 1,
            request_interval_seconds=manifest.get("request_interval_seconds") or 0.5,
            model_configs=model_configs,
            model_configs_match_index=manifest.get("model_configs_match_ingest"),
            allow_config_drift=False,
        )
    connection.commit()

    total = 0
    for dataset in DATASET_NAMES:
        path = source / f"{dataset}.jsonl"
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            repo.record_response(
                connection,
                query_layer_id,
                sample_id=row["sample_id"],
                dataset=row["dataset"],
                question=row["question"],
                requested_at=row["requested_at"],
                latency_ms=row.get("latency_ms"),
                http_status=row.get("http_status") or 0,
                error=row.get("error"),
                response=row.get("response"),
            )
            total += 1
            if total % DEFAULT_BATCH == 0:
                connection.commit()
    repo.finish_query_layer(connection, query_layer_id)
    connection.commit()
    return {"query_layer_id": query_layer_id, "responses": total}


def _subset_config(run_id: str, datasets: list[str], data_dir: Path | None) -> dict[str, Any]:
    """从各数据集的子集 manifest 反推这一层的抽样配置。

    四组的 seed / qa_limit / negatives_ratio 必须一致，否则它们不构成同一个
    索引层 —— 那种情况下当初就该分成两层跑。
    """
    seen: dict[str, set[Any]] = {"seed": set(), "qa_limit": set(), "negatives_ratio": set()}
    found = False
    for dataset in datasets:
        path = subset_dir(run_id, dataset, data_dir) / "manifest.json"
        if not path.is_file():
            continue
        manifest = load_json(path)
        found = True
        for key in seen:
            seen[key].add(manifest[key])
    if not found:
        raise FileNotFoundError(
            f"no subset manifests for run {run_id!r}; run the subset stage first"
        )
    conflicting = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    if conflicting:
        raise RuntimeError(
            f"subset manifests disagree on {conflicting}; these datasets are not one "
            "index layer. Rebuild them under separate run ids."
        )
    return {key: next(iter(values)) for key, values in seen.items()}


def _narrativeqa_doc_count(run_id: str, data_dir: Path | None) -> int:
    """narrativeqa 抽了几篇**文档**（不是几个 chunk）。

    它的 manifest 里 ``strata`` 是 ``{document_id: chunk 数}``，所以键的个数就是
    文档数。这份数据是整篇整篇取的，chunk 数由文档决定而不是由配置决定,
    所以索引层的身份要记文档数。数据集不在本次范围内时返回 0。
    """
    path = subset_dir(run_id, "narrativeqa", data_dir) / "manifest.json"
    if not path.is_file():
        return 0
    return len(load_json(path).get("strata") or {})


def run(
    run_id: str,
    datasets: list[str],
    db_path: Path | None,
    data_dir: Path | None,
    label: str | None = None,
) -> int:
    """从磁盘产物重建索引。返回退出码。"""
    connection = connect(db_path)
    try:
        protected_before = _counts(connection, PROTECTED_TABLES)
        repo.sync_metric_definitions(connection, registry.as_rows())
        connection.commit()

        available = [
            name
            for name in datasets
            if (normalized_dir(name, data_dir) / "manifest.json").is_file()
        ]
        if not available:
            print("no normalized artifacts found on disk", file=sys.stderr)
            return 1

        for dataset in available:
            result = reindex_normalized(connection, dataset, data_dir)
            print(
                f"normalized {dataset:<16} samples={result['samples']:<5} "
                f"corpus={result['corpus']}"
            )

        in_subset = [
            name
            for name in available
            if (subset_dir(run_id, name, data_dir) / "manifest.json").is_file()
        ]
        if not in_subset:
            print(f"\nno subset artifacts for run {run_id!r}; stopped after normalized data")
            _assert_protected_untouched(connection, protected_before)
            return 0

        config = _subset_config(run_id, in_subset, data_dir)
        # 模型配置在入库 manifest 里，而它决定索引层的 config_hash，
        # 所以要在建层之前读出来。
        ingest_manifest_path = (data_dir or DEFAULT_DATA_DIR) / "ingest" / run_id / "manifest.json"
        model_configs = (
            load_json(ingest_manifest_path).get("model_configs")
            if ingest_manifest_path.is_file()
            else None
        )
        narrativeqa_docs = _narrativeqa_doc_count(run_id, data_dir)

        layer_label = label or run_id
        existing = repo.index_layer_by_label(connection, layer_label)
        if existing:
            layer_id = int(existing["id"])
            print(f"\nreusing index layer #{layer_id} ({layer_label})")
        else:
            layer_id = repo.create_index_layer(
                connection,
                label=layer_label,
                # 内容寻址，抽完（这里是导完）才算。reindex 导入的是历史产物,
                # 凭配置算的哈希与实际导进来的那批文档没有因果关系。
                subset_hash="",
                seed=config["seed"],
                qa_limit=config["qa_limit"],
                negatives_ratio=config["negatives_ratio"],
                narrativeqa_docs=narrativeqa_docs,
            )
            connection.commit()
            print(f"\ncreated index layer #{layer_id} ({layer_label})")

        for dataset in in_subset:
            result = reindex_subset(connection, layer_id, dataset, run_id, data_dir)
            print(f"subset     {dataset:<16} qa={result['samples']:<5} docs={result['docs']}")
        digest = repo.recompute_subset_hash(connection, layer_id)
        repo.update_index_layer(
            connection, layer_id, subset_built_at=load_json(
                subset_dir(run_id, in_subset[0], data_dir) / "manifest.json"
            )["generated_at"]
        )
        # 磁盘上已经有入库产物，所以模型配置是已知的，config_hash 可以当场补齐。
        # 必须在建查询层之前做完 —— 查询层的哈希要吃它。
        if model_configs is not None:
            repo.seal_index_layer(
                connection,
                layer_id,
                identity.index_layer_hash(subset_hash=digest, model_configs=model_configs),
            )
        connection.commit()
        print(f"subset_hash  {digest}（按实际文档集内容寻址）")

        ingested = reindex_ingest(connection, layer_id, run_id, data_dir)
        print(f"page_map   rows={ingested['page_map_rows']}")

        responses = reindex_responses(connection, layer_id, run_id, data_dir)
        print(
            f"responses  rows={responses['responses']} "
            f"query_layer=#{responses['query_layer_id']}"
        )

        _assert_protected_untouched(connection, protected_before)
        print(f"\nprotected tables untouched: {protected_before}")
        return 0
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", action="append", choices=list(DATASET_NAMES))
    parser.add_argument("--label", default=None, help="索引层的标签，默认用 run-id")
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        return run(
            args.run_id,
            args.dataset or list(DATASET_NAMES),
            args.db,
            args.data_dir,
            args.label,
        )
    except (RuntimeError, ValueError, FileNotFoundError, sqlite3.Error) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
