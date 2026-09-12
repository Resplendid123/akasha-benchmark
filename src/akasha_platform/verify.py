"""平台小样本验证：抽样、入库编译、响应校验、续跑检查和评测。"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

from akasha_benchmark import evaluate, ingest, run_args, run_queries, subset
from akasha_benchmark.akasha_client import AkashaClient
from akasha_benchmark.config import load_config
from akasha_benchmark.store import connect, repo

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_response(body: Any, page_ids: set[str]) -> None:
    """验证评测依赖的响应结构和文档映射；失败时给出可定位的错误。"""
    require(isinstance(body, dict), "response must be an object")
    require(isinstance(body.get("answer"), str), "answer must be a string")
    mode = body.get("answerMode")
    require(mode in {"knowledge", "general", "no_match"}, "unknown answerMode")
    fields = ("retrievedSources", "citations", "citationEvidence", "snippets")
    for field in (*fields, "warnings"):
        require(isinstance(body.get(field), list), f"{field} must be an array")
    ids = {}
    for field in fields[:3]:
        entries = body[field]
        require(
            all(
                isinstance(item, dict) and isinstance(item.get("sourcePageId"), str)
                for item in entries
            ),
            f"{field}: missing sourcePageId",
        )
        ids[field] = {item["sourcePageId"] for item in entries}
        require(ids[field] <= page_ids, f"{field}: unmapped sourcePageId")
    require(
        ids["citations"] <= ids["retrievedSources"],
        "citations outside retrievedSources",
    )
    require(
        ids["citations"] == ids["citationEvidence"]
        and len(body["citations"]) == len(body["citationEvidence"]),
        "citationEvidence does not match citations",
    )
    if mode == "knowledge":
        require(
            bool(body["retrievedSources"]), "knowledge response has no retrievedSources"
        )
    else:
        require(
            all(not body[field] for field in fields),
            "fallback response has nonempty sources",
        )
    require("[[cite:" not in body["answer"], "answer contains citation markers")
    budget = body.get("budget")
    require(isinstance(budget, dict), "budget must be an object")
    for key in (
        "maxContextLength",
        "responseReserve",
        "perItemMaxLength",
        "usedContextLength",
        "includedItemCount",
        "omittedItemCount",
    ):
        require(
            type(budget.get(key)) is int and budget[key] >= 0, f"invalid budget.{key}"
        )
    require(
        budget["usedContextLength"] <= budget["maxContextLength"],
        "context exceeds budget",
    )
    reasons = set()
    for snippet in body["snippets"]:
        require(isinstance(snippet, dict), "snippet must be an object")
        try:
            uuid.UUID(snippet.get("id", ""))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("snippet.id must be a UUID") from exc
        values = snippet.get("retrievalReasons")
        require(
            isinstance(values, list) and all(isinstance(v, str) for v in values),
            "snippet.retrievalReasons must be a string array",
        )
        reasons.update(values)
    top_reasons = body.get("retrievalReasons", [])
    require(
        isinstance(top_reasons, list) and all(isinstance(v, str) for v in top_reasons),
        "retrievalReasons must be a string array",
    )
    require(
        set(top_reasons) == reasons, "retrievalReasons differs from snippet reasons"
    )


def run(db_path: Path, dataset: str, samples: int = 3) -> int:
    require(dataset in DATASETS, f"dataset must be one of {DATASETS}")
    require(1 <= samples <= 5, "samples must be between 1 and 5")
    connection = connect(db_path)
    try:
        config = load_config(connection)
        config.require_credentials()
        require(
            repo.get_dataset(connection, dataset) is not None,
            "请先在归一化页面处理该数据集",
        )
        label = f"verify{uuid.uuid4().hex[:12]}"
        query_label, eval_label = f"{label}-query", f"{label}-eval"
        print(
            f"小样本验证 {label}: {dataset}, {samples} 条；产物保留在平台中", flush=True
        )
        code = subset.main(
            [
                "--db",
                str(db_path),
                "--label",
                label,
                "--dataset",
                dataset,
                "--qa-limit",
                str(samples),
                "--negatives-ratio",
                "1",
            ]
        )
        require(code == 0, "抽样失败")
        layer = repo.index_layer_by_label(connection, label)
        layer_id = int(layer["id"])
        sample_ids = {
            s["sample_id"] for s in repo.subset_samples(connection, layer_id, dataset)
        }
        docs = repo.subset_docs(connection, layer_id, dataset, with_text=False)
        require(len(sample_ids) == samples, "可用样本不足")
        require(0 < len(docs) <= 50, "验证语料需在 1–50 篇以内")
        require(ingest.run(label, [dataset], db_path) == 0, "入库或编译质量闸门失败")
        require(repo.index_layer_readiness(connection, layer_id)["ready"], "索引未就绪")
        mappings = list(
            connection.execute(
                "SELECT doc_id, page_id FROM page_map WHERE index_layer_id=? AND dataset=? ORDER BY doc_id",
                (layer_id, dataset),
            )
        )
        require(
            {r["doc_id"] for r in mappings} == {d["doc_id"] for d in docs},
            "入库映射不完整",
        )
        page_ids = {r["page_id"] for r in mappings}
        require(len(page_ids) == len(docs), "多个文档映射到同一页面")
        # 只重跑导入，避免为验证续跑再次触发编译或改变质量闸门。
        with AkashaClient(config) as client:
            client.login()
            stats = ingest.import_corpus(
                client,
                connection,
                layer_id,
                dataset,
                repo.spaces_of(connection, layer_id)[dataset],
            )
        require(
            stats["imported"] == stats["failures"] == 0
            and stats["skipped_already_present"] == len(docs),
            "入库续跑重复导入或失败",
        )
        print("入库映射、质量闸门、导入续跑：通过", flush=True)

        def query() -> None:
            require(
                run_queries.run(
                    label,
                    [dataset],
                    db_path,
                    None,
                    None,
                    False,
                    query_label=query_label,
                )
                == 0,
                "查询失败",
            )

        query()
        query_layer = repo.query_layer_by_label(connection, query_label)
        qid = int(query_layer["id"])
        before = repo.responses_of(connection, qid, dataset)
        require({r["sample_id"] for r in before} == sample_ids, "查询响应不完整")
        for row in before:
            require(
                200 <= row["http_status"] < 300 and not row["error"],
                f"{row['sample_id']}: HTTP 查询失败",
            )
            try:
                validate_response(row["response"], page_ids)
            except ValueError as exc:
                raise ValueError(f"{row['sample_id']}: {exc}") from exc
        query()
        require(
            repo.responses_of(connection, qid, dataset) == before,
            "查询续跑改写了已有响应",
        )
        print("响应契约、来源映射、查询续跑：通过", flush=True)
        require(
            evaluate.run(
                query_label, [dataset], db_path, (2, 5, 10), eval_label=eval_label
            )
            == 0,
            "评测失败",
        )
        eid = int(repo.eval_layer_by_label(connection, eval_label)["id"])
        require(
            {r["sample_id"] for r in repo.sample_evals(connection, eid)} == sample_ids,
            "报告未覆盖全部样本",
        )
        print(
            f"验证通过。编译层 #{layer_id}，查询层 #{qid}，报告 #{eid} ({eval_label})",
            flush=True,
        )
        return 0
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, default="hotpotqa")
    parser.add_argument("--samples", type=int, default=3)
    run_args.add_argument(parser)
    args = run_args.apply(parser.parse_args(argv), stage="verify")
    try:
        return run(args.db, args.dataset, args.samples)
    except Exception as exc:
        print(f"验证失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
