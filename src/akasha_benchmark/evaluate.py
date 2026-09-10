"""评测：从落盘的响应算指标。纯离线，不再碰 Akasha。

每个检索指标都出两份 —— 全样本，以及只算 ``answerMode == "knowledge"`` 的切片。
原因是 ``no_match`` 和 ``general`` 两种模式无条件返回
``retrievedSources: []``（``ai-knowledge-chat.service.ts:641,667``），
不管检索实际找到了什么。所以全样本那份数字把「生成端拒答」也算进了检索指标里，
两份的差值就是这个效应的规模。

    uv run python -m akasha_benchmark.evaluate --run-id run001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .datasets import DATASET_NAMES, Capability, CanonicalSample, get_adapter, subset_dir
from .datasets.resolver import DEFAULT_DATA_DIR
from .ingest import ingest_dir
from .io_utils import atomic_write_json, atomic_write_jsonl, atomic_write_text, load_json, read_jsonl, utc_now
from .metrics import attribution, multihop, qa, retrieval
from .run_queries import responses_dir

# 各数据集用哪个 metadata 字段做分层报告。
# musique 有真正的跳数，其余三组只能按题型/文档类型切。
STRATIFY_KEYS = {
    "hotpotqa": "type",
    "2wikimultihopqa": "type",
    "musique": "hop_count",
    "narrativeqa": "kind",
}


def reports_dir(run_id: str, data_dir: Path | None = None) -> Path:
    return (data_dir or DEFAULT_DATA_DIR) / "reports" / run_id


def _page_to_doc(run_id: str, dataset: str, data_dir: Path | None) -> dict[str, str]:
    """读入库阶段的 page_map，建 page_id -> doc_id 的反查表。"""
    path = ingest_dir(run_id, data_dir) / "page_map.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; the ingest stage has not produced a page map")
    mapping: dict[str, str] = {}
    for row in read_jsonl(path):
        if row.get("dataset") == dataset:
            mapping[row["page_id"]] = row["doc_id"]
    if not mapping:
        raise ValueError(f"page_map has no rows for dataset {dataset!r}")
    return mapping


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_dataset(
    dataset: str, run_id: str, data_dir: Path | None, ks: tuple[int, ...]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """算一个数据集的指标，返回 (汇总, 逐样本明细)。"""
    adapter = get_adapter(dataset)
    # 没声明 EVIDENCE_RECALL 的数据集（narrativeqa）跳过全部检索指标，
    # 不是算成 0，见下方的 retrieval_note。
    has_gold = adapter.supports(Capability.EVIDENCE_RECALL)

    samples = {
        s.sample_id: s
        for s in (
            CanonicalSample.model_validate(row)
            for row in read_jsonl(subset_dir(run_id, dataset, data_dir) / "samples.jsonl")
        )
    }
    response_path = responses_dir(run_id, data_dir) / f"{dataset}.jsonl"
    if not response_path.is_file():
        raise FileNotFoundError(f"{response_path} missing; run the query stage first")

    page_to_doc = _page_to_doc(run_id, dataset, data_dir) if has_gold else {}

    per_sample: list[dict[str, Any]] = []
    seen: set[str] = set()
    http_failures = 0
    unmapped_pages: set[str] = set()

    for row in read_jsonl(response_path):
        sample_id = row["sample_id"]
        if sample_id in seen:
            raise ValueError(f"{response_path}: duplicate sample_id {sample_id!r}")
        seen.add(sample_id)

        sample = samples.get(sample_id)
        if sample is None:
            raise ValueError(
                f"{response_path}: sample_id {sample_id!r} is not in the subset; "
                "the response file and the subset are from different runs"
            )
        # PLAN.md 3.4：按 ID 匹配后**再比一次 question 文本**。
        # 这能抓到「响应文件和数据集版本不匹配」——ID 对得上但内容已经变了。
        if row.get("question") != sample.question:
            raise ValueError(
                f"{sample_id}: question text differs between response file and subset. "
                "These artifacts are not from the same dataset snapshot."
            )

        status = row.get("http_status") or 0
        body = row.get("response") or {}
        # 失败行照样参与统计（F1 记 0），因为失败率本身是结果的一部分。
        ok = 200 <= status < 300 and isinstance(body, dict)
        if not ok:
            http_failures += 1

        answer_mode = body.get("answerMode") if ok else None
        retrieved = body.get("retrievedSources") or [] if ok else []
        citations = body.get("citations") or [] if ok else []
        citation_evidence = body.get("citationEvidence") or [] if ok else []
        snippets = body.get("snippets") or [] if ok else []

        entry: dict[str, Any] = {
            "sample_id": sample_id,
            "dataset": dataset,
            "metadata": dict(sample.metadata),
            "http_status": status,
            "ok": ok,
            "answer_mode": answer_mode,
            "gold_count": len(sample.gold_doc_ids),
            "retrieved_count": len(retrieved),
            "citation_count": len(citations),
            "latency_ms": row.get("latency_ms"),
            "warnings": body.get("warnings") if ok else None,
            "completeness_notice": body.get("completenessNotice") if ok else None,
            "budget": body.get("budget") if ok else None,
        }

        answer = body.get("answer") or "" if ok else ""
        entry["qa"] = qa.score_answer(answer, sample.answers)

        if has_gold:
            unmapped_pages.update(retrieval.unmapped_page_ids(retrieved, page_to_doc))
            ranked = retrieval.ranked_doc_ids(retrieved, page_to_doc)
            entry["retrieval"] = retrieval.evaluate_sample(ranked, sample.gold_doc_ids, ks)
            entry["attribution"] = attribution.evaluate_sample(
                citations, retrieved, citation_evidence, sample.gold_doc_ids, page_to_doc
            )
            entry["multihop"] = multihop.evaluate_sample(snippets, sample.gold_doc_ids, page_to_doc)

        per_sample.append(entry)

    missing = sorted(set(samples) - seen)
    knowledge_rows = [e for e in per_sample if e["answer_mode"] == "knowledge"]

    summary: dict[str, Any] = {
        "dataset": dataset,
        "capabilities": sorted(c.value for c in adapter.capabilities),
        "samples_in_subset": len(samples),
        "responses_evaluated": len(per_sample),
        "missing_responses": missing,
        "http_failures": http_failures,
        "answer_mode_distribution": qa.answer_mode_distribution(
            [e["answer_mode"] for e in per_sample]
        ),
        # EM 预期恒为 0（散文答案对不上短跨度参考），当形态探针读；F1 才是质量指标。
        # 见 metrics/qa.py 与 metrics.md「答案质量」。
        "qa": {
            "em": _mean([e["qa"]["em"] for e in per_sample]),
            "em_knowledge_only": _mean([e["qa"]["em"] for e in knowledge_rows]),
            "f1": _mean([e["qa"]["f1"] for e in per_sample]),
            "f1_knowledge_only": _mean([e["qa"]["f1"] for e in knowledge_rows]),
        },
        "latency_ms_mean": _mean([float(e["latency_ms"] or 0) for e in per_sample]),
    }

    if has_gold:
        summary["retrieval"] = retrieval.aggregate([e["retrieval"] for e in per_sample])
        summary["retrieval_knowledge_only"] = retrieval.aggregate(
            [e["retrieval"] for e in knowledge_rows]
        )
        summary["knowledge_answer_count"] = len(knowledge_rows)
        summary["attribution"] = attribution.aggregate([e["attribution"] for e in per_sample])
        summary["multihop"] = multihop.aggregate([e["multihop"] for e in per_sample])
        summary["unmapped_page_ids"] = sorted(unmapped_pages)

        # 分层衰减曲线：musique 按跳数，hotpotqa / 2wiki 按题型。
        key = STRATIFY_KEYS[dataset]
        strata: dict[str, Any] = {}
        for name, rows in multihop.stratify(per_sample, key).items():
            knowledge = [r for r in rows if r["answer_mode"] == "knowledge"]
            strata[name] = {
                "count": len(rows),
                "retrieval": retrieval.aggregate([r["retrieval"] for r in rows]),
                "multihop": multihop.aggregate([r["multihop"] for r in rows]),
                "qa": {
                    "em": _mean([r["qa"]["em"] for r in rows]),
                    "f1": _mean([r["qa"]["f1"] for r in rows]),
                },
                "knowledge_answer_share": len(knowledge) / len(rows) if rows else 0.0,
            }
        summary["stratified"] = {"key": key, "strata": strata}
    else:
        summary["retrieval_note"] = (
            f"{dataset} declares no EVIDENCE_RECALL capability: no gold documents exist, "
            "so retrieval metrics are omitted rather than reported as 0."
        )

    return summary, per_sample


def _format_report(run_id: str, summaries: list[dict[str, Any]], context: dict[str, Any]) -> str:
    """生成人读的 report.md。

    开头那段架构说明是必须的，不是客套：不写清楚「召回跑在生成文本上」，
    读者会拿这些数字直接跟公开 baseline 比，而那个比较是无效的。
    """
    lines = [
        f"# Akasha-Benchmark 评测报告 — {run_id}",
        "",
        f"生成于 {context['generated_at']}。",
        "",
        "## 这些数字该怎么读",
        "",
        "Akasha 的向量召回与词法召回跑在**编译产物**上，而不是原始文档：",
        "`knowledge_chunks` 索引的是 `artifact.markdown`。原文存在",
        "`knowledge_source_chunks` 里，它不参与召回，只在解析引用时提供证据窗口。",
        "两个后果：Recall@k 被系统性压低，且这种压低不是调参能补回来的；多跳表现",
        "则可能被高估，因为编译器会把跨文档的实体合并进同一份产物。",
        "**因此把这里的数字直接与公开 baseline 对比是无效的。**唯一有意义的对照",
        "是原文基线（raw-text baseline）。",
        "",
        "`no_match` 和 `general` 两种回答返回空的 `retrievedSources`，它们的检索得分",
        "按定义就是 0。所以每张检索表都给两份：一份是全样本，一份只算 `knowledge`",
        "回答。两者之差反映的是生成侧的拒答，不是检索失败。",
        "",
        "**这里的 Exact Match 预期就是 0.0000，这不是故障。**",
        "EM 要求归一化后的整段答案与参考答案完全相等，而 Akasha 用解释性散文作答，",
        "这些数据集的参考答案却是短跨度，两者不可能相等。实测过：三条内容正确、",
        "且 gold 全部召回（`recall@10` = `full_coverage@10` = 1.000）的答案，EM 仍是 0.000。",
        "把 EM 当作答案**形状**的探针，而不是答案质量的度量：它变成非零意味着生成侧",
        "开始输出短跨度，而不是答案变好了。Answer F1 被同样的冗长度稀释（精确率被",
        "压垮，因为分母是 20-40 个散文 token，而参考答案只有 1-5 个 token），所以它",
        "只能用于比较本系统的不同配置，绝不能与公开数字对比。",
        "",
        "## 模型配置",
        "",
        "```json",
        context["model_configs_excerpt"],
        "```",
        "",
    ]

    for summary in summaries:
        lines += [
            f"## {summary['dataset']}",
            "",
            f"- 样本数：{summary['responses_evaluated']}/{summary['samples_in_subset']}"
            f"（HTTP 失败：{summary['http_failures']}）",
            f"- 回答模式分布：{summary['answer_mode_distribution']}",
            f"- answer EM：{summary['qa']['em']:.4f}"
            f"（仅 knowledge：{summary['qa']['em_knowledge_only']:.4f}）"
            " — 预期为 0，原因见上",
            f"- answer F1：{summary['qa']['f1']:.4f}"
            f"（仅 knowledge：{summary['qa']['f1_knowledge_only']:.4f}）",
        ]
        if "retrieval" in summary:
            overall, knowledge = summary["retrieval"], summary["retrieval_knowledge_only"]
            lines += [
                "",
                "| 指标 | 全样本 | 仅 knowledge |",
                "| --- | --- | --- |",
            ]
            for key in sorted(overall):
                lines.append(f"| {key} | {overall[key]:.4f} | {knowledge.get(key, 0.0):.4f} |")
            multi = summary["multihop"]
            lines += [
                "",
                f"- 图邻居占比：{multi['graph_neighbor_share']:.4f}；"
                f"精确率 {multi['graph_neighbor_precision']:.4f}",
                f"- 仅靠图扩展才能到达的 gold 占比："
                f"{multi['graph_exclusive_gold_share']:.4f}",
                "- 各信号的 gold 命中率："
                + "、".join(f"{r}={v:.3f}" for r, v in multi["reason_gold_rate"].items()),
                f"- 引用精确率/召回率："
                f"{summary['attribution']['citation_precision']:.4f} /"
                f" {summary['attribution']['citation_recall']:.4f}",
                f"- 截断损失：{summary['attribution']['truncation_loss']:.2f} 篇，"
                f"其中 gold {summary['attribution']['truncated_gold']:.2f} 篇",
                f"- 证据可验证的引用占比："
                f"{summary['attribution']['evidence_verifiable_rate']:.4f}",
                "",
                f"### 按 `{summary['stratified']['key']}` 分层",
                "",
                "| 分层 | n | recall@10 | full_coverage@10 | EM | F1 | knowledge 占比 |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
            for name, bucket in summary["stratified"]["strata"].items():
                lines.append(
                    f"| {name} | {bucket['count']} |"
                    f" {bucket['retrieval'].get('recall@10', 0.0):.4f} |"
                    f" {bucket['retrieval'].get('full_coverage@10', 0.0):.4f} |"
                    f" {bucket['qa']['em']:.4f} |"
                    f" {bucket['qa']['f1']:.4f} |"
                    f" {bucket['knowledge_answer_share']:.4f} |"
                )
            if summary["unmapped_page_ids"]:
                lines += [
                    "",
                    f"> 有 {len(summary['unmapped_page_ids'])} 个被检索到的 page id 不在"
                    " `page_map` 里。它们仍占据排名位次，但永远不可能被算作 gold。",
                ]
        else:
            lines += ["", f"> {summary['retrieval_note']}"]
        lines.append("")

    return "\n".join(lines) + "\n"


def run(
    run_id: str, datasets: list[str], data_dir: Path | None, ks: tuple[int, ...]
) -> int:
    """执行评测，产出 metrics.json / per_sample.jsonl / report.md。"""
    out_dir = reports_dir(run_id, data_dir)
    response_manifest_path = responses_dir(run_id, data_dir) / "manifest.json"
    response_manifest = (
        load_json(response_manifest_path) if response_manifest_path.is_file() else {}
    )

    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    failures = 0
    for dataset in datasets:
        try:
            summary, rows = evaluate_dataset(dataset, run_id, data_dir, ks)
        except FileNotFoundError as exc:
            # 该数据集没跑过查询，跳过而不算失败。
            print(f"skip {dataset}: {exc}", file=sys.stderr)
            continue
        except Exception as exc:  # noqa: BLE001 - 报告后继续做下一个数据集
            failures += 1
            print(f"FAIL {dataset}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        summaries.append(summary)
        all_rows.extend(rows)

    if not summaries:
        print("no datasets evaluated", file=sys.stderr)
        return 1

    context = {
        "generated_at": utc_now(),
        "model_configs_excerpt": str(response_manifest.get("model_configs", "unavailable"))[:2000],
    }
    metrics = {
        "stage": "evaluate",
        "run_id": run_id,
        "generated_at": context["generated_at"],
        "ks": list(ks),
        "query_manifest": {
            "model_configs_match_ingest": response_manifest.get("model_configs_match_ingest"),
            "total_failures": response_manifest.get("total_failures"),
            "score_threshold": response_manifest.get("score_threshold"),
        },
        "datasets": summaries,
    }
    atomic_write_json(out_dir / "metrics.json", metrics)
    atomic_write_jsonl(out_dir / "per_sample.jsonl", all_rows)
    atomic_write_text(out_dir / "report.md", _format_report(run_id, summaries, context))

    for summary in summaries:
        line = (
            f"ok   {summary['dataset']:<16} n={summary['responses_evaluated']:<4} "
            f"EM={summary['qa']['em']:.3f} F1={summary['qa']['f1']:.3f}"
        )
        if "retrieval" in summary:
            line += (
                f" R@10={summary['retrieval'].get('recall@10', 0.0):.3f}"
                f" (knowledge-only {summary['retrieval_knowledge_only'].get('recall@10', 0.0):.3f})"
            )
        print(line)
    print(f"\nwrote {out_dir}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", action="append", choices=list(DATASET_NAMES))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--k", action="append", type=int, default=None)
    args = parser.parse_args(argv)

    ks = tuple(args.k) if args.k else retrieval.DEFAULT_KS
    return run(args.run_id, args.dataset or list(DATASET_NAMES), args.data_dir, ks)


if __name__ == "__main__":
    sys.exit(main())
