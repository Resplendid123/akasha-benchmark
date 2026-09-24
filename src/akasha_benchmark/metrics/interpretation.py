"""把单样本指标翻译成可读结论。"""

from __future__ import annotations

from collections import Counter
from math import log2
from typing import Any

from . import qa, registry

FAMILY_LABELS = {
    registry.FAMILY_RETRIEVAL: "检索质量",
    registry.FAMILY_QA: "答案质量",
    registry.FAMILY_ATTRIBUTION: "引用归因",
    registry.FAMILY_MULTIHOP: "多跳能力",
    registry.FAMILY_JUDGE: "模型评判",
}

COUNT_METRICS = {
    "uncited_count",
    "uncited_gold_count",
    "graph_neighbor_gold_snippets",
}


def requested_metric_names(metrics: list[str], ks: list[int]) -> list[str]:
    """把评测配置里的模板指标展开为实际逐样本指标名。"""
    names: list[str] = []
    for name in metrics:
        try:
            definition = registry.get_metric(name)
        except KeyError:
            continue
        names.extend(f"{name}@{k}" for k in ks) if definition.per_k else names.append(name)
    return names


def interpret_sample_metrics(
    configured_metrics: list[str],
    ks: list[int],
    values: dict[str, float],
    detail: dict[str, Any],
    verdicts: list[dict[str, Any]],
    omitted_metrics: list[str],
    evidence_by_metric: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """返回评测所选每项指标在当前样本上的定义、状态和具体解读。"""
    names = requested_metric_names(configured_metrics, ks)
    names.extend(
        name
        for name in values
        if name not in names and name in registry.METRIC_REGISTRY
    )
    verdict_by_metric = {row["metric"]: row for row in verdicts}
    omitted = set(omitted_metrics)
    return [
        _interpret(
            name,
            values.get(name),
            detail,
            verdict_by_metric.get(name),
            registry.get_metric(name).name in omitted,
            (evidence_by_metric or {}).get(name),
        )
        for name in names
    ]


def build_metric_evidence(
    configured_metrics: list[str],
    ks: list[int],
    values: dict[str, float],
    detail: dict[str, Any],
    response: dict[str, Any],
    page_to_doc: dict[str, str],
    documents: dict[str, dict[str, Any]],
    verdicts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """为每个指标组装与实际计分口径一致的输入、公式与逐项证据。"""
    names = requested_metric_names(configured_metrics, ks)
    names.extend(
        name for name in values if name not in names and name in registry.METRIC_REGISTRY
    )
    gold_ids = list(detail.get("gold_doc_ids") or [])
    gold_set = set(gold_ids)
    citations = _document_rows(
        response.get("citations") or [], page_to_doc, documents, gold_set
    )
    retrieved = _document_rows(
        response.get("retrievedSources") or [], page_to_doc, documents, gold_set
    )
    citation_ids = {row["doc_id"] for row in citations if row["mapped"]}
    retrieved_ids = {row["doc_id"] for row in retrieved if row["mapped"]}
    gold_documents = [
        {
            "doc_id": doc_id,
            "page_id": next(
                (page for page, mapped in page_to_doc.items() if mapped == doc_id), None
            ),
            "title": (documents.get(doc_id) or {}).get("title") or doc_id,
            "retrieved": doc_id in retrieved_ids,
            "cited": doc_id in citation_ids,
        }
        for doc_id in gold_ids
    ]
    snippet_rows = _snippet_rows(
        response.get("snippets") or [], page_to_doc, gold_set
    )
    citation_excerpts = _citation_excerpt_rows(
        response.get("citationEvidence") or [], page_to_doc, documents, gold_set
    )
    verdict_by_metric = {row["metric"]: row for row in verdicts}
    answer = str(response.get("answer") or "")
    references = [str(value) for value in detail.get("reference_answers") or []]

    return {
        name: _evidence_for_metric(
            name,
            values.get(name),
            retrieved,
            citations,
            gold_documents,
            snippet_rows,
            citation_excerpts,
            answer,
            references,
            verdict_by_metric.get(name),
        )
        for name in names
    }


def _evidence_for_metric(
    name: str,
    value: float | None,
    retrieved: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    gold_documents: list[dict[str, Any]],
    snippets: list[dict[str, Any]],
    citation_excerpts: list[dict[str, Any]],
    answer: str,
    references: list[str],
    verdict: dict[str, Any] | None,
) -> dict[str, Any]:
    base, _, k_text = name.partition("@")
    k = int(k_text) if k_text.isdigit() else None
    gold_count = len(gold_documents)
    valid_retrieved = [row for row in retrieved if row["mapped"]]
    valid_citations = [row for row in citations if row["mapped"]]
    cited_gold = {row["doc_id"] for row in valid_citations if row["is_gold"]}
    evidence: dict[str, Any] = {"formula": None, "gold_documents": gold_documents}
    if base in {"precision", "recall", "retrieval_f1", "hit", "full_coverage", "ndcg"}:
        top = retrieved[: k or 0]
        top_gold = {row["doc_id"] for row in top if row["is_gold"]}
        top_pages = {row["page_id"] for row in top}
        evidence.update(
            documents=top,
            gold_documents=gold_documents,
            snippets=[row for row in snippets if set(row["page_ids"]) & top_pages],
        )
        if base == "precision":
            evidence["formula"] = f"前 {k} 条中命中 Gold（{len(top_gold)}）/ 实际返回文档（{len(top)}）"
        elif base == "retrieval_f1":
            precision = len(top_gold) / len(top) if top else 0.0
            recall = len(top_gold) / gold_count if gold_count else 0.0
            evidence["formula"] = f"2 × Precision（{precision:.4f}）× Recall（{recall:.4f}）/（Precision + Recall）"
        elif base == "recall":
            evidence["formula"] = (
                f"前 {k} 条中命中 Gold（{len(top_gold)}）/ 实际需要的 Gold（{gold_count}）"
            )
        elif base == "hit":
            evidence["formula"] = (
                f"前 {k} 条中命中 Gold（{len(top_gold)}），至少命中一篇记为 1"
            )
        elif base == "full_coverage":
            evidence["formula"] = (
                f"前 {k} 条中命中 Gold（{len(top_gold)}）/ 实际需要的 Gold（{gold_count}），全部命中记为 1"
            )
        else:
            contributions = [
                {
                    "rank": row["rank"],
                    "doc_id": row["doc_id"],
                    "gain": 1 / log2(row["rank"] + 1) if row["is_gold"] else 0.0,
                }
                for row in top
            ]
            dcg = sum(row["gain"] for row in contributions)
            ideal = sum(1 / log2(rank + 1) for rank in range(1, min(gold_count, k or 0) + 1))
            evidence.update(
                formula=f"实际排序加权值 DCG（{dcg:.4f}）/ 理想排序加权值 IDCG（{ideal:.4f}）",
                contributions=contributions,
            )
    elif base == "mrr":
        first = next((row for row in retrieved if row["is_gold"]), None)
        evidence.update(
            formula=(
                f"1 / 首个 Gold 排名（{first['rank']}）"
                if first
                else "未检索到 Gold"
            ),
            documents=retrieved,
            gold_documents=gold_documents,
            snippets=snippets,
        )
    elif base in {"em", "f1"}:
        comparisons = _answer_comparisons(answer, references)
        evidence.update(
            formula=(
                "系统答案与参考答案归一化后完全匹配记为 1，否则记为 0；多参考答案取最高"
                if base == "em"
                else "系统答案与各参考答案计算 Token Precision 和 Recall 的调和平均，多参考答案取最高"
            ),
            answer_comparison={
                "answer": answer,
                "normalized_answer": qa.normalize_answer(answer),
                "references": comparisons,
            },
        )
    elif base in {
        "citation_precision",
        "citation_recall",
        "uncited_count",
        "uncited_gold_count",
    }:
        uncited = [
            row
            for row in valid_retrieved
            if row["doc_id"] not in {c["doc_id"] for c in valid_citations}
        ]
        uncited_gold = [row for row in uncited if row["is_gold"]]
        cited_ids = {doc["doc_id"] for doc in valid_citations}
        snippet_ids = (
            {doc["doc_id"] for doc in uncited_gold}
            if base == "uncited_gold_count"
            else {doc["doc_id"] for doc in uncited}
            if base == "uncited_count"
            else cited_ids
        )
        evidence.update(
            documents=citations,
            retrieved_documents=retrieved,
            gold_documents=gold_documents,
            citation_excerpts=citation_excerpts,
            snippets=[
                row
                for row in snippets
                if set(row["doc_ids"]) & snippet_ids
            ],
        )
        formulas = {
            "citation_precision": f"实际引用中命中 Gold（{len(cited_gold)}）/ 实际引用（{len(valid_citations)}）",
            "citation_recall": f"实际引用中命中 Gold（{len(cited_gold)}）/ 实际需要的 Gold（{gold_count}）",
            "uncited_count": f"已检索但未被引用的文档（{len(uncited)}）",
            "uncited_gold_count": f"已检索但未被引用的 Gold 文档（{len(uncited_gold)}）",
        }
        evidence["formula"] = formulas[base]
        if base in {"uncited_count", "uncited_gold_count"}:
            evidence["difference_documents"] = uncited_gold if base == "uncited_gold_count" else uncited
    elif base.startswith("graph_"):
        graph = [row for row in snippets if row["is_graph"]]
        graph_gold = [row for row in graph if row["is_gold"]]
        graph_gold_ids = {doc for row in graph for doc in row["gold_doc_ids"]}
        graph_doc_ids = {doc for row in graph for doc in row["doc_ids"]}
        other_gold_ids = {
            doc for row in snippets if not row["is_graph"] for doc in row["gold_doc_ids"]
        }
        exclusive = graph_gold_ids - other_gold_ids
        formulas = {
            "graph_neighbor_gold_snippets": f"图扩展片段中命中 Gold 的片段（{len(graph_gold)}）",
            "graph_neighbor_precision": f"图扩展命中的 Gold 文档（{len(graph_gold_ids)}）/ 图扩展命中的文档（{len(graph_doc_ids)}）",
            "graph_exclusive_gold_share": f"仅由图扩展命中的 Gold（{len(exclusive)}）/ 实际需要的 Gold（{gold_count}）",
        }
        evidence.update(
            formula=formulas[base],
            snippets=snippets,
            gold_documents=gold_documents,
            exclusive_gold_doc_ids=sorted(exclusive),
        )
    elif registry.get_metric(name).kind == registry.KIND_JUDGE:
        judge_detail = (verdict or {}).get("detail") or {}
        formulas = {
            "faithfulness": f"上下文支持的事实陈述（{judge_detail.get('supported', 0)}）/ 全部事实陈述（{judge_detail.get('claim_count', 0)}）",
            "answer_relevancy": f"直接回答问题的句子（{judge_detail.get('relevant', 0)}）/ 有效句子（{max(0, int(judge_detail.get('sentence_count', 0)) - int(judge_detail.get('ignored', 0)))}）",
            "context_relevancy": f"对问题有用的上下文（{judge_detail.get('useful', 0)}）/ 全部上下文段落（{judge_detail.get('passage_count', 0)}）",
            "answer_correctness": f"Judge 判定档位（{judge_detail.get('verdict', '无')}）：正确 1、部分正确 0.5、错误 0",
        }
        evidence.update(
            formula=formulas[base],
            judge_detail=judge_detail,
            answer_comparison={"answer": answer, "references": references},
        )
        if base in {"faithfulness", "context_relevancy"}:
            evidence["snippets"] = snippets

    return evidence


def _document_rows(
    entries: list[dict[str, Any]],
    page_to_doc: dict[str, str],
    documents: dict[str, dict[str, Any]],
    gold: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        page_id = entry.get("sourcePageId")
        if not page_id:
            continue
        doc_id = page_to_doc.get(page_id)
        key = doc_id or f"unmapped:{page_id}"
        if key in seen:
            continue
        seen.add(key)
        doc = documents.get(doc_id or "") or {}
        rows.append(
            {
                "rank": len(rows) + 1,
                "doc_id": doc_id,
                "page_id": page_id,
                "title": doc.get("title") or entry.get("title") or doc_id or page_id,
                "is_gold": doc_id in gold if doc_id else False,
                "mapped": doc_id is not None,
            }
        )
    return rows


def _snippet_rows(
    snippets: list[dict[str, Any]], page_to_doc: dict[str, str], gold: set[str]
) -> list[dict[str, Any]]:
    rows = []
    for index, snippet in enumerate(snippets, 1):
        doc_ids = sorted(
            {
                page_to_doc[window["sourcePageId"]]
                for window in snippet.get("sourceWindows") or []
                if window.get("sourcePageId") in page_to_doc
            }
        )
        page_ids = sorted(
            {
                window["sourcePageId"]
                for window in snippet.get("sourceWindows") or []
                if window.get("sourcePageId")
            }
        )
        gold_ids = sorted(set(doc_ids) & gold)
        reasons = list(snippet.get("retrievalReasons") or [])
        rows.append(
            {
                "rank": index,
                "id": snippet.get("id"),
                "title": snippet.get("title") or "",
                "text": snippet.get("text") or "",
                "retrieval_reasons": reasons,
                "page_ids": page_ids,
                "doc_ids": doc_ids,
                "gold_doc_ids": gold_ids,
                "is_gold": bool(gold_ids),
                "is_graph": "graph-neighbor" in reasons,
            }
        )
    return rows


def _citation_excerpt_rows(
    entries: list[dict[str, Any]],
    page_to_doc: dict[str, str],
    documents: dict[str, dict[str, Any]],
    gold: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        page_id = entry.get("sourcePageId")
        doc_id = page_to_doc.get(page_id)
        excerpts = [
            value.get("text") if isinstance(value, dict) else str(value)
            for value in entry.get("excerpts") or []
        ]
        rows.append(
            {
                "doc_id": doc_id,
                "page_id": page_id,
                "title": (documents.get(doc_id or "") or {}).get("title")
                or entry.get("title")
                or doc_id
                or page_id,
                "is_gold": doc_id in gold if doc_id else False,
                "excerpts": [value for value in excerpts if value],
            }
        )
    return rows


def _answer_comparisons(answer: str, references: list[str]) -> list[dict[str, Any]]:
    answer_tokens = qa.tokenize(answer)
    rows = []
    for reference in references:
        reference_tokens = qa.tokenize(reference)
        overlap = Counter(answer_tokens) & Counter(reference_tokens)
        shared = sum(overlap.values())
        precision = shared / len(answer_tokens) if answer_tokens else 0.0
        recall = shared / len(reference_tokens) if reference_tokens else 0.0
        rows.append(
            {
                "text": reference,
                "normalized": qa.normalize_answer(reference),
                "exact_match": qa.exact_match(answer, reference),
                "answer_token_count": len(answer_tokens),
                "reference_token_count": len(reference_tokens),
                "shared_token_count": shared,
                "shared_tokens": sorted(overlap.elements()),
                "precision": precision,
                "recall": recall,
                "f1": qa.token_f1(answer, reference),
            }
        )
    return rows


def _interpret(
    name: str,
    value: float | None,
    detail: dict[str, Any],
    verdict: dict[str, Any] | None,
    omitted: bool,
    evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    definition = registry.get_metric(name)
    if value is None:
        reason = _missing_reason(name, verdict, omitted)
        status = "unavailable"
    else:
        reason = _score_reason(name, float(value), detail, verdict)
        status = _status(name, float(value), definition.higher_is_better)
    return {
        "name": name,
        "family": definition.family,
        "family_label": FAMILY_LABELS[definition.family],
        "kind": definition.kind,
        "value": value,
        "higher_is_better": definition.higher_is_better,
        "description": definition.description,
        "reason": reason,
        "evidence": evidence,
        "status": status,
    }


def _missing_reason(
    name: str, verdict: dict[str, Any] | None, omitted: bool
) -> str:
    if omitted:
        return "未计算：该数据集缺少这个指标所需的 gold 文档或参考答案标注。"
    if verdict:
        failure = verdict.get("failure_kind")
        if failure:
            return f"Judge 未产出分数：{failure}。该失败不按 0 分计入汇总。"
        return "该样本上指标无定义（例如拒答、无事实陈述或没有可判定上下文），不按 0 分。"
    if registry.get_metric(name).kind == registry.KIND_JUDGE:
        return "没有这项 Judge 判决，可能尚未执行或本次调用未完成。"
    return "当前样本没有保存这项指标值。"


def _score_reason(
    name: str, value: float, detail: dict[str, Any], verdict: dict[str, Any] | None
) -> str:
    base, _, k_text = name.partition("@")
    k = int(k_text) if k_text.isdigit() else None
    gold_count = len(detail.get("gold_doc_ids") or [])
    judge_detail = (verdict or {}).get("detail") or {}

    if base == "recall":
        hits = round(value * gold_count)
        return f"前 {k} 条检索结果命中 {hits}/{gold_count} 篇 gold 文档。"
    if base == "precision":
        return f"前 {k} 条实际返回文档中，命中 gold 的比例为 {_percent(value)}。"
    if base == "retrieval_f1":
        return f"前 {k} 条检索结果的 Precision 与 Recall 调和平均为 {_percent(value)}。"
    if base == "hit":
        return f"前 {k} 条内{'至少命中一篇' if value else '没有命中任何'} gold 文档。"
    if base == "full_coverage":
        return f"前 {k} 条{'已覆盖全部' if value else '未覆盖全部'} gold 文档。"
    if base == "ndcg":
        if value == 1:
            return f"前 {k} 条中的 gold 排序达到理想位置。"
        if value == 0:
            return f"前 {k} 条没有贡献有效的 gold 排名。"
        return f"前 {k} 条的 gold 排序质量为 {_percent(value)}，存在漏召或排名靠后。"
    if base == "mrr":
        return (
            "没有检索到 gold 文档，因此不存在首个命中排名。"
            if value == 0
            else f"首个 gold 文档约位于第 {round(1 / value)} 名。"
        )
    if base == "em":
        return (
            "系统答案归一化后与某个参考答案完全一致。"
            if value == 1
            else "系统答案与参考答案不是整串完全一致；长解释答案可能因此记 0，需结合 F1 和正确性 Judge。"
        )
    if base == "f1":
        return f"系统答案与最接近参考答案的 token 重叠率为 {_percent(value)}；额外解释会降低该值。"
    if base == "citation_precision":
        return "去重后的有效引用中，属于 gold 文档的比例。下方列出本次计算使用的文档和片段。"
    if base == "citation_recall":
        return f"引用覆盖约 {round(value * gold_count)}/{gold_count} 篇 gold 文档。"
    if base == "uncited_count":
        return f"有 {_count(value)} 篇已检索文档未被引用。"
    if base == "uncited_gold_count":
        return f"有 {_count(value)} 篇已检索到的 gold 文档未被引用。"
    if base == "graph_exclusive_gold_share":
        return f"约 {round(value * gold_count)}/{gold_count} 篇 gold 只能靠图扩展获得。"
    if base == "graph_neighbor_precision":
        total = int(
            (((detail.get("multihop") or {}).get("reason_doc_counts") or {}).get(
                "graph-neighbor"
            ) or 0)
        )
        return _ratio_text(value, total, "图扩展命中的文档属于 gold")
    if base == "faithfulness":
        return _judge_ratio(value, judge_detail, "claim_count", "supported", "事实陈述有检索证据支持")
    if base == "answer_relevancy":
        total = int(judge_detail.get("sentence_count") or 0) - int(judge_detail.get("ignored") or 0)
        return _ratio_text(value, total, "实质句子直接回答了问题")
    if base == "context_relevancy":
        return _judge_ratio(value, judge_detail, "passage_count", "useful", "检索片段对回答问题有用")
    if base == "answer_correctness":
        verdict_name = judge_detail.get("verdict")
        labels = {"correct": "事实一致", "partial": "部分正确", "incorrect": "事实不一致"}
        suffix = f"：{judge_detail['reason']}" if judge_detail.get("reason") else "。"
        return f"Judge 判定答案与参考答案{labels.get(verdict_name, _percent(value))}{suffix}"
    if base in COUNT_METRICS:
        labels = {
            "graph_neighbor_gold_snippets": "命中 gold 的图扩展片段",
        }
        return f"本样本有 {_count(value)} 个{labels.get(base, '计数项')}。"
    return f"本样本该指标值为 {value:.4f}。"


def _status(name: str, value: float, higher_is_better: bool) -> str:
    base = name.split("@", 1)[0]
    if base in COUNT_METRICS and base not in {"uncited_count", "uncited_gold_count"}:
        return "neutral"
    if not higher_is_better:
        return "good" if value == 0 else "bad"
    if value >= 0.8:
        return "good"
    if value >= 0.5:
        return "warning"
    return "bad"


def _detail_value(detail: dict[str, Any], section: str, name: str) -> int:
    return int(((detail.get(section) or {}).get(name) or 0))


def _judge_ratio(value: float, detail: dict[str, Any], total_key: str, hit_key: str, label: str) -> str:
    total = int(detail.get(total_key) or 0)
    hits = int(detail.get(hit_key) or round(value * total))
    return f"{hits}/{total} 条{label}。" if total else f"{label}的比例为 {_percent(value)}。"


def _ratio_text(value: float, total: int, label: str) -> str:
    return f"约 {round(value * total)}/{total} 条{label}。" if total else f"{label}的比例为 {_percent(value)}。"


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _count(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.2f}"
