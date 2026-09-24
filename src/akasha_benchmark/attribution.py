"""按指标和链路信号归因。"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from typing import Any

from .metrics import qa, registry

CAUSE_ANSWER_CORRECT = "answer_correct"
CAUSE_ANSWER_INCORRECT = "answer_incorrect"
CAUSE_GENERATION_IGNORED_RETRIEVAL = "generation_ignored_retrieval"
CAUSE_GENERATION_FALLBACK = "generation_fallback"
CAUSE_RETRIEVAL_EVIDENCE_INCOMPLETE = "retrieval_evidence_incomplete"
CAUSE_COMPILED_AWAY = "compiled_away"
CAUSE_CITATION_DROPPED = "citation_dropped"
CAUSE_RETRIEVAL_MISS = "retrieval_miss"
CAUSE_GRAPH_EDGE_MISSING = "graph_edge_missing"
CAUSE_UNKNOWN = "unknown"

EVIDENCE_CHAIN_SUPPORTED_OVERLAP = 0.8
EVIDENCE_CHAIN_PARTIAL_OVERLAP = 0.35
REFERENCE_STOPWORDS = {"a", "an", "and", "in", "of", "on", "the", "to"}


def _normalized_tokens(text: str) -> list[str]:
    return qa.tokenize(unicodedata.normalize("NFKC", text or ""))


def _contains_tokens(haystack: list[str], needle: list[str]) -> bool:
    width = len(needle)
    return bool(
        width
        and any(
            haystack[index : index + width] == needle
            for index in range(len(haystack) - width + 1)
        )
    )


def _token_recall(reference: str, context_tokens: list[str]) -> float:
    reference_tokens = set(_normalized_tokens(reference))
    if not reference_tokens:
        return 0.0
    return len(reference_tokens & set(context_tokens)) / len(reference_tokens)


def _response_evidence(response: dict[str, Any] | None) -> list[dict[str, str]]:
    if not isinstance(response, dict):
        return []
    rows: list[dict[str, str]] = []
    for snippet in response.get("snippets") or []:
        if not isinstance(snippet, dict):
            continue
        rows.append({
            "source_type": "context",
            "title": str(snippet.get("title") or ""),
            "text": str(snippet.get("text") or ""),
        })
        for window in snippet.get("sourceWindows") or []:
            if isinstance(window, dict):
                rows.append({
                    "source_type": "source_window",
                    "title": str(window.get("title") or snippet.get("title") or ""),
                    "text": str(window.get("text") or ""),
                })
    return [row for row in rows if row["title"] or row["text"]]


def _matching_evidence(
    evidence: list[dict[str, str]], support_text: str, answer: str
) -> list[dict[str, Any]]:
    support_tokens = set(_normalized_tokens(support_text))
    answer_tokens = _normalized_tokens(answer)
    best_by_text: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in evidence:
        ordered_tokens = _normalized_tokens(f"{row['title']} {row['text']}")
        tokens = set(ordered_tokens)
        if not tokens:
            continue
        support_score = len(tokens & support_tokens) / len(support_tokens) if support_tokens else 0.0
        answer_hit = _contains_tokens(ordered_tokens, answer_tokens)
        if support_score >= EVIDENCE_CHAIN_PARTIAL_OVERLAP or answer_hit:
            scored = {
                **row,
                "support_overlap": round(support_score, 6),
                "answer_match": answer_hit,
            }
            key = (
                row["source_type"],
                " ".join(_normalized_tokens(row["title"])),
                " ".join(_normalized_tokens(row["text"])),
            )
            previous = best_by_text.get(key)
            if previous is None or (
                scored["support_overlap"], scored["answer_match"]
            ) > (previous["support_overlap"], previous["answer_match"]):
                best_by_text[key] = scored
    return sorted(
        best_by_text.values(),
        key=lambda row: (row["support_overlap"], row["answer_match"]),
        reverse=True,
    )[:3]


def analyze_evidence_chain(
    sample: dict[str, Any], response: dict[str, Any] | None
) -> dict[str, Any]:
    """检查 MuSiQue 各推理步骤的证据是否进入回答上下文。"""
    detail = sample.get("detail") or {}
    metadata = detail.get("metadata") or {}
    decomposition = metadata.get("question_decomposition") or []
    if sample.get("dataset") != "musique" or not decomposition:
        return {
            "status": "unavailable",
            "reason": "dataset_has_no_decomposition_supports",
            "steps": [],
        }

    response_evidence = _response_evidence(response)
    context = "\n".join(
        value
        for row in response_evidence
        for value in (row["title"], row["text"])
        if value
    )
    context_tokens = _normalized_tokens(context)
    steps: list[dict[str, Any]] = []
    for position, step in enumerate(decomposition, 1):
        answer = str(step.get("answer") or "")
        support_title = str(step.get("support_title") or "")
        support_text = str(step.get("support_text") or "")
        answer_present = _contains_tokens(context_tokens, _normalized_tokens(answer))
        title_present = _contains_tokens(context_tokens, _normalized_tokens(support_title))
        support_overlap = _token_recall(support_text, context_tokens)
        matched_evidence = _matching_evidence(response_evidence, support_text, answer)

        if answer_present and support_overlap >= EVIDENCE_CHAIN_SUPPORTED_OVERLAP:
            status = "supported"
        elif (
            answer_present
            or title_present
            or support_overlap >= EVIDENCE_CHAIN_PARTIAL_OVERLAP
        ):
            status = "partial"
        else:
            status = "missing"

        steps.append(
            {
                "position": position,
                "question": step.get("question"),
                "answer": answer,
                "support_doc_id": step.get("support_doc_id"),
                "support_title": support_title,
                "status": status,
                "answer_present": answer_present,
                "support_title_present": title_present,
                "support_token_recall": round(support_overlap, 6),
                "claim_retrieved": bool(matched_evidence),
                "retrieved_evidence": matched_evidence,
            }
        )

    counts = Counter(step["status"] for step in steps)
    if counts["missing"]:
        status = "incomplete"
    elif counts["partial"]:
        status = "partial"
    else:
        status = "complete"
    answer_mode = sample.get("answer_mode")
    return {
        "status": status,
        "step_count": len(steps),
        "supported_step_count": counts["supported"],
        "partial_step_count": counts["partial"],
        "missing_step_count": counts["missing"],
        "model_false_negative_candidate": answer_mode == "general" and status == "complete",
        "steps": steps,
        "thresholds": {
            "supported_token_recall": EVIDENCE_CHAIN_SUPPORTED_OVERLAP,
            "partial_token_recall": EVIDENCE_CHAIN_PARTIAL_OVERLAP,
        },
    }


def _contains_reference(answer: str, references: list[str]) -> bool:
    """答案是否覆盖某个参考答案至少 80% 的有效 token。"""
    answer_tokens = qa.tokenize(answer)
    for reference in references:
        reference_tokens = qa.tokenize(reference)
        if not reference_tokens or (
            len(reference_tokens) == 1 and reference_tokens[0] in REFERENCE_STOPWORDS
        ):
            continue
        overlap = len(set(answer_tokens) & set(reference_tokens)) / len(set(reference_tokens))
        if overlap >= 0.8:
            return True
    return False


def _at_max_k(metrics: dict[str, float], prefix: str) -> float | None:
    """取最大 k 的那一项，没有则返回 None 而不是 0。"""
    keys = sorted(
        (k for k in metrics if k.startswith(prefix)),
        key=lambda k: int(k.split("@", 1)[1]),
        reverse=True,
    )
    return metrics[keys[0]] if keys else None


def classify(
    sample: dict[str, Any],
    lineage: list[dict[str, Any]] | None,
    evidence_chain: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """规则归因，返回 ``{root_cause, evidence}``。

    ``lineage`` 是每篇 gold 的 diff 结果。为 None 时 ``compiled_away`` 判不了，
    退到 ``retrieval_miss`` 并在 evidence 里注明。
    """
    detail = sample.get("detail") or {}
    metrics: dict[str, float] = {}
    for section in ("qa", "retrieval", "attribution", "multihop"):
        metrics.update(
            {
                name: float(value)
                for name, value in (detail.get(section) or {}).items()
                if isinstance(value, (int, float, bool))
            }
        )
    metrics.update(sample.get("metrics") or {})
    answer_mode = sample.get("answer_mode")
    hit = _at_max_k(metrics, "hit@")
    recall = _at_max_k(metrics, "recall@")
    coverage = _at_max_k(metrics, "full_coverage@")
    uncited_gold = float(metrics.get("uncited_gold_count", 0.0) or 0.0)
    graph_exclusive = float(metrics.get("graph_exclusive_gold_share", 0.0) or 0.0) > 0
    has_retrieval = hit is not None and hit > 0
    reference_contained = _contains_reference(
        str(sample.get("answer") or ""),
        [str(value) for value in detail.get("reference_answers") or []],
    )

    lost_terms: list[str] = []
    for entry in lineage or []:
        lost_terms.extend(entry.get("question_terms_lost") or [])
    lost_terms = sorted(set(lost_terms))

    answer_correct = float(metrics.get("em", 0.0)) >= 1.0 or reference_contained
    evidence: dict[str, Any] = {
        "answer_mode": answer_mode,
        "has_retrieval": has_retrieval,
        "hit": hit,
        "recall": recall,
        "full_coverage": coverage,
        "uncited_gold_count": uncited_gold,
        "question_terms_lost": lost_terms,
        "graph_exclusive_gold_share": metrics.get("graph_exclusive_gold_share"),
        "reference_answer_contained": reference_contained,
        "answer_correct": answer_correct,
        "gold_count": len(detail.get("gold_doc_ids") or []),
        "lineage_available": lineage is not None,
    }
    if evidence_chain is not None:
        evidence["evidence_chain"] = evidence_chain

    chain = evidence.get("evidence_chain") or {}
    if answer_mode == "general":
        if answer_correct:
            cause = CAUSE_GENERATION_FALLBACK
        elif not has_retrieval:
            cause = CAUSE_GENERATION_FALLBACK
        elif chain.get("status") in {"incomplete", "partial"}:
            cause = CAUSE_RETRIEVAL_EVIDENCE_INCOMPLETE
        elif recall is not None and recall >= 1.0:
            cause = CAUSE_GENERATION_IGNORED_RETRIEVAL
        elif recall is not None and recall < 1.0:
            cause = CAUSE_RETRIEVAL_EVIDENCE_INCOMPLETE
        else:
            cause = CAUSE_UNKNOWN
    elif answer_mode == "knowledge" and answer_correct:
        cause = CAUSE_ANSWER_CORRECT
    elif uncited_gold > 0:
        cause = CAUSE_CITATION_DROPPED
    elif hit is not None and hit == 0 and lost_terms:
        cause = CAUSE_COMPILED_AWAY
    elif hit is not None and hit == 0:
        cause = CAUSE_RETRIEVAL_MISS
    elif coverage is not None and coverage < 1.0 and not graph_exclusive:
        cause = CAUSE_GRAPH_EDGE_MISSING
    elif answer_mode == "knowledge":
        cause = CAUSE_ANSWER_INCORRECT
    else:
        cause = CAUSE_UNKNOWN

    return {"root_cause": cause, "evidence": evidence}


SYSTEM_PROMPT = """你是一名 RAG 评测分析师。你分析的是一次完整评测，
不是某个样本。输入包含本次实际产出的各项指标、数据集切片、回答模式分布、
HTTP 失败、遗漏指标，以及最多 10 条 general 回答案例。案例直接从评测数据抽取，
不包含任何规则归因结论。

请写一份可直接交付的中文分析报告，要求：
1. 先总结整体表现，再逐项分析输入中实际存在的每个指标；没有的指标不要臆测。
2. 明确区分“指标直接说明的事实”和“潜在原因”。潜在原因必须使用可能、疑似、
   建议验证等审慎措辞，不能把相关性写成已证实因果。
3. 对 overall、knowledge_only、judge 等不同 scope 分开解读；注意样本数和省略指标。
4. EM/F1 受解释性长答案影响，只描述其表现，不能仅凭低分断言答案错误或归因根因。
5. Judge 指标可以辅助观察，但要说明它们来自模型判定，存在模型与提示词偏差。
6. 给出按优先级排序、可验证的下一步建议，并说明报告局限。
7. general 可能仍保留 retrievedSources 和 graph-neighbor 证据。分析 general 案例时必须
   结合检索、引用和图指标，不得把 general 自动解释为没有召回。不要逐条复述或外推整体。
8. 不要声称看过未提供的原文、答案或日志。

只输出 JSON，不要用代码围栏：
{"report": "使用简短标题和分段组织的完整纯文本分析报告"}"""


def build_report_prompt(
    eval_run: dict[str, Any],
    metric_summaries: list[dict[str, Any]],
    dataset_summaries: list[dict[str, Any]],
    general_samples: list[dict[str, Any]],
) -> tuple[str, str]:
    """拼整轮报告提示词：评测汇总加 general 案例，不提供规则归因数据。"""
    definitions: dict[str, dict[str, Any]] = {}
    for row in metric_summaries:
        name = str(row["metric"])
        try:
            definition = registry.get_metric(name)
        except KeyError:
            continue
        definitions[name] = {
            "family": definition.family,
            "kind": definition.kind,
            "higher_is_better": definition.higher_is_better,
            "description": definition.description,
        }

    payload = {
        "evaluation": {
            "id": eval_run.get("id"),
            "name": eval_run.get("name"),
        },
        "metric_definitions": definitions,
        "metric_summaries": metric_summaries,
        "dataset_summaries": dataset_summaries,
        "general_answer_examples": general_samples[:10],
    }
    return SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
