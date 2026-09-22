"""归因判据：从指标 + 链路推出根因。

两段式：规则不需要模型配置，逐样本分类；模型可选地分析整轮汇总指标并生成报告。

| 根因                    | 判据                                          |
| ----------------------- | --------------------------------------------- |
| not_a_failure           | EM 命中或系统答案完整包含参考答案               |
| generation_ignored_retrieval | general 且仍有检索结果                    |
| generation_fallback     | general/no_match 且没有检索结果                 |
| citation_dropped        | truncated_gold > 0（召回到了但引用被截断）      |
| compiled_away           | hit@k == 0 且问题实词在编译时丢失               |
| retrieval_miss          | hit@k == 0                                    |
| graph_edge_missing      | gold 不全且图扩展没贡献独有 gold                |
| unknown                 | 现有非 Judge 信号不足以定位                    |

顺序即优先级。``not_a_failure`` 必须第一：答案明确正确时不再归为异常。
其余 general / no_match 回答再归为 ``generation_fallback``。
"""

from __future__ import annotations

import json
from typing import Any

from .metrics import qa, registry

CAUSE_NOT_A_FAILURE = "not_a_failure"
CAUSE_GENERATION_IGNORED_RETRIEVAL = "generation_ignored_retrieval"
CAUSE_GENERATION_FALLBACK = "generation_fallback"
CAUSE_COMPILED_AWAY = "compiled_away"
CAUSE_CITATION_DROPPED = "citation_dropped"
CAUSE_RETRIEVAL_MISS = "retrieval_miss"
CAUSE_GRAPH_EDGE_MISSING = "graph_edge_missing"
CAUSE_UNKNOWN = "unknown"


def _contains_reference(answer: str, references: list[str]) -> bool:
    """参考答案的归一化 token 是否连续出现在系统答案中。"""
    answer_tokens = qa.tokenize(answer)
    for reference in references:
        reference_tokens = qa.tokenize(reference)
        width = len(reference_tokens)
        if width and any(
            answer_tokens[index : index + width] == reference_tokens
            for index in range(len(answer_tokens) - width + 1)
        ):
            return True
    return False


def _answered_correctly(sample: dict[str, Any]) -> bool:
    """答案是否算对：只使用本地确定性信号，不依赖 Judge 指标。

    EM 命中或参考答案完整包含即成立。低 EM/F1 不作为失败证据：它们分不开
    「答对了被散文稀释」与「答错了但词有重叠」。
    """
    metrics: dict[str, float] = sample.get("metrics") or {}
    em = metrics.get("em")
    if em is not None and em >= 1.0:
        return True
    detail = sample.get("detail") or {}
    return _contains_reference(
        str(sample.get("answer") or ""),
        [str(value) for value in detail.get("reference_answers") or []],
    )


def _at_max_k(metrics: dict[str, float], prefix: str) -> float | None:
    """取最大 k 的那一项，没有则返回 None 而不是 0。"""
    keys = sorted(
        (k for k in metrics if k.startswith(prefix)),
        key=lambda k: int(k.split("@", 1)[1]),
        reverse=True,
    )
    return metrics[keys[0]] if keys else None


def classify(
    sample: dict[str, Any], lineage: list[dict[str, Any]] | None
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
    coverage = _at_max_k(metrics, "full_coverage@")
    truncated = float(metrics.get("truncated_gold", 0.0) or 0.0)
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

    evidence: dict[str, Any] = {
        "answer_mode": answer_mode,
        "has_retrieval": has_retrieval,
        "hit": hit,
        "full_coverage": coverage,
        "truncated_gold": truncated,
        "question_terms_lost": lost_terms,
        "graph_exclusive_gold_share": metrics.get("graph_exclusive_gold_share"),
        "reference_answer_contained": reference_contained,
        "gold_count": len(detail.get("gold_doc_ids") or []),
        "lineage_available": lineage is not None,
    }

    # 顺序即优先级，见模块开头。答案明确正确时优先排除失败归因。
    if _answered_correctly(sample):
        cause = CAUSE_NOT_A_FAILURE
    elif answer_mode == "general" and has_retrieval:
        cause = CAUSE_GENERATION_IGNORED_RETRIEVAL
    elif answer_mode and answer_mode != "knowledge":
        cause = CAUSE_GENERATION_FALLBACK
    elif truncated > 0:
        cause = CAUSE_CITATION_DROPPED
    elif hit is not None and hit == 0 and lost_terms:
        cause = CAUSE_COMPILED_AWAY
    elif hit is not None and hit == 0:
        cause = CAUSE_RETRIEVAL_MISS
    elif coverage is not None and coverage < 1.0 and not graph_exclusive:
        cause = CAUSE_GRAPH_EDGE_MISSING
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
