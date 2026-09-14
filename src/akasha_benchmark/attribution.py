"""归因判据：从指标 + 链路推出根因。

**两段式，规则在前。** 规则不需要任何模型配置就能出分类，所以归因层在没配
分析模型时也能用；模型在它之上补一段因果叙述。

分开不是为了稳妥，而是两者答的问题不同：规则给分类（哪一段断了），
模型给叙述（为什么断在那里）。把后者当分类用会得到一个听起来有道理、
但与指标对不上的结论 —— 而那种错误没有任何地方会报出来。

| 根因                    | 判据                                          |
| ----------------------- | --------------------------------------------- |
| generation_fallback     | answer_mode != knowledge（retrievedSources 空）|
| compiled_away           | 问题实词落在编译丢掉的词里                     |
| citation_dropped        | truncated_gold > 0（召回到了但引用被截断）      |
| retrieval_miss          | hit@k == 0                                    |
| graph_edge_missing      | gold 不全且图扩展没贡献独有 gold                |
| gold_annotation_suspect | gold 全召回、引用完整，答案仍判错               |

顺序即优先级：``generation_fallback`` 必须最先判，否则它那 0 分会被解释成
检索失败。
"""

from __future__ import annotations

from typing import Any

CAUSE_GENERATION_FALLBACK = "generation_fallback"
CAUSE_COMPILED_AWAY = "compiled_away"
CAUSE_CITATION_DROPPED = "citation_dropped"
CAUSE_RETRIEVAL_MISS = "retrieval_miss"
CAUSE_GRAPH_EDGE_MISSING = "graph_edge_missing"
CAUSE_GOLD_SUSPECT = "gold_annotation_suspect"
CAUSE_UNKNOWN = "unknown"

# 每个根因该怎么处置。「这条能不能靠调参救」是归因结论里最有用的一句。
REMEDIES = {
    CAUSE_GENERATION_FALLBACK: "生成端拒答，检索指标按定义为 0。不是检索问题，"
    "先看 answerMode 分布而不是 recall。",
    CAUSE_COMPILED_AWAY: "编译产物里缺少问题中的实词，三条召回路径同时断。"
    "调参救不了 —— 词已经不在被索引的文本里。要改编译提示词或换 compiler。",
    CAUSE_CITATION_DROPPED: "召回到了 gold 但引用被截断。检索没问题，"
    "问题在引用预算或答案长度限制。",
    CAUSE_RETRIEVAL_MISS: "原文里有问题的实词、编译产物也留着，但没召回到。"
    "这是排序或阈值问题，属于可调范围。",
    CAUSE_GRAPH_EDGE_MISSING: "多跳缺跳：图扩展没有贡献任何独有 gold。"
    "跨文档实体没连起来。",
    CAUSE_GOLD_SUSPECT: "gold 全召回、引用完整，答案仍判错。"
    "先怀疑参考答案或评分口径，而不是系统。",
    CAUSE_UNKNOWN: "现有信号不足以定位。看链路视图里的原文/编译 diff。",
}


def _at_max_k(metrics: dict[str, float], prefix: str) -> float | None:
    """取最大 k 的那一项。没有则返回 None（不是 0）。"""
    keys = sorted(
        (k for k in metrics if k.startswith(prefix)),
        key=lambda k: int(k.split("@", 1)[1]),
        reverse=True,
    )
    return metrics[keys[0]] if keys else None


def classify(
    sample: dict[str, Any], lineage: list[dict[str, Any]] | None
) -> dict[str, Any]:
    """规则归因。返回 ``{root_cause, evidence}``。

    ``lineage`` 是每篇 gold 的 diff 结果，没配只读库时为 None —— 那时
    ``compiled_away`` 判不了，会退到 ``retrieval_miss`` 并在 evidence 里注明。
    """
    metrics: dict[str, float] = sample.get("metrics") or {}
    detail = sample.get("detail") or {}
    answer_mode = sample.get("answer_mode")
    hit = _at_max_k(metrics, "hit@")
    coverage = _at_max_k(metrics, "full_coverage@")
    truncated = float(metrics.get("truncated_gold", 0.0) or 0.0)
    graph_exclusive = metrics.get("graph_exclusive_gold_count")

    lost_terms: list[str] = []
    for entry in lineage or []:
        lost_terms.extend(entry.get("question_terms_lost") or [])
    lost_terms = sorted(set(lost_terms))

    evidence: dict[str, Any] = {
        "answer_mode": answer_mode,
        "hit": hit,
        "full_coverage": coverage,
        "truncated_gold": truncated,
        "question_terms_lost": lost_terms,
        "graph_exclusive_gold_count": graph_exclusive,
        "f1": metrics.get("f1"),
        "gold_count": len(detail.get("gold_doc_ids") or []),
        "lineage_available": lineage is not None,
    }

    # 顺序即优先级，见模块开头。
    if answer_mode and answer_mode != "knowledge":
        cause = CAUSE_GENERATION_FALLBACK
    elif lost_terms:
        cause = CAUSE_COMPILED_AWAY
    elif truncated > 0:
        cause = CAUSE_CITATION_DROPPED
    elif hit is not None and hit == 0:
        cause = CAUSE_RETRIEVAL_MISS
    elif coverage is not None and coverage < 1.0 and not graph_exclusive:
        cause = CAUSE_GRAPH_EDGE_MISSING
    else:
        # 检索与引用都没问题，答案却不对：先怀疑标注与评分口径。
        # 用 F1 而不是 EM —— EM 对长答案要求整串相等，误判率太高。
        f1 = metrics.get("f1")
        cause = (
            CAUSE_GOLD_SUSPECT
            if coverage == 1.0 and f1 is not None and f1 < 0.3
            else CAUSE_UNKNOWN
        )

    return {"root_cause": cause, "evidence": evidence, "remedy": REMEDIES[cause]}


SYSTEM_PROMPT = """你在分析一个检索增强问答系统的失败样本。

这套系统的向量与词法召回跑在**编译产物**上，不是原始文档：编译器会重写原文，
重写时可能删掉原文里的修饰语与专有名词。被删掉的词不在被索引的文本里，
所以查询命中它们时三条召回路径会同时断，且调参救不回来。

`no_match` 与 `general` 两种回答无条件返回空的 retrievedSources，它们的检索
得分按定义为 0 —— 那是生成端拒答，不是检索失败。

已经有一个规则归因给出了分类。你的任务是**解释因果**，不是重新分类：
如果你认为分类错了，在 disagreement 里说明理由，不要直接改 root_cause。

只输出 JSON：
{"narrative": "两三句话说明这条为什么失败，引用给你的具体证据",
 "contributing_factors": ["..."],
 "disagreement": null 或 "为什么规则分类可能不对",
 "confidence": 0.0 到 1.0}"""


def build_prompt(
    sample: dict[str, Any], ruling: dict[str, Any], lineage: list[dict[str, Any]] | None
) -> tuple[str, str]:
    """拼归因提示词。链路证据裁剪到可读长度。"""
    detail = sample.get("detail") or {}
    lines = [
        f"问题：{detail.get('question')}",
        f"参考答案：{detail.get('reference_answers')}",
        f"系统答案：{(sample.get('answer') or '')[:1200]}",
        f"answerMode：{sample.get('answer_mode')}",
        "",
        f"规则归因：{ruling['root_cause']}",
        f"证据：{ruling['evidence']}",
    ]
    if lineage:
        for entry in lineage[:3]:
            diff = entry.get("diff") or {}
            lines += [
                "",
                f"gold 文档 {entry.get('doc_id')}（page {entry.get('page_id')}）：",
                f"  编译扩写比：{diff.get('expansion_ratio')}",
                f"  编译丢掉的实词：{(diff.get('dropped') or [])[:30]}",
                f"  问题实词里丢掉的：{entry.get('question_terms_lost')}",
                f"  原文片段：{(entry.get('source_text') or '')[:600]}",
                f"  编译片段：{(entry.get('compiled_text') or '')[:600]}",
            ]
    else:
        lines += ["", "（血缘链路不可用：未配置只读数据库，拿不到原文/编译 diff。）"]
    return SYSTEM_PROMPT, "\n".join(lines)
