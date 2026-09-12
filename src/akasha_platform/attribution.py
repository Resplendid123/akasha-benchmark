"""badcase 归因：从指标 + 完整链路推出根因。

**两段式，规则在前。** 规则归因不需要任何模型配置就能出结果，所以「自动根据
指标以及链路分析归因」在没配分析模型时也成立；模型归因在它之上补一段因果叙述。

分开的理由不是稳妥，是两者答的问题不同：规则给分类（哪一段断了），
模型给叙述（为什么断在那里）。把后者当分类用会得到一个听起来很有道理、
但与指标对不上的结论 —— 而那种错误没有任何地方会报出来。

规则判据全部来自已有信号，没有一条是新造的：

| 根因                 | 判据                                                    |
| -------------------- | ------------------------------------------------------- |
| generation_fallback  | ``answer_mode != knowledge``（retrievedSources 被清空）  |
| compiled_away        | ``question_terms_lost`` 非空（编译产物里没有问题的实词） |
| citation_dropped     | ``truncated_gold > 0``（召回到了但引用被截断）            |
| retrieval_miss       | ``hit@k == 0`` 且原文里有问题的实词                      |
| graph_edge_missing   | 多跳、gold 不全，且图扩展没贡献独有 gold                  |
| gold_annotation_suspect | gold 全召回、引用完整，但答案仍判错                    |

顺序即优先级：``generation_fallback`` 必须最先判，否则它那 0 分会被解释成
检索失败 —— run001 上四条 ``recall@5 < 1.0`` 里三条正是这种。
"""

from __future__ import annotations

import sqlite3
from typing import Any

from akasha_benchmark.judge.client import JudgeClient, JudgeConfigError, parse_json_object
from akasha_benchmark.judge.run import resolve_provider
from akasha_benchmark.store import identity, repo

PROMPT_VERSION = "attribution-1"

CAUSE_GENERATION_FALLBACK = "generation_fallback"
CAUSE_COMPILED_AWAY = "compiled_away"
CAUSE_CITATION_DROPPED = "citation_dropped"
CAUSE_RETRIEVAL_MISS = "retrieval_miss"
CAUSE_GRAPH_EDGE_MISSING = "graph_edge_missing"
CAUSE_GOLD_SUSPECT = "gold_annotation_suspect"
CAUSE_UNKNOWN = "unknown"

# 每个根因该怎么处置。写在这里而不是前端，因为它属于判据的一部分 ——
# 「这条能不能靠调参救」是归因结论里最有用的一句。
REMEDIES = {
    CAUSE_GENERATION_FALLBACK: "生成端拒答，检索指标按定义为 0。不是检索问题，"
    "先看 answerMode 分布而不是 recall。",
    CAUSE_COMPILED_AWAY: "编译产物里缺少问题中的实词，三条召回路径同时断。"
    "**调参救不了** —— 词已经不在被索引的文本里。要改编译提示词或换 compiler。",
    CAUSE_CITATION_DROPPED: "召回到了 gold 但引用被截断。检索没问题，"
    "问题在引用预算或答案长度限制。",
    CAUSE_RETRIEVAL_MISS: "原文里有问题的实词、编译产物也留着，但没召回到。"
    "这是排序或阈值问题，属于可调范围。",
    CAUSE_GRAPH_EDGE_MISSING: "多跳缺跳：图扩展没有贡献任何独有 gold。"
    "图边极稀疏，跨文档实体没连起来。",
    CAUSE_GOLD_SUSPECT: "gold 全召回、引用完整，答案仍判错。"
    "先怀疑参考答案或评分口径，而不是系统。",
    CAUSE_UNKNOWN: "现有信号不足以定位。看链路视图里的原文/编译 diff。",
}


def _hit(metrics: dict[str, float]) -> float | None:
    """取任意一个 ``hit@k``，优先大 k。没有则返回 None（不是 0）。"""
    keys = sorted(
        (k for k in metrics if k.startswith("hit@")),
        key=lambda k: int(k.split("@", 1)[1]),
        reverse=True,
    )
    return metrics[keys[0]] if keys else None


def _coverage(metrics: dict[str, float]) -> float | None:
    keys = sorted(
        (k for k in metrics if k.startswith("full_coverage@")),
        key=lambda k: int(k.split("@", 1)[1]),
        reverse=True,
    )
    return metrics[keys[0]] if keys else None


def classify(sample: dict[str, Any], lineage: dict[str, Any] | None) -> dict[str, Any]:
    """规则归因。返回 ``{root_cause, labels, evidence}``。

    ``lineage`` 是 ``/api/layers/eval/{id}/samples/{sid}/lineage`` 的形状，
    没配只读库时为 None —— 那时 ``compiled_away`` 判不了，会退到
    ``retrieval_miss``，并在 evidence 里注明链路不可用。
    """
    metrics: dict[str, float] = sample.get("metrics") or {}
    detail = sample.get("detail") or {}
    answer_mode = sample.get("answer_mode")
    hit = _hit(metrics)
    coverage = _coverage(metrics)
    truncated = float(metrics.get("truncated_gold", 0.0) or 0.0)
    graph_exclusive = metrics.get("graph_exclusive_gold_count")

    # 编译丢词：把每篇 gold 的 question_terms_lost 汇总起来。
    lost_terms: list[str] = []
    lineage_available = lineage is not None
    if lineage:
        for entry in lineage.get("gold") or []:
            lost_terms.extend((entry.get("diff") or {}).get("question_terms_lost") or [])
    lost_terms = sorted(set(lost_terms))

    evidence: dict[str, Any] = {
        "answer_mode": answer_mode,
        "hit": hit,
        "full_coverage": coverage,
        "gold_count": sample.get("gold_count"),
        "retrieved_count": sample.get("retrieved_count"),
        "citation_count": sample.get("citation_count"),
        "truncated_gold": truncated,
        "question_terms_lost": lost_terms,
        "graph_exclusive_gold_count": graph_exclusive,
        "f1": metrics.get("f1"),
        "lineage_available": lineage_available,
        "reason_counts": (detail.get("multihop") or {}).get("reason_counts"),
    }

    labels: list[str] = []

    # 顺序即优先级。第一条必须最先判 —— 见模块开头的说明。
    if answer_mode and answer_mode != "knowledge":
        labels.append(answer_mode)
        return {"root_cause": CAUSE_GENERATION_FALLBACK, "labels": labels, "evidence": evidence}

    if lost_terms:
        labels.append(f"lost:{len(lost_terms)}")
        return {"root_cause": CAUSE_COMPILED_AWAY, "labels": labels, "evidence": evidence}

    if truncated > 0:
        labels.append(f"truncated:{int(truncated)}")
        return {"root_cause": CAUSE_CITATION_DROPPED, "labels": labels, "evidence": evidence}

    if hit is not None and hit == 0:
        if not lineage_available:
            labels.append("lineage-unavailable")
        return {"root_cause": CAUSE_RETRIEVAL_MISS, "labels": labels, "evidence": evidence}

    # 命中了但没凑齐全部 gold，且图扩展没有独有贡献 —— 多跳缺跳。
    if coverage is not None and coverage < 1.0 and not graph_exclusive:
        labels.append("multihop")
        return {"root_cause": CAUSE_GRAPH_EDGE_MISSING, "labels": labels, "evidence": evidence}

    # 检索与引用都没问题，答案却不对：先怀疑标注与评分口径。
    # 规则以 F1 判断答案词面重叠，避免 EM 对长答案的整串匹配限制。
    f1 = metrics.get("f1")
    if coverage == 1.0 and f1 is not None and f1 < 0.3:
        labels.append("low-f1")
        return {"root_cause": CAUSE_GOLD_SUSPECT, "labels": labels, "evidence": evidence}

    return {"root_cause": CAUSE_UNKNOWN, "labels": labels, "evidence": evidence}


SYSTEM_PROMPT = """你在分析一个检索增强问答系统的失败样本。

这套系统的向量与词法召回跑在**编译产物**上，不是原始文档：编译器会重写原文
（实测中位扩写 2.19 倍），重写时可能删掉原文里的修饰语与专有名词。被删掉的词
不在被索引的文本里，所以查询命中它们时三条召回路径会同时断，且调参救不回来。

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
    sample: dict[str, Any], ruling: dict[str, Any], lineage: dict[str, Any] | None
) -> tuple[str, str]:
    """拼归因提示词。链路证据裁剪到可读长度。"""
    lines = [
        f"问题：{sample.get('question')}",
        f"参考答案：{sample.get('reference_answers')}",
        f"系统答案：{(sample.get('answer') or '')[:1200]}",
        f"answerMode：{sample.get('answer_mode')}",
        "",
        f"规则归因：{ruling['root_cause']}",
        f"证据：{ruling['evidence']}",
    ]
    if lineage:
        for entry in (lineage.get("gold") or [])[:3]:
            diff = entry.get("diff") or {}
            lines += [
                "",
                f"gold 文档 {entry.get('doc_id')}（page {entry.get('page_id')}）：",
                f"  编译扩写比：{(diff.get('diff') or {}).get('expansion_ratio')}",
                f"  编译丢掉的实词（前 30）：{((diff.get('diff') or {}).get('dropped') or [])[:30]}",
                f"  问题实词里丢掉的：{diff.get('question_terms_lost')}",
                f"  原文片段：{((diff.get('source') or {}).get('text') or '')[:600]}",
                f"  编译片段：{((diff.get('compiled') or {}).get('text') or '')[:600]}",
            ]
    else:
        lines += ["", "（血缘链路不可用：未配置只读数据库，所以拿不到原文/编译 diff。）"]
    return SYSTEM_PROMPT, "\n".join(lines)


def analyze(
    connection: sqlite3.Connection,
    eval_layer_id: int,
    sample: dict[str, Any],
    lineage: dict[str, Any] | None,
    *,
    provider_label: str | None = None,
    use_model: bool = True,
) -> dict[str, Any]:
    """归因一条样本并写库。返回写进去的那条记录。

    ``use_model=False`` 或分析模型没配好时只写规则结论 —— 那仍然是一条有效的
    归因，不是失败。这一点决定了归因层在零配置下也能用。
    """
    ruling = classify(sample, lineage)
    narrative: str | None = None
    provider_hash = ""
    prompt_version = ""
    rule_based = True
    model_error: str | None = None

    if use_model:
        try:
            provider = resolve_provider(connection, provider_label, role="analysis")
        except JudgeConfigError as exc:
            model_error = str(exc)
        else:
            provider_hash = identity.judge_hash(
                base_url=provider.base_url, model=provider.model, params=None
            )
            system, user = build_prompt(sample, ruling, lineage)
            with JudgeClient(provider) as client:
                reply = client.complete(system, user)
            if reply.failure_kind:
                model_error = f"{reply.failure_kind}: {(reply.raw or '')[:200]}"
            else:
                try:
                    payload = parse_json_object(reply.content or "")
                except ValueError as exc:
                    # 模型没按 schema 输出。规则结论照样写 —— 丢掉它等于
                    # 因为叙述失败而放弃了分类。
                    model_error = f"parse_error: {exc}"
                else:
                    narrative = str(payload.get("narrative") or "").strip() or None
                    ruling["evidence"]["model"] = {
                        "contributing_factors": payload.get("contributing_factors"),
                        "disagreement": payload.get("disagreement"),
                        "confidence": payload.get("confidence"),
                    }
                    prompt_version = PROMPT_VERSION
                    rule_based = False

    if model_error:
        ruling["evidence"]["model_error"] = model_error

    repo.record_badcase_analysis(
        connection,
        eval_layer_id,
        sample_id=sample["sample_id"],
        root_cause=ruling["root_cause"],
        labels=ruling["labels"],
        evidence=ruling["evidence"],
        narrative=narrative,
        rule_based=rule_based,
        provider_hash=provider_hash,
        prompt_version=prompt_version,
    )
    # 同时进 annotation：judge-human 一致率靠 author_kind 分组，
    # 归因结论不进那张表的话就没法与人工判断比对。
    repo.add_annotation(
        connection,
        level="sample",
        target_id=sample["sample_id"],
        author_kind="model",
        author=f"attribution/{'rules' if rule_based else provider_hash}",
        labels=[ruling["root_cause"], *ruling["labels"]],
        note=narrative,
        source="model",
        confidence=(ruling["evidence"].get("model") or {}).get("confidence"),
    )
    connection.commit()

    return {
        **ruling,
        "sample_id": sample["sample_id"],
        "narrative": narrative,
        "rule_based": rule_based,
        "remedy": REMEDIES[ruling["root_cause"]],
        "model_error": model_error,
    }
