"""指标口径：排名、依赖闸门、归因判据优先级。"""

from __future__ import annotations

import pytest

from akasha_benchmark import attribution
from akasha_benchmark.datasets import DataDependency, DependencyError
from akasha_benchmark.metrics import attribution as citation
from akasha_benchmark.metrics.interpretation import (
    build_metric_evidence,
    interpret_sample_metrics,
)
from akasha_benchmark.metrics import multihop, qa, registry, retrieval


def test_unmapped_pages_keep_their_rank():
    """丢掉未映射的 page 会让后面的结果整体前移，把对排名敏感的指标都算高。"""
    retrieved = [{"sourcePageId": "unknown"}, {"sourcePageId": "p1"}]
    ranked = retrieval.ranked_doc_ids(retrieved, {"p1": "d1"})
    assert len(ranked) == 2
    assert ranked[1] == "d1"
    assert retrieval.mrr(ranked, ["d1"]) == pytest.approx(0.5)
    assert retrieval.unmapped_page_ids(retrieved, {"p1": "d1"}) == ["unknown"]


def test_full_coverage_differs_from_recall():
    """多跳少一跳就答不对，所以「凑齐全部 gold」比 recall 均值更贴近实际需求。"""
    ranked = ["d1", "x"]
    gold = ["d1", "d2"]
    assert retrieval.recall_at_k(ranked, gold, 2) == pytest.approx(0.5)
    assert retrieval.full_coverage(ranked, gold, 2) == pytest.approx(0.0)
    assert retrieval.full_coverage(["d1", "d2"], gold, 2) == pytest.approx(1.0)


def test_ndcg_rewards_earlier_gold():
    early = retrieval.ndcg_at_k(["d1", "x", "y"], ["d1"], 3)
    late = retrieval.ndcg_at_k(["x", "y", "d1"], ["d1"], 3)
    assert early == pytest.approx(1.0)
    assert late < early


def test_retrieval_refuses_without_gold():
    """没有 gold 时拒绝计算，不返回 0.0。"""
    with pytest.raises(ValueError):
        retrieval.recall_at_k(["d1"], [], 2)


def test_registry_gate_is_set_comparison():
    """判据是 provides 与 requires 的集合比对，不是数据集名字。"""
    gold = frozenset({DataDependency.GOLD_DOCS, DataDependency.REFERENCE_ANSWERS})
    assert registry.require("hotpotqa", gold, "recall").name == "recall"
    with pytest.raises(DependencyError):
        registry.require("narrativeqa", frozenset({DataDependency.REFERENCE_ANSWERS}), "recall")
    # faithfulness 的依赖是空集，所以它对任何数据集都成立。
    assert registry.require("narrativeqa", frozenset(), "faithfulness").kind == "judge"


def test_truncation_loss_separates_retrieval_from_citation():
    """召回到了但没被引用 —— 那是引用过滤太严，不是检索没找到。要调的地方不同。"""
    page_to_doc = {"p1": "gold", "p2": "other"}
    result = citation.evaluate_sample(
        citations=[{"sourcePageId": "p2"}],
        retrieved=[{"sourcePageId": "p1"}, {"sourcePageId": "p2"}],
        citation_evidence=[{"excerpts": ["x"]}],
        gold=["gold"],
        page_to_doc=page_to_doc,
    )
    assert result["truncation_loss"] == pytest.approx(1.0)
    assert result["truncated_gold"] == pytest.approx(1.0)
    assert result["citation_recall"] == pytest.approx(0.0)
    assert "evidence_verifiable_rate" not in result
    assert "evidence_verifiable_rate" not in registry.METRIC_REGISTRY
    assert "citation_count" not in result
    assert "citation_count" not in registry.METRIC_REGISTRY
    assert "retrieved_count" not in result
    assert "retrieved_count" not in registry.METRIC_REGISTRY
    assert "evidence_entries" not in result
    assert "evidence_entries" not in registry.METRIC_REGISTRY


def test_graph_exclusive_gold_is_net_contribution():
    """图扩展的净价值 = 只有它才拿到的 gold，不含语义召回本来就能找到的。"""
    page_to_doc = {"p1": "g1", "p2": "g2"}
    snippets = [
        {"retrievalReasons": ["semantic"], "sourceWindows": [{"sourcePageId": "p1"}]},
        {"retrievalReasons": ["graph-neighbor"], "sourceWindows": [{"sourcePageId": "p2"}]},
    ]
    result = multihop.evaluate_sample(snippets, ["g1", "g2"], page_to_doc)
    assert result["graph_exclusive_gold_share"] == pytest.approx(0.5)
    assert "graph_exclusive_gold_count" not in result
    assert "graph_exclusive_gold_count" not in registry.METRIC_REGISTRY
    assert "graph_neighbor_share" not in result
    assert "graph_neighbor_share" not in registry.METRIC_REGISTRY
    assert "snippet_count" not in result
    assert "snippet_count" not in registry.METRIC_REGISTRY
    assert "graph_neighbor_snippets" not in result
    assert "graph_neighbor_snippets" not in registry.METRIC_REGISTRY


def test_direct_and_graph_hit_is_not_graph_exclusive():
    result = multihop.evaluate_sample(
        [
            {
                "retrievalReasons": ["semantic", "graph-neighbor"],
                "sourceWindows": [{"sourcePageId": "p1"}],
            }
        ],
        ["g1"],
        {"p1": "g1"},
    )
    assert result["graph_exclusive_gold_share"] == 0.0
    assert result["graph_neighbor_precision"] == pytest.approx(1.0)
    assert "graph_neighbor_precision" in registry.METRIC_REGISTRY


def test_graph_neighbor_precision_deduplicates_documents():
    page_to_doc = {"gold-page": "gold", "other-page": "other"}
    snippets = [
        {
            "retrievalReasons": ["graph-neighbor"],
            "sourceWindows": [{"sourcePageId": "gold-page"}],
        },
        *[
            {
                "retrievalReasons": ["graph-neighbor"],
                "sourceWindows": [{"sourcePageId": "other-page"}],
            }
            for _ in range(3)
        ],
    ]

    result = multihop.evaluate_sample(snippets, ["gold"], page_to_doc)

    assert result["graph_neighbor_precision"] == pytest.approx(0.5)
    assert result["reason_doc_counts"]["graph-neighbor"] == 2


def test_answer_scoring_takes_max_over_references():
    scored = qa.score_answer("Rita Moreno", ["someone else", "rita moreno"])
    assert scored["em"] == pytest.approx(1.0)
    assert scored["f1"] == pytest.approx(1.0)
    assert qa.normalize_answer("The  Answer!") == "answer"


def test_answer_mode_distribution():
    assert qa.answer_mode_distribution(["knowledge", "no_match", None]) == {
        "knowledge": pytest.approx(1 / 3),
        "missing": pytest.approx(1 / 3),
        "no_match": pytest.approx(1 / 3),
    }


def test_sample_metric_interpretations_explain_values_with_sample_counts():
    rows = interpret_sample_metrics(
        ["recall", "mrr", "truncated_gold", "faithfulness"],
        [5],
        {"recall@5": 0.5, "mrr": 0.5, "truncated_gold": 1.0, "faithfulness": 2 / 3},
        {"gold_doc_ids": ["g1", "g2"]},
        [
            {
                "metric": "faithfulness",
                "score": 2 / 3,
                "failure_kind": None,
                "detail": {"claim_count": 3, "supported": 2},
            }
        ],
        [],
    )
    by_name = {row["name"]: row for row in rows}
    assert by_name["recall@5"]["reason"] == "前 5 条检索结果命中 1/2 篇 gold 文档。"
    assert by_name["mrr"]["reason"] == "首个 gold 文档约位于第 2 名。"
    assert by_name["truncated_gold"]["status"] == "bad"
    assert by_name["faithfulness"]["reason"] == "2/3 条事实陈述有检索证据支持。"


def test_sample_metric_interpretations_explain_missing_and_failed_values():
    rows = interpret_sample_metrics(
        ["recall", "answer_relevancy", "answer_correctness"],
        [2],
        {},
        {},
        [
            {"metric": "answer_relevancy", "score": None, "failure_kind": None},
            {"metric": "answer_correctness", "score": None, "failure_kind": "timeout"},
        ],
        ["recall"],
    )
    by_name = {row["name"]: row for row in rows}
    assert "缺少" in by_name["recall@2"]["reason"]
    assert "无定义" in by_name["answer_relevancy"]["reason"]
    assert "timeout" in by_name["answer_correctness"]["reason"]
    assert all(row["status"] == "unavailable" for row in rows)


def test_every_registered_metric_has_structured_evidence_and_formula():
    configured = sorted(registry.METRIC_REGISTRY)
    actual_names = [f"{name}@2" if registry.get_metric(name).per_k else name for name in configured]
    values = {name: 0.5 for name in actual_names}
    evidence = build_metric_evidence(
        configured,
        [2],
        values,
        {"gold_doc_ids": ["g1"], "reference_answers": ["reference answer"]},
        {
            "answer": "answer",
            "retrievedSources": [{"sourcePageId": "p1"}],
            "citations": [{"sourcePageId": "p1"}],
            "citationEvidence": [{"sourcePageId": "p1", "excerpts": ["quote"]}],
            "snippets": [
                {
                    "id": "s1",
                    "title": "Gold",
                    "text": "evidence",
                    "retrievalReasons": ["semantic", "graph-neighbor"],
                    "sourceWindows": [{"sourcePageId": "p1"}],
                }
            ],
        },
        {"p1": "g1"},
        {"g1": {"doc_id": "g1", "title": "Gold"}},
        [
            {"metric": name, "detail": {}}
            for name in configured
            if registry.get_metric(name).kind == registry.KIND_JUDGE
        ],
    )

    assert set(evidence) == set(actual_names)
    assert all(row.get("formula") for row in evidence.values())


# ------------------------------------------------------------ 归因判据


def _sample(metrics: dict, mode: str = "knowledge", gold=("g1",)) -> dict:
    return {
        "answer_mode": mode,
        "metrics": metrics,
        "answer": "x",
        "detail": {"gold_doc_ids": list(gold), "question": "q"},
    }


def test_correct_answer_is_not_a_failure():
    """全量归因包含正确答案，应排除这些样本的失败归因。"""
    # 检索一条 gold 都没命中，但答案 EM 命中：系统没依赖那篇 gold。
    ruling = attribution.classify(_sample({"hit@5": 0.0, "em": 1.0}), [])
    assert ruling["root_cause"] == attribution.CAUSE_NOT_A_FAILURE


def test_rule_attribution_ignores_judge_metrics():
    """规则归因不能因评测是否配置 Judge 而改变。"""
    ruling = attribution.classify(
        _sample(
            {
                "hit@5": 0.0,
                "answer_correctness": 1.0,
                "faithfulness": 1.0,
                "answer_relevancy": 1.0,
                "context_relevancy": 1.0,
            }
        ),
        [],
    )
    assert ruling["root_cause"] == attribution.CAUSE_RETRIEVAL_MISS
    assert "faithfulness" not in ruling["evidence"]


def test_not_a_failure_outranks_every_failure_cause():
    """答案明确正确时，不再追究回答模式、检索或引用信号。"""
    for metrics, lineage, mode in (
        ({"hit@5": 0.0, "em": 1.0}, [{"question_terms_lost": ["grammy"]}], "knowledge"),
        ({"hit@5": 0.0, "em": 1.0, "truncated_gold": 1.0}, [], "knowledge"),
        ({"hit@5": 0.0, "em": 1.0}, None, "general"),
    ):
        ruling = attribution.classify(_sample(metrics, mode=mode), lineage)
        assert ruling["root_cause"] == attribution.CAUSE_NOT_A_FAILURE


def test_correct_general_answer_is_not_a_failure():
    """general 回答明确正确时，同样优先归入正常样本。"""
    em_hit = attribution.classify(
        _sample({"hit@5": 1.0, "em": 1.0}, mode="general"), None
    )
    assert em_hit["root_cause"] == attribution.CAUSE_NOT_A_FAILURE

    contained = _sample({"hit@5": 1.0, "em": 0.0}, mode="general")
    contained["answer"] = "The answer is Rita Moreno."
    contained["detail"]["reference_answers"] = ["Rita Moreno"]
    ruling = attribution.classify(contained, None)
    assert ruling["root_cause"] == attribution.CAUSE_NOT_A_FAILURE


def test_fully_supported_answer_is_not_a_failure_even_with_long_context():
    """解释性答案完整包含参考答案时，不应因严格 EM 落到 unknown。"""
    sample = _sample(
        {
            "em": 0.0,
            "f1": 0.35,
            "faithfulness": 1.0,
            "hit@5": 1.0,
            "full_coverage@5": 1.0,
            "truncated_gold": 1.0,
        }
    )
    sample["answer"] = (
        "Roger Capellani died during the Battle of Dunkirk, which was fought "
        "between the Allies and Nazi Germany."
    )
    sample["detail"]["reference_answers"] = ["the Allies and Nazi Germany"]
    ruling = attribution.classify(sample, [])
    assert ruling["root_cause"] == attribution.CAUSE_NOT_A_FAILURE
    assert ruling["evidence"]["reference_answer_contained"] is True


@pytest.mark.parametrize(
    "metrics,expected",
    [
        ({"faithfulness": 1.0, "hit@5": 1.0, "full_coverage@5": 1.0}, attribution.CAUSE_UNKNOWN),
        ({"hit@5": 0.0, "em": 0.0, "f1": 0.9}, attribution.CAUSE_RETRIEVAL_MISS),
        ({"hit@5": 1.0, "full_coverage@5": 1.0, "f1": 0.1}, attribution.CAUSE_UNKNOWN),
    ],
)
def test_weak_answer_metrics_do_not_drive_rule_attribution(metrics, expected):
    ruling = attribution.classify(_sample(metrics), [])
    assert ruling["root_cause"] == expected
    assert "f1" not in ruling["evidence"]


def test_fallback_is_judged_before_retrieval():
    """生成端兜底必须最先判，否则它的检索信号会被解释成检索失败。"""
    ruling = attribution.classify(_sample({"hit@5": 0.0}, mode="general"), None)
    assert ruling["root_cause"] == attribution.CAUSE_GENERATION_FALLBACK


def test_compiled_away_needs_lineage():
    lineage = [{"question_terms_lost": ["grammy"]}]
    assert (
        attribution.classify(_sample({"hit@5": 0.0}), lineage)["root_cause"]
        == attribution.CAUSE_COMPILED_AWAY
    )
    # 链路不可用时判不了 compiled_away，退到 retrieval_miss 并记下这一点。
    ruling = attribution.classify(_sample({"hit@5": 0.0}), None)
    assert ruling["root_cause"] == attribution.CAUSE_RETRIEVAL_MISS
    assert ruling["evidence"]["lineage_available"] is False


def test_compiled_away_requires_a_complete_retrieval_miss():
    lineage = [{"question_terms_lost": ["access"]}]

    fully_retrieved = attribution.classify(
        _sample({"hit@10": 1.0, "full_coverage@10": 1.0}), lineage
    )
    assert fully_retrieved["root_cause"] == attribution.CAUSE_UNKNOWN

    partially_retrieved = attribution.classify(
        _sample({"hit@10": 1.0, "full_coverage@10": 0.0}), lineage
    )
    assert partially_retrieved["root_cause"] != attribution.CAUSE_COMPILED_AWAY


def test_citation_drop_outranks_lost_question_terms():
    ruling = attribution.classify(
        _sample(
            {
                "hit@10": 1.0,
                "full_coverage@10": 1.0,
                "truncated_gold": 1.0,
            }
        ),
        [{"question_terms_lost": ["access"]}],
    )
    assert ruling["root_cause"] == attribution.CAUSE_CITATION_DROPPED


def test_citation_dropped_outranks_retrieval_miss():
    ruling = attribution.classify(_sample({"hit@5": 0.0, "truncated_gold": 1.0}), [])
    assert ruling["root_cause"] == attribution.CAUSE_CITATION_DROPPED


def test_graph_edge_missing_when_coverage_incomplete():
    ruling = attribution.classify(
        _sample({"hit@5": 1.0, "full_coverage@5": 0.0, "graph_exclusive_gold_share": 0.0}), []
    )
    assert ruling["root_cause"] == attribution.CAUSE_GRAPH_EDGE_MISSING


def test_general_with_retrieval_is_not_classified_as_no_evidence_fallback():
    ruling = attribution.classify(
        _sample({"hit@5": 1.0, "full_coverage@5": 1.0}, mode="general"), []
    )
    assert ruling["root_cause"] == attribution.CAUSE_GENERATION_IGNORED_RETRIEVAL


def test_general_retrieval_uses_full_detail_when_hit_metric_was_not_selected():
    sample = _sample({}, mode="general")
    sample["detail"]["retrieval"] = {"hit@5": 1.0, "full_coverage@5": 1.0}
    ruling = attribution.classify(sample, [])
    assert ruling["root_cause"] == attribution.CAUSE_GENERATION_IGNORED_RETRIEVAL
