"""指标口径：排名、依赖闸门、归因判据优先级。"""

from __future__ import annotations

import pytest

from akasha_benchmark import attribution
from akasha_benchmark.datasets import DataDependency, DependencyError
from akasha_benchmark.metrics import attribution as citation
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
    assert result["evidence_verifiable_rate"] == pytest.approx(1.0)


def test_graph_exclusive_gold_is_net_contribution():
    """图扩展的净价值 = 只有它才拿到的 gold，不含语义召回本来就能找到的。"""
    page_to_doc = {"p1": "g1", "p2": "g2"}
    snippets = [
        {"retrievalReasons": ["semantic"], "sourceWindows": [{"sourcePageId": "p1"}]},
        {"retrievalReasons": ["graph-neighbor"], "sourceWindows": [{"sourcePageId": "p2"}]},
    ]
    result = multihop.evaluate_sample(snippets, ["g1", "g2"], page_to_doc)
    assert result["graph_exclusive_gold_count"] == 1
    assert result["graph_neighbor_share"] == pytest.approx(0.5)
    assert result["graph_neighbor_precision"] == pytest.approx(1.0)


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


# ------------------------------------------------------------ 归因判据


def _sample(metrics: dict, mode: str = "knowledge", gold=("g1",)) -> dict:
    return {
        "answer_mode": mode,
        "metrics": metrics,
        "answer": "x",
        "detail": {"gold_doc_ids": list(gold), "question": "q"},
    }


def test_correct_answer_is_not_a_failure():
    """最差 N 条可能包含正确答案，应排除这些样本的失败归因。"""
    # 检索一条 gold 都没命中，但答案 EM 命中：系统没依赖那篇 gold。
    ruling = attribution.classify(_sample({"hit@5": 0.0, "em": 1.0}), [])
    assert ruling["root_cause"] == attribution.CAUSE_NOT_A_FAILURE

    # judge 判事实一致，同样算对。
    ruling = attribution.classify(_sample({"hit@5": 0.0, "em": 0.0, "answer_correctness": 1.0}), [])
    assert ruling["root_cause"] == attribution.CAUSE_NOT_A_FAILURE


def test_not_a_failure_outranks_every_failure_cause():
    """答案正确的判据优先于其他失败判据。"""
    for metrics, lineage, mode in (
        ({"hit@5": 0.0, "em": 1.0}, [{"question_terms_lost": ["grammy"]}], "knowledge"),
        ({"hit@5": 0.0, "em": 1.0, "truncated_gold": 1.0}, [], "knowledge"),
        ({"hit@5": 0.0, "em": 1.0}, None, "general"),
    ):
        ruling = attribution.classify(_sample(metrics, mode=mode), lineage)
        assert ruling["root_cause"] == attribution.CAUSE_NOT_A_FAILURE


def test_high_f1_alone_does_not_clear_a_sample():
    """高词汇重叠率不足以判定答案正确。"""
    ruling = attribution.classify(_sample({"hit@5": 0.0, "em": 0.0, "f1": 0.9}), [])
    assert ruling["root_cause"] != attribution.CAUSE_NOT_A_FAILURE


def test_fallback_is_judged_before_retrieval():
    """生成端拒答必须最先判，否则它那 0 分会被解释成检索失败。"""
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


def test_citation_dropped_outranks_retrieval_miss():
    ruling = attribution.classify(_sample({"hit@5": 0.0, "truncated_gold": 1.0}), [])
    assert ruling["root_cause"] == attribution.CAUSE_CITATION_DROPPED


def test_graph_edge_missing_when_coverage_incomplete():
    ruling = attribution.classify(
        _sample({"hit@5": 1.0, "full_coverage@5": 0.0, "graph_exclusive_gold_count": 0.0}), []
    )
    assert ruling["root_cause"] == attribution.CAUSE_GRAPH_EDGE_MISSING


def test_gold_suspect_when_everything_retrieved_but_answer_wrong():
    ruling = attribution.classify(_sample({"hit@5": 1.0, "full_coverage@5": 1.0, "f1": 0.1}), [])
    assert ruling["root_cause"] == attribution.CAUSE_GOLD_SUSPECT


def test_every_cause_has_a_remedy():
    """「这条能不能靠调参救」是归因结论里最有用的一句，不能缺。"""
    causes = {
        value
        for name, value in vars(attribution).items()
        if name.startswith("CAUSE_") and isinstance(value, str)
    }
    assert causes == set(attribution.REMEDIES)
