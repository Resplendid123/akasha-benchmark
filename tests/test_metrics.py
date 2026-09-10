"""指标算法：拿手算的值对，外加 capability 闸门。

这些数字都是能手算验证的，不是「跑一遍看着差不多」——
指标算错了不会报错，只会给出一个看着合理的假结果。
"""

from __future__ import annotations

from math import log2

import pytest

from akasha_benchmark.datasets import Capability, CapabilityError, get_adapter
from akasha_benchmark.metrics import attribution, multihop, qa, retrieval


def test_recall_and_hit_at_k():
    """Recall 看命中比例，Hit 只看有没有命中，k 的边界要对。"""
    ranked = ["a", "b", "c", "d"]
    gold = ["c", "z"]
    assert retrieval.recall_at_k(ranked, gold, 2) == 0.0
    assert retrieval.recall_at_k(ranked, gold, 3) == 0.5
    assert retrieval.hit_at_k(ranked, gold, 2) == 0.0
    assert retrieval.hit_at_k(ranked, gold, 3) == 1.0


def test_mrr_uses_first_gold_rank():
    """MRR 只认首个 gold 的位置。"""
    assert retrieval.mrr(["x", "gold"], ["gold"]) == 0.5
    assert retrieval.mrr(["gold", "x"], ["gold"]) == 1.0
    assert retrieval.mrr(["x", "y"], ["gold"]) == 0.0


def test_ndcg_matches_manual_computation():
    """gold 在第 2、3 位；理想排序是它们占第 1、2 位。"""
    ranked = ["x", "g1", "g2", "y"]
    gold = ["g1", "g2"]
    dcg = 1 / log2(3) + 1 / log2(4)
    idcg = 1 / log2(2) + 1 / log2(3)
    assert retrieval.ndcg_at_k(ranked, gold, 4) == pytest.approx(dcg / idcg)
    assert retrieval.ndcg_at_k(gold, gold, 4) == pytest.approx(1.0)


def test_full_coverage_needs_every_gold():
    """凑齐全部 gold 才算 1，差一篇就是 0。"""
    assert retrieval.full_coverage(["g1", "g2"], ["g1", "g2"], 2) == 1.0
    assert retrieval.full_coverage(["g1", "x"], ["g1", "g2"], 2) == 0.0
    # 两篇都在，但第二篇落在 k 之外。
    assert retrieval.full_coverage(["g1", "x", "g2"], ["g1", "g2"], 2) == 0.0


def test_ranked_doc_ids_keeps_rank_slot_for_unmapped_pages():
    """反查不到的 page 要占住名次，同一个 page 重复出现只占一个名次。"""
    retrieved = [
        {"sourcePageId": "p-unknown"},
        {"sourcePageId": "p1"},
        {"sourcePageId": "p1"},  # 同一个 page，只占一个名次
    ]
    page_map = {"p1": "doc1"}
    ranked = retrieval.ranked_doc_ids(retrieved, page_map)
    assert ranked == ["__unmapped__:p-unknown", "doc1"]
    # 未知 page 不能把 doc1 顶到第 1 位，否则 MRR 会被算高。
    assert retrieval.mrr(ranked, ["doc1"]) == 0.5
    assert retrieval.unmapped_page_ids(retrieved, page_map) == ["p-unknown"]


def test_retrieval_metrics_refuse_datasets_without_gold():
    """narrativeqa 没有 gold，请求检索指标必须抛异常而不是返回 0。"""
    narrativeqa = get_adapter("narrativeqa")
    assert Capability.EVIDENCE_RECALL not in narrativeqa.capabilities
    with pytest.raises(CapabilityError):
        retrieval.require_evidence_capability("narrativeqa", narrativeqa.capabilities)
    # 另外三组声明了该 capability，应当放行。
    for name in ("hotpotqa", "2wikimultihopqa", "musique"):
        retrieval.require_evidence_capability(name, get_adapter(name).capabilities)


def test_recall_raises_without_gold_rather_than_returning_zero():
    """gold 为空时分母无定义，报错而不是返回 0.0。"""
    with pytest.raises(ValueError):
        retrieval.recall_at_k(["a"], [], 5)
    with pytest.raises(ValueError):
        retrieval.ndcg_at_k(["a"], [], 5)


def test_normalize_answer_is_the_standard_recipe():
    """标准口径：小写、去标点、去冠词、合并空白。"""
    assert qa.normalize_answer("The Beatles!") == "beatles"
    assert qa.normalize_answer("  A  Hard   Day's Night ") == "hard days night"
    assert qa.normalize_answer("An apple, an orange.") == "apple orange"


def test_f1_takes_max_over_references():
    """多参考取 max。"""
    scored = qa.score_answer("the beatles", ["The Beatles", "Beatles band"])
    assert scored["f1"] == 1.0

    partial = qa.score_answer("John Lennon", ["Lennon"])
    # 共有 1 个词；precision 1/2、recall 1/1，F1 = 2/3。
    assert partial["f1"] == pytest.approx(2 / 3)


def test_score_answer_reports_no_exact_match():
    """EM 是刻意去掉的：对散文答案它恒等于 0，不随质量变化。

    这条钉住「不报 EM」这个决定本身 —— 哪天有人顺手把它加回来，
    报告里就会多出一列恒为 0 的数，读者会据此判断系统坏了。
    """
    scored = qa.score_answer("the beatles", ["The Beatles"])
    assert set(scored) == {"f1"}, f"score_answer should report f1 only, got {sorted(scored)}"
    assert not hasattr(qa, "exact_match")


def test_f1_survives_a_verbose_answer_where_em_could_not():
    """散文答案里含有正确短答案时，F1 仍是正数 —— 这就是保留它的理由。

    数字很低（这里约 0.13），所以 F1 的绝对值不可跨系统比较；
    但它随答案质量变化，而 EM 在这种形态下恒为 0。
    """
    verbose = (
        "The disease described is yellow fever, which is caused by the yellow fever "
        "virus belonging to the genus Flavivirus."
    )
    scored = qa.score_answer(verbose, ["Flavivirus"])
    assert 0.0 < scored["f1"] < 0.3, scored


def test_f1_empty_prediction():
    """退化情况按官方 SQuAD 脚本口径。"""
    assert qa.token_f1("", "something") == 0.0
    assert qa.token_f1("", "") == 1.0


def test_answer_mode_distribution():
    """缺失的 answerMode 单独归到 missing，不能悄悄并进别的桶。"""
    dist = qa.answer_mode_distribution(["knowledge", "knowledge", "no_match", None])
    assert dist == {"knowledge": 0.5, "no_match": 0.25, "missing": 0.25}


def test_attribution_separates_truncated_gold():
    """被检索到但没进引用的 gold，要能和「检索没找到」区分开。"""
    page_map = {"p1": "d1", "p2": "d2", "p3": "d3"}
    retrieved = [{"sourcePageId": "p1"}, {"sourcePageId": "p2"}, {"sourcePageId": "p3"}]
    citations = [{"sourcePageId": "p1"}]
    evidence = [{"sourcePageId": "p1", "excerpts": ["quote"]}, {"sourcePageId": "p9", "excerpts": []}]

    scored = attribution.evaluate_sample(citations, retrieved, evidence, ["d1", "d2"], page_map)
    assert scored["citation_precision"] == 1.0
    assert scored["citation_recall"] == 0.5
    assert scored["truncation_loss"] == 2.0
    # d2 被检索到了却没被引用：这是过滤损失，不是检索没命中。
    assert scored["truncated_gold"] == 1.0
    assert scored["evidence_verifiable_rate"] == 0.5


def test_multihop_graph_exclusive_gold():
    """只有图扩展才拿到的 gold，要单独算出来。"""
    page_map = {"pa": "d1", "pb": "d2"}
    snippets = [
        {"retrievalReasons": ["semantic"], "sourceWindows": [{"sourcePageId": "pa"}]},
        {"retrievalReasons": ["graph-neighbor"], "sourceWindows": [{"sourcePageId": "pb"}]},
    ]
    scored = multihop.evaluate_sample(snippets, ["d1", "d2"], page_map)
    assert scored["graph_neighbor_share"] == 0.5
    assert scored["graph_neighbor_precision"] == 1.0
    # d2 只来自图扩展；d1 语义召回本来就找到了。
    assert scored["graph_exclusive_gold_count"] == 1
    assert scored["graph_exclusive_gold_share"] == 0.5


def test_multihop_gold_found_by_both_signals_is_not_graph_exclusive():
    """两个信号都找到的 gold 不算图扩展的净增量。"""
    page_map = {"pa": "d1"}
    snippets = [
        {"retrievalReasons": ["semantic"], "sourceWindows": [{"sourcePageId": "pa"}]},
        {"retrievalReasons": ["graph-neighbor"], "sourceWindows": [{"sourcePageId": "pa"}]},
    ]
    scored = multihop.evaluate_sample(snippets, ["d1"], page_map)
    assert scored["graph_exclusive_gold_count"] == 0


def test_multihop_aggregate_reason_rates():
    """按信号汇总时，计数要累加而不是取均值，命中率再由累加值算。"""
    page_map = {"pa": "d1"}
    rows = [
        multihop.evaluate_sample(
            [{"retrievalReasons": ["semantic"], "sourceWindows": [{"sourcePageId": "pa"}]}],
            ["d1"],
            page_map,
        ),
        multihop.evaluate_sample(
            [{"retrievalReasons": ["semantic"], "sourceWindows": []}], ["d1"], page_map
        ),
    ]
    agg = multihop.aggregate(rows)
    assert agg["reason_totals"] == {"semantic": 2}
    assert agg["reason_gold_totals"] == {"semantic": 1}
    assert agg["reason_gold_rate"]["semantic"] == 0.5
