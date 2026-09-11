"""指标勾选：勾了的才算，没勾的连同原因一起报告。

**两种「没有值」必须分开说**：缺依赖是「这个数据集永远算不了」（narrativeqa 没有
gold 文档），没勾选是「这一轮没要」。混成一句话的话，读者会把后者当成前者，
进而以为其他组也缺 gold 标注。

另一条是过滤发生的位置：逐样本的检索族指标是一次算出来的一组，拆开单算不会更快,
所以过滤在**写库之前**做，而 ``detail`` 保留全部明细 —— 那是归因要读的原始链路。
"""

from __future__ import annotations

import pytest
from akasha_benchmark import evaluate
from akasha_benchmark.metrics import registry


def test_no_selection_means_everything():
    """命令行不传 ``--metric`` 就是全量，这条路不能变。"""
    names, selected = evaluate.resolve_metrics(None)
    assert selected is None
    assert names == sorted(registry.METRIC_REGISTRY)
    assert evaluate.resolve_metrics([])[1] is None


def test_unknown_metric_names_are_rejected():
    """拼错的指标名要报错，不能静默跑全量。

    静默的后果是那份报告比预期多出好几列，而没有任何地方提示过。
    """
    with pytest.raises(ValueError, match="unknown metrics"):
        evaluate.resolve_metrics(["recall", "recal"])


def test_selection_is_deduplicated_and_sorted():
    names, selected = evaluate.resolve_metrics(["f1", "recall", "recall"])
    assert names == ["f1", "recall"]
    assert selected == frozenset({"f1", "recall"})


def test_filter_strips_the_k_before_comparing():
    """勾选名不带 k（``recall``），实际指标名带（``recall@5``）。"""
    keep = evaluate._metric_filter(frozenset({"recall"}), (2, 5))
    assert keep("recall@2") and keep("recall@5")
    assert not keep("ndcg@5")
    assert not keep("f1")


def test_filter_keeps_names_the_registry_does_not_know():
    """registry 里没声明的名字一律保留 —— 让它悄悄消失比留着更糟。"""
    keep = evaluate._metric_filter(frozenset({"recall"}), (5,))
    assert keep("some_future_metric")


def test_omission_reason_separates_missing_dependencies_from_deselection():
    """两种原因分开陈述，且都能被读到。"""
    both = evaluate._omission_reason("narrativeqa", has_gold=False, deselected=["f1"])
    assert "does not provide GOLD_DOCS" in both
    assert "not selected for this run" in both

    only_deselected = evaluate._omission_reason("hotpotqa", has_gold=True, deselected=["ndcg"])
    assert "GOLD_DOCS" not in only_deselected
    assert "ndcg" in only_deselected

    assert evaluate._omission_reason("hotpotqa", has_gold=True, deselected=[]) is None


def test_available_metrics_follow_the_dependency_sets():
    """勾选范围由数据集的 ``provides`` 决定，不由数据集名字决定。

    faithfulness 的 requires 是空集，所以它对四组都成立 —— narrativeqa
    整族检索指标省略之后，它恰好能填上那个洞。
    """
    from akasha_benchmark.datasets import get_adapter

    hotpot = {d.name for d in registry.available(get_adapter("hotpotqa").provides)}
    narrative = {d.name for d in registry.available(get_adapter("narrativeqa").provides)}

    assert "recall" in hotpot
    assert "recall" not in narrative
    assert "faithfulness" in hotpot and "faithfulness" in narrative
    assert "f1" in narrative
