"""归因：规则分类的优先级，以及模型不可用时的降级。"""

from __future__ import annotations

from pathlib import Path

import pytest
from akasha_benchmark.store import connect, repo
from akasha_benchmark.store.migrate import migrate
from akasha_platform import attribution

DATASET = "hotpotqa"


def _sample(**overrides):
    """一条「检索完全正常」的样本，各测试按需覆盖字段。"""
    base = {
        "sample_id": "s1",
        "dataset": DATASET,
        "answer_mode": "knowledge",
        "question": "who won the Grammy and Emmy award",
        "answer": "an answer",
        "reference_answers": ["Cyndi Lauper"],
        "gold_count": 2,
        "retrieved_count": 5,
        "citation_count": 2,
        "metrics": {"hit@5": 1.0, "full_coverage@5": 1.0, "f1": 0.8, "recall@5": 1.0},
        "detail": {},
    }
    return {**base, **overrides}


def _lineage(lost: list[str]):
    return {"gold": [{"doc_id": "d1", "page_id": "p1", "diff": {"question_terms_lost": lost}}]}


# --- 优先级（这一组是红线）--------------------------------------------------


@pytest.mark.parametrize("mode", ["general", "no_match"])
def test_generation_fallback_wins_over_every_retrieval_signal(mode: str):
    """生成端回落必须最先判，哪怕检索信号看起来也很糟。"""
    ruling = attribution.classify(
        _sample(answer_mode=mode, metrics={"hit@5": 0.0, "recall@5": 0.0, "f1": 0.0}),
        _lineage(["grammy"]),
    )
    assert ruling["root_cause"] == attribution.CAUSE_GENERATION_FALLBACK
    assert mode in ruling["labels"]


def test_compiled_away_wins_over_retrieval_miss():
    """编译丢词优先于「没召回到」：后者是前者的表现，不是另一个原因。"""
    ruling = attribution.classify(
        _sample(metrics={"hit@5": 0.0, "recall@5": 0.0}), _lineage(["grammy", "emmy"])
    )
    assert ruling["root_cause"] == attribution.CAUSE_COMPILED_AWAY
    # 去重后排序，所以顺序与 gold 的出现顺序无关。
    assert ruling["evidence"]["question_terms_lost"] == ["emmy", "grammy"]
    assert "调参救不了" in attribution.REMEDIES[ruling["root_cause"]]


def test_citation_dropped_when_gold_was_retrieved_but_truncated():
    """召回到了却没引用：检索没问题，问题在引用预算。"""
    ruling = attribution.classify(
        _sample(metrics={"hit@5": 1.0, "full_coverage@5": 1.0, "truncated_gold": 2.0, "f1": 0.5}),
        _lineage([]),
    )
    assert ruling["root_cause"] == attribution.CAUSE_CITATION_DROPPED
    assert "truncated:2" in ruling["labels"]


def test_retrieval_miss_when_nothing_hit_and_the_words_survived():
    """词还在编译产物里却没召回到 —— 这是排序或阈值问题，属于可调范围。"""
    ruling = attribution.classify(_sample(metrics={"hit@5": 0.0, "recall@5": 0.0}), _lineage([]))
    assert ruling["root_cause"] == attribution.CAUSE_RETRIEVAL_MISS


def test_graph_edge_missing_when_coverage_is_partial_without_graph_help():
    """命中了但没凑齐 gold，且图扩展没有独有贡献 —— 多跳缺跳。"""
    ruling = attribution.classify(
        _sample(
            metrics={
                "hit@5": 1.0,
                "full_coverage@5": 0.0,
                "graph_exclusive_gold_count": 0.0,
                "f1": 0.4,
            }
        ),
        _lineage([]),
    )
    assert ruling["root_cause"] == attribution.CAUSE_GRAPH_EDGE_MISSING


def test_gold_suspect_when_retrieval_is_perfect_but_the_answer_scores_low():
    """检索与引用都对，答案仍判错：先怀疑参考答案或评分口径，而不是系统。"""
    ruling = attribution.classify(
        _sample(metrics={"hit@5": 1.0, "full_coverage@5": 1.0, "f1": 0.1}), _lineage([])
    )
    assert ruling["root_cause"] == attribution.CAUSE_GOLD_SUSPECT


def test_hit_missing_is_not_treated_as_zero():
    """没有 ``hit@k`` 这一项（比如 narrativeqa 无 gold）不等于 hit=0。"""
    ruling = attribution.classify(
        _sample(metrics={"f1": 0.6}, gold_count=0), _lineage([])
    )
    assert ruling["root_cause"] != attribution.CAUSE_RETRIEVAL_MISS
    assert ruling["evidence"]["hit"] is None


def test_lineage_unavailable_is_recorded_not_silently_ignored():
    """没配只读库时 compiled_away 判不了 —— 要在证据里说清楚，不能假装判过。"""
    ruling = attribution.classify(_sample(metrics={"hit@5": 0.0}), None)
    assert ruling["root_cause"] == attribution.CAUSE_RETRIEVAL_MISS
    assert ruling["evidence"]["lineage_available"] is False
    assert "lineage-unavailable" in ruling["labels"]


# --- 写库与降级 --------------------------------------------------------------


@pytest.fixture
def seeded(tmp_path: Path):
    db = tmp_path / "t.db"
    migrate(db, verbose=False)
    connection = connect(db)
    repo.upsert_dataset(
        connection,
        name=DATASET,
        adapter="A",
        adapter_version="1",
        provides=["gold_docs", "reference_answers"],
        identity_rules={},
        qa_path="q",
        qa_sha256="0" * 64,
        qa_rows=1,
        corpus_path="c",
        corpus_sha256="1" * 64,
        corpus_rows=1,
        dedup_stats={},
        gold_count_distribution={},
        unique_question_texts=1,
    )
    layer_id = repo.create_index_layer(
        connection,
        label="L",
        subset_hash="sh",
        seed=1,
        qa_limit=1,
        negatives_ratio=1.0,
        narrativeqa_docs=1,
    )
    query_layer_id = repo.create_query_layer(
        connection,
        index_layer_id=layer_id,
        label="Q",
        config_hash="qh",
        score_threshold=None,
        concurrency=1,
        request_interval_seconds=0.5,
        model_configs=None,
        model_configs_match_index=True,
        allow_config_drift=False,
    )
    eval_id = repo.create_eval_layer(
        connection,
        query_layer_id=query_layer_id,
        label="E",
        config_hash="eh",
        ks=[5],
        metrics=["recall"],
    )
    connection.commit()
    yield connection, eval_id
    connection.close()


def test_rule_only_analysis_is_a_valid_result(seeded):
    """没配分析模型时只写规则结论 —— 那仍然是一条有效归因，不是失败。"""
    connection, eval_id = seeded
    result = attribution.analyze(
        connection, eval_id, _sample(answer_mode="general"), None, use_model=False
    )
    assert result["rule_based"] is True
    assert result["narrative"] is None
    assert result["root_cause"] == attribution.CAUSE_GENERATION_FALLBACK

    stored = repo.badcase_analyses(connection, eval_id)
    assert len(stored) == 1
    assert stored[0]["root_cause"] == attribution.CAUSE_GENERATION_FALLBACK


def test_a_missing_analysis_provider_degrades_instead_of_failing(seeded):
    """要模型但没配好：规则结论照样写，失败原因记进证据。"""
    connection, eval_id = seeded
    result = attribution.analyze(connection, eval_id, _sample(), None, use_model=True)
    assert result["rule_based"] is True
    assert "no analysis model provider" in (result["model_error"] or "")

    stored = repo.badcase_analyses(connection, eval_id)[0]
    assert "model_error" in stored["evidence"]


def test_analysis_also_lands_in_annotations_for_agreement(seeded):
    """归因结论要进 annotation，否则 judge-human 一致率算不了。"""
    connection, eval_id = seeded
    attribution.analyze(connection, eval_id, _sample(answer_mode="general"), None, use_model=False)

    notes = repo.annotations_for(connection, "sample", "s1")
    assert len(notes) == 1
    assert notes[0]["author_kind"] == "model"
    assert attribution.CAUSE_GENERATION_FALLBACK in repo.loads(notes[0]["labels_json"], [])


def test_re_analysing_replaces_rather_than_duplicates(seeded):
    """重跑归因是「换一个更好的判断」，不是追加一条。"""
    connection, eval_id = seeded
    attribution.analyze(connection, eval_id, _sample(answer_mode="general"), None, use_model=False)
    attribution.analyze(connection, eval_id, _sample(metrics={"hit@5": 0.0}), None, use_model=False)

    stored = repo.badcase_analyses(connection, eval_id)
    assert len(stored) == 1
    assert stored[0]["root_cause"] == attribution.CAUSE_RETRIEVAL_MISS


def test_cause_counts_group_by_root_cause(seeded):
    """归因层总览读它：哪一类占多数决定先修什么。"""
    connection, eval_id = seeded
    for sample_id, mode in (("a", "general"), ("b", "general"), ("c", "knowledge")):
        attribution.analyze(
            connection,
            eval_id,
            _sample(sample_id=sample_id, answer_mode=mode, metrics={"hit@5": 0.0}),
            None,
            use_model=False,
        )
    counts = repo.badcase_cause_counts(connection, eval_id)
    assert counts[attribution.CAUSE_GENERATION_FALLBACK] == 2
    assert counts[attribution.CAUSE_RETRIEVAL_MISS] == 1


def test_every_cause_has_a_remedy():
    """每个根因都要给出处置建议 —— 「这条能不能靠调参救」是最有用的一句。"""
    causes = {
        value
        for name, value in vars(attribution).items()
        if name.startswith("CAUSE_") and isinstance(value, str)
    }
    assert causes == set(attribution.REMEDIES)
