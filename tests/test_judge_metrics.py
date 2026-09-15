"""RAGAS 那几条 judge 判据的解析与计分。

共同的口径：**无定义时回 None，不回 0**。拒答、没有上下文、没有参考答案都属于
无定义 —— 记 0 会把「没找到资料」算成「答错了」，而这两件事的处置完全不同。
"""

from __future__ import annotations

import pytest

from akasha_benchmark.judge import (
    answer_correctness,
    answer_relevancy,
    context_relevancy,
)
from akasha_benchmark.metrics import registry
from akasha_benchmark.store import eval_store


def test_all_judge_metrics_are_registered():
    """判据模块与 registry 必须对齐，否则勾了却没实现，或实现了却勾不到。"""
    judge = {d.name for d in registry.METRIC_DEFINITIONS if d.kind == registry.KIND_JUDGE}
    assert judge == {
        "faithfulness",
        "answer_relevancy",
        "context_relevancy",
        "answer_correctness",
    }


def test_answer_correctness_needs_reference_answers():
    """只有这一条依赖参考答案，其余三条对 narrativeqa 也成立。"""
    from akasha_benchmark.datasets.models import DataDependency

    definition = registry.get_metric("answer_correctness")
    assert DataDependency.REFERENCE_ANSWERS in definition.requires
    for name in ("faithfulness", "answer_relevancy", "context_relevancy"):
        assert registry.get_metric(name).requires == frozenset()


# --- answer_relevancy ---


def test_answer_relevancy_is_the_relevant_share():
    score, detail = answer_relevancy.parse_verdict(
        {
            "sentences": [
                {"sentence": "a", "verdict": "relevant"},
                {"sentence": "b", "verdict": "relevant"},
                {"sentence": "c", "verdict": "irrelevant"},
            ]
        }
    )
    assert score == pytest.approx(2 / 3)
    assert detail["relevant"] == 2
    assert detail["sentence_count"] == 3


def test_answer_relevancy_undefined_on_refusal():
    """没有实质句子时无定义。记 0 会把拒答算成「答偏了」。"""
    score, _ = answer_relevancy.parse_verdict({"sentences": []})
    assert score is None


def test_answer_relevancy_rejects_unknown_verdict():
    with pytest.raises(ValueError, match="unknown verdict"):
        answer_relevancy.parse_verdict({"sentences": [{"sentence": "a", "verdict": "maybe"}]})


def test_answer_relevancy_skips_empty_answer():
    assert answer_relevancy.build_prompt("q", "", {}) is None
    assert answer_relevancy.build_prompt("", "a", {}) is None
    assert answer_relevancy.build_prompt("q", "a", {}) is not None


# --- context_relevancy ---


def test_context_relevancy_is_the_useful_share():
    score, detail = context_relevancy.parse_verdict(
        {
            "passages": [
                {"index": 1, "verdict": "useful"},
                {"index": 2, "verdict": "useless"},
            ]
        },
        expected=2,
    )
    assert score == pytest.approx(0.5)
    assert detail["useful"] == 1


def test_context_relevancy_rejects_missing_verdicts():
    """漏判不能当成 useless —— 那会把「prompt 不听话」伪装成「检索很脏」，
    而这个指标的用途正是判断检索脏不脏。"""
    with pytest.raises(ValueError, match="expected 3 verdicts"):
        context_relevancy.parse_verdict(
            {"passages": [{"index": 1, "verdict": "useful"}]}, expected=3
        )


def test_context_relevancy_numbers_the_passages():
    """上下文要编号，否则模型没法逐条对应。"""
    built = context_relevancy.build_prompt(
        "q",
        "a",
        {"snippets": [{"title": "T1", "text": "x"}, {"title": "T2", "text": "y"}]},
    )
    assert built is not None
    _, user, count = built
    assert count == 2
    assert "[1] T1" in user
    assert "[2] T2" in user


def test_context_relevancy_falls_back_to_titles():
    """没有正文时退到 retrievedSources 的标题，仍然可判。"""
    built = context_relevancy.build_prompt(
        "q", "a", {"retrievedSources": [{"title": "Only a title"}]}
    )
    assert built is not None
    assert built[2] == 1


def test_context_relevancy_skips_when_nothing_retrieved():
    assert context_relevancy.build_prompt("q", "a", {}) is None


# --- answer_correctness ---


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [("correct", 1.0), ("partial", 0.5), ("incorrect", 0.0), ("no_answer", None)],
)
def test_answer_correctness_scores_each_band(verdict, expected):
    score, detail = answer_correctness.parse_verdict({"verdict": verdict, "reason": "r"})
    assert score == expected
    assert detail["verdict"] == verdict


def test_answer_correctness_rejects_unknown_verdict():
    with pytest.raises(ValueError, match="unknown verdict"):
        answer_correctness.parse_verdict({"verdict": "mostly right"})


def test_answer_correctness_skips_without_reference():
    assert answer_correctness.build_prompt("q", "a", "") is None
    assert answer_correctness.build_prompt("q", "", "ref") is None
    assert answer_correctness.build_prompt("q", "a", "ref") is not None


# --- 存储：一个样本多条 judge 结论 ---


def test_verdicts_are_stored_per_metric(db, eval_id):
    """同一样本的不同指标独立保存。"""
    for name, score in (("faithfulness", 0.5), ("answer_relevancy", 1.0)):
        eval_store.record_judge_verdict(
            db,
            eval_id,
            sample_id="s1",
            metric=name,
            score=score,
            failure_kind=None,
            detail={"raw_response": f"raw-{name}"},
        )
    assert eval_store.judged_sample_ids(db, eval_id, "faithfulness") == {"s1"}
    assert eval_store.judged_sample_ids(db, eval_id, "context_relevancy") == set()
    assert eval_store.judged_sample_ids(db, eval_id) == {"s1"}

    verdicts = {v["metric"]: v["score"] for v in eval_store.judge_verdicts(db, eval_id)}
    assert verdicts == {"faithfulness": 0.5, "answer_relevancy": 1.0}
    assert eval_store.judge_verdicts(db, eval_id)[0]["detail"]["raw_response"]
    assert eval_store.judge_verdicts(db, eval_id, include_detail=False)[0]["detail"] is None


@pytest.mark.parametrize("status", [200, 400], ids=["success", "failure"])
def test_judge_reply_carries_latency(status):
    import httpx

    from akasha_benchmark.judge import JudgeClient, JudgeProvider

    provider = JudgeProvider(base_url="https://x/v1", model="m", api_key="k")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, json={"choices": [{"message": {"content": "{}"}}]})
    )
    with JudgeClient(provider, client=httpx.Client(transport=transport)) as client:
        reply = client.complete("s", "u")
    assert (reply.failure_kind is None) == (status == 200)
    assert reply.latency_ms is not None and reply.latency_ms >= 0


def test_judge_summary_separates_scores_failures_and_skipped_calls(db, eval_id):
    for sample_id, metric, score, failure, latency in (
        ("s1", "faithfulness", 1.0, None, 2000),
        ("s2", "faithfulness", 0.5, None, 0),
        ("s3", "faithfulness", None, None, None),
        ("s1", "answer_relevancy", None, "rate_limit", 4000),
    ):
        eval_store.record_judge_verdict(
            db, eval_id, sample_id=sample_id, metric=metric, score=score,
            failure_kind=failure, latency_ms=latency, detail=None,
        )

    summary = eval_store.judge_summary(db, eval_id)
    assert summary == {
        "total": 4, "scored": 2, "failed": 1, "mean": 0.75,
        "latency_mean": 2000, "failure_rate": 0.25,
        "failures_by_kind": {"rate_limit": 1},
    }
    faithfulness = eval_store.judge_summary(db, eval_id, "faithfulness")
    assert faithfulness["total"] == 3
    assert faithfulness["failure_rate"] == 0.0
    assert faithfulness["latency_mean"] == 1000
    assert eval_store.judge_summary(db, eval_id, "answer_relevancy")["failure_rate"] == 1.0
