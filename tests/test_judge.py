"""judge：失败分类、该条排除、失败率闸门、以及密钥绝不外泄。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from akasha_benchmark.judge import faithfulness
from akasha_benchmark.judge.client import (
    FAILURE_PARSE,
    FAILURE_RATE_LIMIT,
    FAILURE_REFUSAL,
    FAILURE_TIMEOUT,
    JudgeClient,
    JudgeConfigError,
    JudgeProvider,
    parse_json_object,
)
from akasha_benchmark.judge.run import METRIC, judge_sample
from akasha_benchmark.store import connect, identity, repo
from akasha_benchmark.store.migrate import migrate

PROVIDER = JudgeProvider(base_url="https://judge.test/v1", model="m1", api_key="sk-secret-value")


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_JUDGE_KEY", "sk-secret-value")


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("akasha_benchmark.judge.client.time.sleep", lambda _s: None)


def _client(handler) -> JudgeClient:
    return JudgeClient(PROVIDER, httpx.Client(transport=httpx.MockTransport(handler)))


def _chat(content: str, **choice: object) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": content}, **choice}]}
    )


# --- 凭据 -------------------------------------------------------------------


def test_missing_api_key_is_a_clear_error():
    """Akasha 的 apiKeySet 只是布尔量，不回传 key，所以平台必须自己配一份。"""
    with pytest.raises(JudgeConfigError, match="no api key configured"):
        JudgeProvider(base_url="https://judge.test/v1", model="m1").resolve_key()


def test_a_stored_key_is_used_as_is():
    assert PROVIDER.resolve_key() == "sk-secret-value"


def test_redacted_is_a_whitelist_and_never_leaks_the_key():
    """``redacted()`` 要白名单式。黑名单漏写一个字段就泄露密钥。"""
    view = PROVIDER.redacted()
    assert view["api_key_set"] is True
    assert "sk-secret-value" not in json.dumps(view)
    # 新增字段的默认行为必须是「不输出」，所以键集合是封闭的。
    assert set(view) == {
        "base_url",
        "model",
        "api_key_set",
        "temperature",
        "max_tokens",
    }


def test_provider_hash_excludes_the_api_key():
    """provider 身份只吃 base_url + model。"""
    left = identity.judge_hash(base_url=PROVIDER.base_url, model=PROVIDER.model)
    right = identity.judge_hash(base_url=PROVIDER.base_url, model=PROVIDER.model, params={})
    assert left == right
    assert "sk-secret-value" not in left


def test_api_key_is_sent_as_a_bearer_header():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return _chat('{"claims": []}')

    with _client(handler) as client:
        client.complete("sys", "user")
    assert seen["authorization"] == "Bearer sk-secret-value"


# --- 失败四分类 -------------------------------------------------------------


def test_rate_limit_after_retries_is_its_own_failure_kind(no_sleep: None):
    """限流必须单独记。混进质量分会让一次 429 风暴看起来像模型变笨。"""
    with _client(lambda _r: httpx.Response(429, text="slow down")) as client:
        reply = client.complete("sys", "user")
    assert reply.failure_kind == FAILURE_RATE_LIMIT
    assert reply.content is None


def test_timeout_is_classified_as_timeout(no_sleep: None):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with _client(handler) as client:
        reply = client.complete("sys", "user")
    assert reply.failure_kind == FAILURE_TIMEOUT


def test_refusal_is_classified_as_refusal():
    with _client(lambda _r: _chat("", finish_reason="content_filter")) as client:
        reply = client.complete("sys", "user")
    assert reply.failure_kind == FAILURE_REFUSAL


def test_unparseable_body_is_a_parse_error():
    with _client(lambda _r: httpx.Response(200, text="not json at all")) as client:
        reply = client.complete("sys", "user")
    assert reply.failure_kind == FAILURE_PARSE


def test_transient_5xx_is_retried_then_succeeds(no_sleep: None):
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="restarting")
        return _chat('{"claims": []}')

    with _client(handler) as client:
        reply = client.complete("sys", "user")
    assert reply.failure_kind is None
    assert calls["n"] == 3


# --- 输出解析 ---------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        '{"claims": []}',
        '```json\n{"claims": []}\n```',
        'Here you go:\n{"claims": []}\nHope that helps.',
    ],
)
def test_parse_json_object_tolerates_fences_and_prose(content: str):
    """模型常把 JSON 包在围栏里或加一句解释，这些都要能吃下。"""
    assert parse_json_object(content) == {"claims": []}


def test_parse_json_object_refuses_to_guess():
    with pytest.raises(ValueError):
        parse_json_object("no object here")


def test_unknown_verdict_raises_instead_of_being_treated_as_unsupported():
    """静默当成 unsupported 会把「prompt 不听话」伪装成「答案不忠实」。"""
    with pytest.raises(ValueError, match="unknown verdict"):
        faithfulness.parse_verdict({"claims": [{"claim": "x", "verdict": "maybe"}]})


def test_score_is_the_supported_share():
    score, reasoning = faithfulness.parse_verdict(
        {
            "claims": [
                {"claim": "a", "verdict": "supported"},
                {"claim": "b", "verdict": "unsupported"},
                {"claim": "c", "verdict": "contradicted"},
                {"claim": "d", "verdict": "supported"},
            ]
        }
    )
    assert score == pytest.approx(0.5)
    assert reasoning["supported"] == 2
    assert reasoning["unsupported"] == 1
    assert reasoning["contradicted"] == 1


def test_an_answer_with_no_claims_scores_none_not_zero_or_one():
    """拒答既不忠实也不不忠实 —— 这个指标在它上面无定义。"""
    assert faithfulness.score_claims([]) is None


# --- 上下文构造 -------------------------------------------------------------


def test_context_prefers_snippets_and_falls_back_to_titles():
    with_text = faithfulness.build_context(
        {"snippets": [{"title": "T", "text": "body text"}], "retrievedSources": []}
    )
    assert "body text" in with_text

    titles_only = faithfulness.build_context(
        {"snippets": [], "retrievedSources": [{"title": "OnlyTitle"}]}
    )
    assert "OnlyTitle" in titles_only


def test_empty_context_means_skip_not_zero():
    """no_match / general 会无条件清空 retrievedSources。此时应跳过，不记 0。"""
    assert faithfulness.build_prompt("q", "a", {"snippets": [], "retrievedSources": []}) is None


def test_judge_sample_skips_when_there_is_no_context():
    with _client(lambda _r: _chat('{"claims": []}')) as client:
        result = judge_sample(client, "q", "some answer", {"retrievedSources": []})
    assert result["skipped"] is True
    assert result["score"] is None
    assert result["failure_kind"] is None


# --- 汇总：该条排除，不记 0 -------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    path = tmp_path / "t.db"
    migrate(path, verbose=False)
    connection = connect(path)
    yield connection
    connection.close()


def _eval_layer(connection: sqlite3.Connection) -> int:
    repo.upsert_dataset(
        connection,
        name="hotpotqa",
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
        narrativeqa_docs=2,
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
    return repo.create_eval_layer(
        connection, query_layer_id=query_layer_id, label="E", config_hash="eh", ks=[10], metrics=[]
    )


def test_failures_are_excluded_from_the_mean_rather_than_scored_zero(db: sqlite3.Connection):
    """决策 13：失败该条排除，另叠失败率闸门。"""
    eval_id = _eval_layer(db)
    for sample_id, score, failure in (
        ("s1", 0.8, None),
        ("s2", 0.8, None),
        ("s3", None, FAILURE_RATE_LIMIT),
    ):
        repo.record_judge_verdict(
            db,
            eval_id,
            sample_id=sample_id,
            metric=METRIC,
            score=score,
            failure_kind=failure,
            reasoning=None,
            raw_response=None,
            provider_hash="ph",
            prompt_version="v1",
        )
    db.commit()

    summary = repo.judge_summary(db, eval_id, METRIC)
    assert summary["scored"] == 2
    assert summary["excluded"] == 1
    assert summary["mean"] == pytest.approx(0.8)
    assert summary["failure_rate"] == pytest.approx(1 / 3)
    assert summary["failures_by_kind"] == {FAILURE_RATE_LIMIT: 1}


def test_resume_reruns_failures_but_not_successes(db: sqlite3.Connection):
    """已成功的不重判（省钱），失败的留给下次重试。"""
    eval_id = _eval_layer(db)
    repo.record_judge_verdict(
        db,
        eval_id,
        sample_id="ok",
        metric=METRIC,
        score=1.0,
        failure_kind=None,
        reasoning=None,
        raw_response=None,
        provider_hash="ph",
        prompt_version="v1",
    )
    repo.record_judge_verdict(
        db,
        eval_id,
        sample_id="bad",
        metric=METRIC,
        score=None,
        failure_kind=FAILURE_TIMEOUT,
        reasoning=None,
        raw_response=None,
        provider_hash="ph",
        prompt_version="v1",
    )
    db.commit()
    assert repo.judged_sample_ids(db, eval_id, METRIC) == {"ok"}


def test_faithfulness_requires_no_annotations_so_it_covers_narrativeqa():
    """具体收获：judge 能填上 narrativeqa 那个洞。"""
    from akasha_benchmark.datasets import DataDependency, get_adapter
    from akasha_benchmark.metrics import registry

    narrativeqa = get_adapter("narrativeqa")
    assert DataDependency.GOLD_DOCS not in narrativeqa.provides
    # 检索族全部算不了。
    assert "recall" in {d.name for d in registry.omitted(narrativeqa.provides)}
    # 但 faithfulness 的依赖是空集，所以它可用。
    assert registry.METRIC_REGISTRY["faithfulness"].requires == frozenset()
    assert "faithfulness" in {d.name for d in registry.available(narrativeqa.provides)}
