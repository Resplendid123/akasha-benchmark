"""阶段：归一化验收、抽样不变量、指标计算与省略口径。"""

from __future__ import annotations

import threading

import pytest

from akasha_benchmark.datasets import DataDependency, get_adapter
from akasha_benchmark.metrics import registry
from akasha_benchmark.stages import attribute, clean_params, compile, evaluate, normalize
from akasha_benchmark.store import (
    attribution_store,
    compile_store,
    data_store,
    eval_store,
    query_store,
    run_store,
)
from akasha_benchmark.task import TaskContext


def context(connection, params=None) -> TaskContext:
    return TaskContext(
        task_id=1,
        stage="test",
        params=params or {},
        connection=connection,
        pause_event=threading.Event(),
    )


def _task_row(connection) -> None:
    connection.execute(
        "INSERT INTO task (id, stage, status, params_json, created_at) "
        "VALUES (1, 'test', 'running', '{}', 'now')"
    )
    connection.commit()




def test_normalize_writes_samples_and_corpus(normalized):
    samples = data_store.samples_of(normalized, "hotpotqa")
    assert len(samples) == 2
    assert samples[0]["gold_doc_ids"], "hotpotqa 应当有 gold 标注"
    assert len(data_store.corpus_of(normalized, "hotpotqa")) == 4


def test_validate_catches_dangling_gold(normalized):
    assert normalize.validate_dataset(normalized, "hotpotqa") == []

    normalized.execute("DELETE FROM corpus_doc WHERE dataset='hotpotqa' AND doc_id='0'")
    normalized.commit()
    problems = normalize.validate_dataset(normalized, "hotpotqa")
    assert any("gold" in p for p in problems)


def test_normalize_rejects_duplicate_sample_ids(db, dataset_dir, monkeypatch):
    import json

    from akasha_benchmark.datasets.hotpotqa import HotpotQAAdapter

    monkeypatch.setattr(HotpotQAAdapter, "expected_qa_rows", lambda self: None)
    rows = json.loads((dataset_dir / "hotpotqa.json").read_text(encoding="utf-8"))
    rows.append(rows[0])
    (dataset_dir / "hotpotqa.json").write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(ValueError, match="重复"):
        normalize.normalize_dataset(db, "hotpotqa", None, dataset_dir)




def test_subset_covers_every_gold(normalized):
    compile_id = compile_store.create_compile_run(
        normalized, run_id="r", datasets=["hotpotqa"], seed=7, qa_limit=2, negatives_ratio=1.0
    )
    normalized.commit()
    stats = compile.build_subset(
        normalized, compile_id, "hotpotqa", seed=7, qa_limit=2, negatives_ratio=1.0
    )

    docs = {d["doc_id"] for d in compile_store.compile_docs(normalized, compile_id)}
    samples = compile_store.compile_samples(normalized, compile_id)
    assert stats["samples"] == len(samples) == 2
    for sample in samples:
        assert set(sample["gold_doc_ids"]) <= docs

    gold = {d["doc_id"] for d in compile_store.compile_docs(normalized, compile_id) if d["is_gold"]}
    assert stats["gold"] == len(gold)
    assert stats["negatives"] == len(docs - gold)


def test_subset_is_seed_stable(normalized):

    def docs_for(seed: int, run_id: str) -> set[str]:
        compile_id = compile_store.create_compile_run(
            normalized,
            run_id=run_id,
            datasets=["hotpotqa"],
            seed=seed,
            qa_limit=1,
            negatives_ratio=1.0,
        )
        normalized.commit()
        compile.build_subset(
            normalized, compile_id, "hotpotqa", seed=seed, qa_limit=1, negatives_ratio=1.0
        )
        return {d["doc_id"] for d in compile_store.compile_docs(normalized, compile_id)}

    assert docs_for(7, "a") == docs_for(7, "b")


def test_subset_negatives_ratio_zero_keeps_only_gold(normalized):
    compile_id = compile_store.create_compile_run(
        normalized, run_id="r", datasets=["hotpotqa"], seed=1, qa_limit=2, negatives_ratio=0.0
    )
    normalized.commit()
    stats = compile.build_subset(
        normalized, compile_id, "hotpotqa", seed=1, qa_limit=2, negatives_ratio=0.0
    )
    assert stats["negatives"] == 0
    assert all(d["is_gold"] for d in compile_store.compile_docs(normalized, compile_id))


def test_subset_can_select_one_exact_sample(normalized):
    compile_id = compile_store.create_compile_run(
        normalized, run_id="exact", datasets=["hotpotqa"], seed=1, qa_limit=1,
        negatives_ratio=1.0,
    )
    target = data_store.samples_of(normalized, "hotpotqa")[1]

    stats = compile.build_subset(
        normalized,
        compile_id,
        "hotpotqa",
        seed=1,
        qa_limit=1,
        negatives_ratio=1.0,
        sample_ids=[target["sample_id"]],
    )

    assert stats["samples"] == 1
    assert [row["sample_id"] for row in compile_store.compile_samples(normalized, compile_id)] == [
        target["sample_id"]
    ]
    docs = {row["doc_id"] for row in compile_store.compile_docs(normalized, compile_id)}
    assert set(target["gold_doc_ids"]) <= docs


def test_full_corpus_keeps_every_document(normalized):
    compile_id = compile_store.create_compile_run(
        normalized, run_id="r", datasets=["hotpotqa"], seed=1, qa_limit=1, negatives_ratio=1.0
    )
    normalized.commit()
    stats = compile.build_subset(
        normalized,
        compile_id,
        "hotpotqa",
        seed=1,
        qa_limit=1,
        negatives_ratio=1.0,
        full_corpus=True,
    )

    docs = compile_store.compile_docs(normalized, compile_id)
    assert {doc["doc_id"] for doc in docs} == {"0", "1", "2", "3"}
    assert all(doc["title"] for doc in docs)
    assert stats["docs"] == 4
    assert stats["gold"] + stats["negatives"] == 4


def test_markdown_uses_heading_for_title(normalized):
    markdown = compile.markdown_of(normalized, "hotpotqa", "0")
    assert markdown.startswith("# Rita Moreno")


def test_unsafe_doc_id_is_rejected():
    with pytest.raises(ValueError):
        compile._safe_doc_id("../escape")


def test_image_only_document_has_no_indexable_text():
    assert not compile._has_indexable_text("# 标题\n\n![image](files/example.png)")
    assert compile._has_indexable_text("# 标题\n\n这里有可检索的正文。")




def _fixture_chain(connection, response: dict) -> tuple[int, int, int]:
    compile_id = compile_store.create_compile_run(
        connection, run_id="r", datasets=["hotpotqa"], seed=1, qa_limit=2, negatives_ratio=1.0
    )
    connection.commit()
    compile.build_subset(
        connection, compile_id, "hotpotqa", seed=1, qa_limit=2, negatives_ratio=1.0
    )
    for doc in compile_store.compile_docs(connection, compile_id):
        compile_store.record_page(
            connection,
            compile_id,
            doc["dataset"],
            doc["doc_id"],
            page_id=f"p{doc['doc_id']}",
            error=None,
        )
    compile_store.update_compile_run(
        connection, compile_id, space_id="s", status=run_store.STATUS_SUCCEEDED
    )
    query_id = query_store.create_query_run(
        connection,
        name="q",
        compile_id=compile_id,
        concurrency=1,
        model_configs={},
    )
    samples = compile_store.compile_samples(connection, compile_id)
    for sample in samples:
        query_store.record_response(
            connection,
            query_id,
            sample_id=sample["sample_id"],
            dataset="hotpotqa",
            question=sample["question"],
            http_status=200,
            latency_ms=100,
            error=None,
            response={
                **response,
                "retrievedSources": [{"sourcePageId": f"p{doc}"} for doc in sample["gold_doc_ids"]]
                if response.get("answerMode") == "knowledge"
                else [],
            },
        )
    run_store.set_run_status(connection, "query", query_id, run_store.STATUS_SUCCEEDED)
    eval_id = eval_store.create_eval_run(
        connection,
        name="e",
        query_id=query_id,
        ks=[2],
        metrics=["recall", "hit", "em", "f1"],
        judge_provider_id=None,
    )
    connection.commit()
    return compile_id, query_id, eval_id


def test_evaluate_scores_perfect_retrieval(normalized):
    compile_id, query_id, eval_id = _fixture_chain(
        normalized, {"answerMode": "knowledge", "answer": "Rita Moreno", "citations": []}
    )
    summary = evaluate.evaluate_dataset(
        normalized,
        eval_id,
        query_id,
        compile_id,
        "hotpotqa",
        (2,),
        frozenset({"recall", "hit", "em", "f1"}),
    )
    assert summary["responses_evaluated"] == 2
    assert summary["http_failures"] == 0
    assert summary["overall"]["recall@2"] == pytest.approx(1.0)
    assert summary["overall"]["hit@2"] == pytest.approx(1.0)


def test_evaluate_separates_fallback_from_retrieval_failure(normalized):
    compile_id, query_id, eval_id = _fixture_chain(
        normalized, {"answerMode": "no_match", "answer": "", "citations": []}
    )
    evaluate.evaluate_dataset(
        normalized, eval_id, query_id, compile_id, "hotpotqa", (2,), frozenset({"recall"})
    )
    summaries = {
        (row["scope"], row["metric"]): row
        for row in eval_store.metric_summaries(normalized, eval_id)
    }
    assert summaries[("overall", "recall@2")]["value"] == pytest.approx(0.0)
    # 空切片不生成指标行。
    assert ("knowledge_only", "recall@2") not in summaries
    modes = eval_store.dataset_evals(normalized, eval_id)[0]["answer_modes"]
    assert modes == {"no_match": pytest.approx(1.0)}


def test_evaluate_filters_to_selected_metrics(normalized):
    compile_id, query_id, eval_id = _fixture_chain(
        normalized, {"answerMode": "knowledge", "answer": "Rita Moreno", "citations": []}
    )
    evaluate.evaluate_dataset(
        normalized, eval_id, query_id, compile_id, "hotpotqa", (2,), frozenset({"em"})
    )
    sample = eval_store.sample_evals(normalized, eval_id)[0]
    metrics = eval_store.sample_metrics_of(normalized, eval_id, sample["sample_id"])
    assert set(metrics) == {"em"}
    # 明细保留完整链路供归因使用。
    assert "retrieval" in sample["detail"]


def test_evaluate_rejects_rebuilt_subset(normalized):
    compile_id, query_id, eval_id = _fixture_chain(
        normalized, {"answerMode": "knowledge", "answer": "x", "citations": []}
    )
    normalized.execute("UPDATE sample SET question = 'changed' WHERE sample_id LIKE 'hotpotqa%'")
    normalized.commit()
    with pytest.raises(ValueError, match="重建"):
        evaluate.evaluate_dataset(
            normalized, eval_id, query_id, compile_id, "hotpotqa", (2,), frozenset({"recall"})
        )


def test_attribution_processes_all_eval_samples_without_metric_selection(normalized):
    compile_id, _, eval_id = _fixture_chain(
        normalized, {"answerMode": "knowledge", "answer": "Rita Moreno"}
    )
    samples = compile_store.compile_samples(normalized, compile_id)
    for sample in samples:
        eval_store.record_sample_eval(
            normalized,
            eval_id,
            sample_id=sample["sample_id"],
            dataset=sample["dataset"],
            answer_mode="knowledge",
            http_status=200,
            answer="Rita Moreno",
            detail={"question": sample["question"], "gold_doc_ids": sample["gold_doc_ids"]},
            metrics={},
        )
    normalized.commit()
    _task_row(normalized)
    attribute.run(
        context(
            normalized,
            {
                "eval_id": eval_id,
                "metric": "missing_metric",
                "sample_limit": 1,
                "use_model": False,
            },
        )
    )

    run = attribution_store.list_attribution_runs(normalized, eval_id)[0]
    assert "metric" not in run
    assert "sample_limit" not in run
    assert len(attribution_store.attribution_results(normalized, int(run["id"]))) == len(samples)


def test_attribution_model_writes_one_overall_report(normalized, monkeypatch):
    from akasha_benchmark.judge.client import JudgeReply

    compile_id, query_id, eval_id = _fixture_chain(
        normalized, {"answerMode": "general", "answer": "A general answer", "citations": []}
    )
    evaluate.evaluate_dataset(
        normalized,
        eval_id,
        query_id,
        compile_id,
        "hotpotqa",
        (2,),
        frozenset({"recall", "hit", "em", "f1"}),
    )
    _task_row(normalized)
    provider = type("Provider", (), {"provider_id": None})()
    monkeypatch.setattr(attribute, "resolve_provider", lambda *args: provider)
    calls: list[list[tuple[str, str]]] = []

    def fake_complete_many(_provider, prompts, concurrency, *, max_tokens):
        calls.append(prompts)
        assert concurrency == 1
        assert max_tokens == 4096
        user_prompt = prompts[0][1]
        assert "rule_root_cause_counts" not in user_prompt
        assert "representative_rule_evidence" not in user_prompt
        assert "root_cause" not in user_prompt
        assert "sample_id" not in user_prompt
        assert '"general_answer_examples":[' in user_prompt
        assert '"answer":"A general answer"' in user_prompt
        assert user_prompt.count('"answer":"A general answer"') == 2
        return [
            JudgeReply(
                content='{"report":"整体表现\\n潜在原因\\n验证建议"}',
                failure_kind=None,
                raw=None,
                status=200,
                latency_ms=123,
            )
        ]

    monkeypatch.setattr(attribute, "complete_many", fake_complete_many)
    attribute.run(context(normalized, {"eval_id": eval_id, "use_model": True}))

    run = attribution_store.list_attribution_runs(normalized, eval_id)[0]
    results = attribution_store.attribution_results(normalized, int(run["id"]))
    assert len(calls) == 1
    assert len(calls[0]) == 1
    assert "metric_summaries" in calls[0][0][1]
    assert run["report"] == "整体表现\n潜在原因\n验证建议"
    assert run["report_error"] is None
    assert run["report_latency_ms"] == 123
    assert all(set(row) >= {"root_cause", "evidence"} for row in results)


def test_omitted_metrics_are_not_faked_as_zero():
    adapter = get_adapter("narrativeqa")
    assert DataDependency.GOLD_DOCS not in adapter.provides
    omitted = {d.name for d in registry.omitted(adapter.provides)}
    assert {"recall", "ndcg", "citation_recall"} <= omitted
    assert "faithfulness" not in omitted


def test_resolve_metrics_rejects_unknown():
    with pytest.raises(ValueError, match="未知指标"):
        evaluate.resolve_metrics(["recall", "bogus"])
    assert evaluate.resolve_metrics([]) == sorted(registry.METRIC_REGISTRY)




@pytest.mark.parametrize(
    "stage,payload,expected",
    [
        ("query", {"compile_id": "3", "retry_failed": True, "evil": "x"}, {"compile_id": 3, "retry_failed": True}),
        ("attribute", {"eval_id": 3, "metric": "em", "sample_limit": 1}, {"eval_id": 3}),
    ],
)
def test_clean_params_drops_undeclared_keys(stage, payload, expected):
    assert clean_params(stage, payload) == expected


def test_clean_params_rejects_wrong_types():
    with pytest.raises(ValueError):
        clean_params("query", {"compile_id": "not-a-number"})
    with pytest.raises(ValueError):
        clean_params("normalize", {"datasets": "hotpotqa"})
    with pytest.raises(ValueError):
        clean_params("bogus-stage", {})


@pytest.mark.parametrize(
    "payload,expected",
    [({"ks": ["2", "5"]}, {"ks": [2, 5]}), ({"concurrency": "4"}, {"concurrency": 4})],
)
def test_clean_params_coerces_evaluate_values(payload, expected):
    assert clean_params("evaluate", payload) == expected


def test_evaluate_resume_restores_judge_scores_and_summary(normalized, monkeypatch):
    from akasha_benchmark.judge.client import JudgeProvider
    from akasha_benchmark.store import task_store
    from akasha_benchmark.task import execute

    compile_id, query_id, eval_id = _fixture_chain(
        normalized, {"answerMode": "knowledge", "answer": "a"}
    )
    for sample in compile_store.compile_samples(normalized, compile_id):
        eval_store.record_judge_verdict(
            normalized,
            eval_id,
            sample_id=sample["sample_id"],
            metric="faithfulness",
            score=0.5,
            failure_kind=None,
            detail=None,
        )
    params = {"query_id": query_id, "name": "e", "metrics": ["em", "faithfulness"], "ks": [2]}
    task_id = task_store.create_task(normalized, stage="evaluate", params=params)
    task_store.set_task_target(normalized, task_id, "eval", eval_id)
    normalized.commit()
    monkeypatch.setattr(
        evaluate, "resolve_provider", lambda *args: JudgeProvider("https://x", "m", "k")
    )

    def no_calls(*args, **kwargs):
        pytest.fail("续跑不应重复调用已完成的 Judge")

    monkeypatch.setattr(evaluate, "complete_many", no_calls)
    ctx = TaskContext(
        task_id=task_id,
        stage="evaluate",
        params=params,
        connection=normalized,
        pause_event=threading.Event(),
    )
    execute(evaluate.run, ctx)
    for sample in eval_store.sample_evals(normalized, eval_id):
        assert (
            eval_store.sample_metrics_of(normalized, eval_id, sample["sample_id"])["faithfulness"]
            == 0.5
        )
    summaries = [
        r for r in eval_store.metric_summaries(normalized, eval_id) if r["scope"] == "judge"
    ]
    assert len(summaries) == 1
    assert summaries[0]["value"] == 0.5
    assert summaries[0]["sample_count"] == 2


def test_judge_runs_all_metrics_for_each_sample_before_next(normalized, monkeypatch):
    from akasha_benchmark.judge.client import JudgeProvider, JudgeReply
    from akasha_benchmark.store import task_store

    compile_id, query_id, eval_id = _fixture_chain(
        normalized, {"answerMode": "knowledge", "answer": "a"}
    )
    evaluate.evaluate_dataset(
        normalized,
        eval_id,
        query_id,
        compile_id,
        "hotpotqa",
        (2,),
        frozenset({"em"}),
    )
    task_id = task_store.create_task(normalized, stage="evaluate", params={})
    normalized.commit()
    ctx = TaskContext(
        task_id=task_id,
        stage="evaluate",
        params={},
        connection=normalized,
        pause_event=threading.Event(),
    )
    calls: list[list[tuple[str, str]]] = []

    def fake_task(metric, question, answer, reference, body):
        return ((metric, question), lambda payload: (1.0, {}))

    def fake_answer_relevancy(connection, question, answer, model_snapshot):
        return (("answer_relevancy", question), lambda payload: (1.0, {}))

    def fake_complete_many(provider, prompts, concurrency):
        calls.append(list(prompts))
        return [
            JudgeReply(content="{}", failure_kind=None, raw="SECRET_RAW", status=200)
            for _ in prompts
        ]

    monkeypatch.setattr(evaluate, "_judge_task", fake_task)
    monkeypatch.setattr(evaluate, "_answer_relevancy_task", fake_answer_relevancy)
    monkeypatch.setattr(evaluate, "complete_many", fake_complete_many)
    provider = JudgeProvider("https://x", "m", "k")

    evaluate._judge(
        ctx,
        eval_id,
        query_id,
        provider,
        ["faithfulness", "answer_relevancy"],
        4,
    )

    rows = eval_store.sample_evals(normalized, eval_id)
    assert calls == [
        [("faithfulness", rows[0]["detail"]["question"]), ("answer_relevancy", rows[0]["detail"]["question"])],
        [("faithfulness", rows[1]["detail"]["question"]), ("answer_relevancy", rows[1]["detail"]["question"])],
    ]
    task = task_store.get_task(normalized, task_id)
    assert (task["progress_done"], task["progress_total"]) == (2, 2)
    assert all("SECRET_RAW" not in entry["message"] for entry in task_store.task_logs(normalized, task_id))


def test_judge_resume_retries_failed_verdicts(normalized, monkeypatch):
    from akasha_benchmark.judge.client import JudgeProvider, JudgeReply
    from akasha_benchmark.store import task_store

    compile_id, query_id, eval_id = _fixture_chain(
        normalized, {"answerMode": "knowledge", "answer": "a"}
    )
    evaluate.evaluate_dataset(
        normalized, eval_id, query_id, compile_id, "hotpotqa", (2,), frozenset({"em"})
    )
    rows = eval_store.sample_evals(normalized, eval_id)
    for row in rows:
        eval_store.record_judge_verdict(
            normalized,
            eval_id,
            sample_id=row["sample_id"],
            metric="answer_relevancy",
            score=None,
            failure_kind="parse_error",
            detail={"error": "invalid json"},
        )
    task_id = task_store.create_task(normalized, stage="evaluate", params={})
    normalized.commit()
    ctx = TaskContext(
        task_id=task_id, stage="evaluate", params={}, connection=normalized, pause_event=threading.Event()
    )
    calls = 0

    def fake_answer_relevancy(connection, question, answer, model_snapshot):
        return (("system", "user"), lambda payload: (1.0, {}))

    def fake_complete_many(provider, prompts, concurrency):
        nonlocal calls
        calls += len(prompts)
        return [JudgeReply("{}", None, "{}", 200) for _ in prompts]

    monkeypatch.setattr(evaluate, "_answer_relevancy_task", fake_answer_relevancy)
    monkeypatch.setattr(evaluate, "complete_many", fake_complete_many)
    evaluate._judge(
        ctx, eval_id, query_id, JudgeProvider("https://x", "m", "k"),
        ["answer_relevancy"], 1
    )

    assert calls == len(rows)
    assert all(
        verdict["failure_kind"] is None
        for verdict in eval_store.judge_verdicts(normalized, eval_id)
        if verdict["metric"] == "answer_relevancy"
    )
