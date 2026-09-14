"""阶段：归一化验收、抽样不变量、指标计算与省略口径。"""

from __future__ import annotations

import threading

import pytest

from akasha_benchmark.datasets import DataDependency, get_adapter
from akasha_benchmark.metrics import registry
from akasha_benchmark.stages import clean_params, compile, evaluate, normalize
from akasha_benchmark.store import (
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


# ------------------------------------------------------------ 归一化


def test_normalize_writes_samples_and_corpus(normalized):
    samples = data_store.samples_of(normalized, "hotpotqa")
    assert len(samples) == 2
    assert samples[0]["gold_doc_ids"], "hotpotqa 应当有 gold 标注"
    assert len(data_store.corpus_of(normalized, "hotpotqa")) == 4


def test_validate_catches_dangling_gold(normalized):
    """gold 指向不存在的 doc_id 不会让任何阶段报错，只会让指标永远差一截。"""
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


# ------------------------------------------------------------ 编译（抽样部分）


def test_subset_covers_every_gold(normalized):
    """先 QA 后 corpus：所选样本的 gold 必须全在子集语料内，否则 Recall 上限不是 1。"""
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
    """同 seed 抽同一批 —— 「同子集换 embedding」的对照实验靠这一条。"""

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


def test_markdown_uses_heading_for_title(normalized):
    """heading 承担 title，文件名承担 doc_id，两者独立，所以重复 title 不影响身份。"""
    markdown = compile.markdown_of(normalized, "hotpotqa", "0")
    assert markdown.startswith("# Rita Moreno")


def test_unsafe_doc_id_is_rejected():
    with pytest.raises(ValueError):
        compile._safe_doc_id("../escape")


# ------------------------------------------------------------ 评测


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
        score_threshold=None,
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
    """no_match 无条件返回空 retrievedSources —— 全样本与 knowledge 切片的差值就是它。"""
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
    # knowledge 切片里一条样本都没有，所以那一档没有指标行 —— 前端显示「—」
    # 而不是 0，两者的区别正是「这一档没有样本」与「这一档得分为 0」。
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
    # 明细仍然保留全部链路 —— 那是归因要读的东西，与「这一轮报哪些指标」是两件事。
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


def test_omitted_metrics_are_not_faked_as_zero():
    """narrativeqa 没有 gold 标注，整族检索指标必须省略而不是记 0。"""
    adapter = get_adapter("narrativeqa")
    assert DataDependency.GOLD_DOCS not in adapter.provides
    omitted = {d.name for d in registry.omitted(adapter.provides)}
    assert {"recall", "ndcg", "citation_recall"} <= omitted
    # faithfulness 的依赖是空集，所以它对 narrativeqa 仍然成立。
    assert "faithfulness" not in omitted


def test_resolve_metrics_rejects_unknown():
    with pytest.raises(ValueError, match="未知指标"):
        evaluate.resolve_metrics(["recall", "bogus"])
    assert evaluate.resolve_metrics([]) == sorted(registry.METRIC_REGISTRY)


# ------------------------------------------------------------ 参数白名单


def test_clean_params_drops_undeclared_keys():
    """参数经 HTTP 进来，不过滤等于让请求体决定阶段代码看到什么。"""
    cleaned = clean_params("query", {"compile_id": "3", "retry_failed": True, "evil": "x"})
    assert cleaned == {"compile_id": 3, "retry_failed": True}


def test_clean_params_rejects_wrong_types():
    with pytest.raises(ValueError):
        clean_params("query", {"compile_id": "not-a-number"})
    with pytest.raises(ValueError):
        clean_params("normalize", {"datasets": "hotpotqa"})
    with pytest.raises(ValueError):
        clean_params("bogus-stage", {})


def test_clean_params_coerces_ks_to_int():
    assert clean_params("evaluate", {"ks": ["2", "5"]}) == {"ks": [2, 5]}


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

    class NoCalls:
        def __init__(self, provider):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def complete(self, *args):
            pytest.fail("续跑不应重复调用已完成的 Judge")

    monkeypatch.setattr(evaluate, "JudgeClient", NoCalls)
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
