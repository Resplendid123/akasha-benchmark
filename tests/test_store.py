"""存储层：schema、级联清理、续跑依据、judge 汇总口径。"""

from __future__ import annotations

import sqlite3

import pytest

from akasha_benchmark.store import (
    attribution_store,
    compile_store,
    config_store,
    connect,
    data_store,
    eval_store,
    init_db,
    query_store,
    run_store,
    task_store,
)


@pytest.fixture
def sample_dataset(db):
    data_store.upsert_dataset(
        db, name="d", qa_sha256="a", qa_rows=2, corpus_sha256="b", corpus_rows=1
    )
    data_store.replace_samples(
        db,
        "d",
        [
            {
                "sample_id": f"d:{i}",
                "dataset_sample_id": str(i),
                "question": f"q{i}",
                "answers": ["a"],
                "gold_doc_ids": ["0"],
                "metadata": {},
            }
            for i in (1, 2)
        ],
    )


def test_init_db_is_idempotent(db_path):
    init_db(db_path)
    connection = connect(db_path)
    try:
        tables = {
            r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        connection.close()
    assert {"compile_run", "query_run", "eval_run", "attribution_run", "audit_log"} <= tables


@pytest.mark.usefixtures("sample_dataset")
def test_foreign_keys_cascade(db, compile_id, query_id, eval_id):
    compile_store.replace_compile_subset(
        db, compile_id, "d", ["d:1"], [{"doc_id": "0", "is_gold": True}]
    )
    attribution_id = attribution_store.create_attribution_run(
        db, name="a", eval_id=eval_id, report_provider_id=None
    )

    compile_store.delete_compile_run(db, compile_id)

    assert query_store.get_query_run(db, query_id) is None
    assert eval_store.get_eval_run(db, eval_id) is None
    assert attribution_store.get_attribution_run(db, attribution_id) is None
    assert compile_store.compile_docs(db, compile_id) == []


@pytest.mark.usefixtures("sample_dataset")
def test_dataset_delete_refuses_when_compiled(db, compile_id):
    compile_store.replace_compile_subset(db, compile_id, "d", ["d:1"], [])

    with pytest.raises(ValueError, match="编译层"):
        data_store.delete_dataset(db, "d")


@pytest.mark.usefixtures("sample_dataset")
def test_pending_query_samples_drives_resume(db, query_id):
    query_store.freeze_query_samples(
        db, query_id, [{"sample_id": "d:1"}, {"sample_id": "d:2"}]
    )
    assert query_store.query_samples(db, query_id) == [
        {"query_id": query_id, "sample_id": f"d:{i}", "dataset": "d"} for i in (1, 2)
    ]
    assert query_store.pending_query_samples(db, query_id) == [
        {"sample_id": f"d:{i}", "dataset": "d", "question": f"q{i}"} for i in (1, 2)
    ]

    query_store.record_response(
        db,
        query_id,
        sample_id="d:1",
        dataset="d",
        question="q1",
        http_status=200,
        latency_ms=10,
        error=None,
        response={"answerMode": "knowledge"},
    )
    pending = query_store.pending_query_samples(db, query_id)
    assert [p["sample_id"] for p in pending] == ["d:2"]

    query_store.freeze_query_samples(db, query_id, [{"sample_id": "d:1", "dataset": "d"}])
    assert len(query_store.pending_query_samples(db, query_id)) == 1
    with pytest.raises(sqlite3.IntegrityError):
        query_store.freeze_query_samples(db, query_id, [{"sample_id": "missing"}])


@pytest.mark.usefixtures("sample_dataset")
def test_compile_subset_replacement_is_scoped_to_dataset(db, compile_id):
    data_store.upsert_dataset(
        db, name="other", qa_sha256="a", qa_rows=1, corpus_sha256="b", corpus_rows=0
    )
    data_store.replace_samples(
        db,
        "other",
        [{**data_store.get_sample(db, "d:1"), "sample_id": "other:1"}],
    )
    compile_store.replace_compile_subset(db, compile_id, "d", ["d:1"], [])
    compile_store.replace_compile_subset(db, compile_id, "other", ["other:1"], [])
    compile_store.replace_compile_subset(db, compile_id, "d", ["d:2"], [])

    assert [
        s["sample_id"] for s in compile_store.compile_samples(db, compile_id, "d")
    ] == ["d:2"]
    assert [s["sample_id"] for s in compile_store.compile_samples(db, compile_id)] == [
        "d:2",
        "other:1",
    ]
    assert compile_store.compile_stats(db, compile_id) == {
        "d": {"dataset": "d", "samples": 1},
        "other": {"dataset": "other", "samples": 1},
    }


def test_clear_eval_cascades_metrics_only_for_selected_dataset(db, eval_id):
    for dataset in ("a", "b"):
        eval_store.record_sample_eval(
            db,
            eval_id,
            sample_id=dataset,
            dataset=dataset,
            answer_mode="knowledge",
            http_status=200,
            answer="answer",
            detail={},
            metrics={"em": 1.0},
        )
        eval_store.record_judge_verdict(
            db, eval_id, sample_id=dataset, score=0.5, failure_kind=None, detail=None
        )
    eval_store.restore_judge_metrics(db, eval_id, "faithfulness")
    assert [
        r["sample_id"]
        for r in eval_store.samples_ranked_by(db, eval_id, "em", dataset="a")
    ] == ["a"]

    eval_store.clear_eval_results(db, eval_id, "a")

    assert eval_store.sample_metrics_of(db, eval_id, "a") == {}
    assert eval_store.sample_metrics_of(db, eval_id, "b") == {
        "em": 1.0,
        "faithfulness": 0.5,
    }
    assert eval_store.judged_sample_ids(db, eval_id) == {"a", "b"}
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO sample_metric (eval_id, sample_id, metric, value) VALUES (?, ?, ?, ?)",
            (eval_id, "a", "em", 1.0),
        )
    assert not db.execute("PRAGMA foreign_key_check").fetchall()


@pytest.mark.parametrize(
    ("body", "mode"),
    [({"answerMode": "knowledge"}, "knowledge"), ({}, None), (None, None), ([], None), ("error", None)],
)
def test_response_mode_is_derived_from_current_body(db, query_id, body, mode):
    for response in ({"answerMode": "fallback"}, body):
        query_store.record_response(
            db,
            query_id,
            sample_id="a",
            dataset="d",
            question="q",
            http_status=200,
            latency_ms=0,
            error=None,
            response=response,
        )
    row = query_store.response_of(db, query_id, "a")
    assert row["response"] == body
    assert row["answer_mode"] == mode
    assert query_store.responses_of(db, query_id) == [row]


def test_delete_failed_responses_keeps_successes(db, query_id):
    for sample_id, status in (("a", 200), ("b", 500), ("c", 0)):
        query_store.record_response(
            db,
            query_id,
            sample_id=sample_id,
            dataset="d",
            question="q",
            http_status=status,
            latency_ms=None,
            error=None,
            response=None,
        )

    assert query_store.delete_failed_responses(db, query_id) == 2
    assert [r["sample_id"] for r in query_store.responses_of(db, query_id)] == ["a"]


def test_compile_ready_requires_quality_gate(db, compile_id):
    compile_store.replace_compile_subset(
        db, compile_id, "d", [], [{"doc_id": "0", "is_gold": True}]
    )
    assert compile_store.compile_ready(db, compile_id)["ready"] is False

    compile_store.record_page(db, compile_id, "d", "0", page_id="p0", error=None)
    compile_store.update_compile_run(
        db,
        compile_id,
        space_id="s",
        status=run_store.STATUS_SUCCEEDED,
        quality_json='{"passed": false}',
    )
    assert compile_store.compile_ready(db, compile_id)["ready"] is False

    compile_store.update_compile_run(db, compile_id, quality_json='{"passed": true}')
    assert compile_store.compile_ready(db, compile_id)["ready"] is True


def test_compile_ready_rejects_failed_compile_without_partial_success(db, compile_id):
    compile_store.replace_compile_subset(
        db, compile_id, "d", [], [{"doc_id": "0", "is_gold": True}]
    )
    compile_store.record_page(db, compile_id, "d", "0", page_id="p0", error=None)
    compile_store.update_compile_run(
        db,
        compile_id,
        space_id="s",
        status=run_store.STATUS_FAILED,
        quality_json=(
            '{"passed": false, "progress": '
            '{"expected": 1, "succeeded": 0, "failed": 1, "skipped": 0}}'
        ),
    )

    readiness = compile_store.compile_ready(db, compile_id)
    assert readiness["ready"] is False
    assert readiness["warnings"] == []


def test_provider_api_key_roundtrip(db):
    provider_id = config_store.upsert_model_provider(
        db,
        purpose="judge",
        label="default",
        base_url="https://x/v1",
        model="m",
        api_key="secret",
    )
    assert config_store.get_model_provider(db, provider_id)["api_key"] == "secret"

    with pytest.raises(ValueError):
        config_store.upsert_model_provider(
            db,
            purpose="bogus",
            label="x",
            base_url="u",
            model="m",
            api_key="",
        )


def test_providers_are_listed_by_most_recent_update(db):
    older = config_store.upsert_model_provider(
        db, purpose="judge", label="z-old", base_url="u", model="m1", api_key="k"
    )
    newer = config_store.upsert_model_provider(
        db, purpose="judge", label="a-new", base_url="u", model="m2", api_key="k"
    )
    db.execute("UPDATE model_provider SET updated_at = '2026-01-01T00:00:00Z' WHERE id = ?", (older,))
    db.execute("UPDATE model_provider SET updated_at = '2026-01-02T00:00:00Z' WHERE id = ?", (newer,))

    assert [row["id"] for row in config_store.list_model_providers(db, "judge")] == [newer, older]


def test_task_tree_preserves_one_to_many_run_branches(db, compile_id, query_id, eval_id):
    second_query = query_store.create_query_run(
        db,
        name="q2",
        compile_id=compile_id,
        concurrency=1,
        model_configs={},
    )
    second_eval = eval_store.create_eval_run(
        db,
        name="e2",
        query_id=query_id,
        ks=[2],
        metrics=["em"],
        judge_provider_id=None,
    )
    for evaluation, suffix in ((eval_id, "a1"), (eval_id, "a2"), (second_eval, "a3")):
        attribution_store.create_attribution_run(
            db, name=suffix, eval_id=evaluation, report_provider_id=None
        )

    compile_task = task_store.create_task(db, stage="compile", params={})
    task_store.set_task_target(db, compile_task, "compile", compile_id)
    query_task = task_store.create_task(db, stage="query", params={"compile_id": compile_id})
    task_store.set_task_target(db, query_task, "query", query_id)
    pending_eval = task_store.create_task(db, stage="evaluate", params={"query_id": second_query})
    unlinked = task_store.create_task(db, stage="normalize", params={})
    db.commit()

    tree = task_store.task_tree(db)
    root = tree["compiles"][0]
    assert root["id"] == compile_id
    assert [task["id"] for task in root["tasks"]] == [compile_task]
    assert {child["id"] for child in root["children"]} == {query_id, second_query}

    first_query = next(child for child in root["children"] if child["id"] == query_id)
    assert [task["id"] for task in first_query["tasks"]] == [query_task]
    assert {child["id"] for child in first_query["children"]} == {eval_id, second_eval}
    first_eval = next(child for child in first_query["children"] if child["id"] == eval_id)
    assert len(first_eval["children"]) == 2

    other_query = next(child for child in root["children"] if child["id"] == second_query)
    assert [task["id"] for task in other_query["pending_tasks"]] == [pending_eval]
    assert [task["id"] for task in tree["unlinked_tasks"]] == [unlinked]


def test_connection_rejects_unknown_fields(db):
    with pytest.raises(ValueError):
        config_store.update_connection(db, bogus="x")


def test_localhost_is_rewritten_to_ipv4():
    clean = config_store.sanitize_connection(
        {
            "base_url": "http://localhost:3000",
            "database_url": "postgresql://u:p@localhost:5432/db",
        }
    )
    assert clean["base_url"] == "http://127.0.0.1:3000"
    assert clean["database_url"] == "postgresql://u:p@127.0.0.1:5432/db"

    untouched = config_store.sanitize_connection(
        {"base_url": "http://[::1]:3000", "database_url": "postgresql://localhostish/db"}
    )
    assert untouched["base_url"] == "http://[::1]:3000"
    assert untouched["database_url"] == "postgresql://localhostish/db"
