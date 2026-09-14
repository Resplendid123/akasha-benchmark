"""存储层：schema、级联清理、续跑依据、judge 汇总口径。"""

from __future__ import annotations

import sqlite3

import pytest

from akasha_benchmark.store import config_store, connect, data_store, init_db, run_store


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


def test_legacy_db_is_moved_aside(tmp_path):
    """重构前的库 schema 与现在的对不上，就地建表会两套混在一起。"""
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE index_layer (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()

    archived = init_db(path)
    assert archived is not None and archived.is_file()
    connection = connect(path)
    try:
        tables = {
            r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        connection.close()
    assert "index_layer" not in tables
    assert "compile_run" in tables


def test_foreign_keys_cascade(db):
    """清理一层就是删主表那一行 —— 下游必须跟着走，否则留下孤儿数据。"""
    data_store.upsert_dataset(
        db, name="d", qa_sha256="a", qa_rows=1, corpus_sha256="b", corpus_rows=1
    )
    data_store.replace_samples(
        db,
        "d",
        [
            {
                "sample_id": "d:1",
                "dataset_sample_id": "1",
                "question": "q",
                "answers": ["a"],
                "gold_doc_ids": ["0"],
                "metadata": {},
            }
        ],
    )
    compile_id = run_store.create_compile_run(
        db, run_id="r", datasets=["d"], seed=1, qa_limit=1, negatives_ratio=1.0
    )
    run_store.replace_compile_subset(db, compile_id, "d", ["d:1"], [{"doc_id": "0", "is_gold": True}])
    query_id = run_store.create_query_run(
        db, name="q", compile_id=compile_id, score_threshold=None, concurrency=1, model_configs={}
    )
    eval_id = run_store.create_eval_run(
        db, name="e", query_id=query_id, ks=[2], metrics=["recall"], judge_provider_id=None
    )
    attribution_id = run_store.create_attribution_run(
        db, name="a", eval_id=eval_id, metric="recall@2", sample_limit=1, provider_id=None
    )
    db.commit()

    run_store.delete_compile_run(db, compile_id)
    db.commit()

    assert run_store.get_query_run(db, query_id) is None
    assert run_store.get_eval_run(db, eval_id) is None
    assert run_store.get_attribution_run(db, attribution_id) is None
    assert run_store.compile_docs(db, compile_id) == []


def test_dataset_delete_refuses_when_compiled(db):
    """归一化产物被编译层引用时不许删 —— 那会连带删掉编译与其下游。"""
    data_store.upsert_dataset(
        db, name="d", qa_sha256="a", qa_rows=1, corpus_sha256="b", corpus_rows=1
    )
    data_store.replace_samples(
        db,
        "d",
        [
            {
                "sample_id": "d:1",
                "dataset_sample_id": "1",
                "question": "q",
                "answers": ["a"],
                "gold_doc_ids": [],
                "metadata": {},
            }
        ],
    )
    compile_id = run_store.create_compile_run(
        db, run_id="r", datasets=["d"], seed=1, qa_limit=1, negatives_ratio=1.0
    )
    run_store.replace_compile_subset(db, compile_id, "d", ["d:1"], [])
    db.commit()

    with pytest.raises(ValueError, match="编译层"):
        data_store.delete_dataset(db, "d")


def test_pending_query_samples_drives_resume(db):
    """待办 = 固化选择 - 已落库响应。这就是暂停后继续不重复消耗的依据。"""
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
                "gold_doc_ids": [],
                "metadata": {},
            }
            for i in (1, 2)
        ],
    )
    compile_id = run_store.create_compile_run(
        db, run_id="r", datasets=["d"], seed=1, qa_limit=2, negatives_ratio=1.0
    )
    query_id = run_store.create_query_run(
        db, name="q", compile_id=compile_id, score_threshold=None, concurrency=1, model_configs={}
    )
    run_store.freeze_query_samples(
        db, query_id, [{"sample_id": "d:1", "dataset": "d"}, {"sample_id": "d:2", "dataset": "d"}]
    )
    db.commit()
    assert len(run_store.pending_query_samples(db, query_id)) == 2

    run_store.record_response(
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
    db.commit()
    pending = run_store.pending_query_samples(db, query_id)
    assert [p["sample_id"] for p in pending] == ["d:2"]

    # 固化选择是幂等的：再冻结一次不会让待办回退。
    run_store.freeze_query_samples(db, query_id, [{"sample_id": "d:1", "dataset": "d"}])
    db.commit()
    assert len(run_store.pending_query_samples(db, query_id)) == 1


def test_delete_failed_responses_keeps_successes(db):
    compile_id = run_store.create_compile_run(
        db, run_id="r", datasets=[], seed=1, qa_limit=1, negatives_ratio=1.0
    )
    query_id = run_store.create_query_run(
        db, name="q", compile_id=compile_id, score_threshold=None, concurrency=1, model_configs={}
    )
    for sample_id, status in (("a", 200), ("b", 500), ("c", 0)):
        run_store.record_response(
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
    db.commit()

    assert run_store.delete_failed_responses(db, query_id) == 2
    db.commit()
    assert [r["sample_id"] for r in run_store.responses_of(db, query_id)] == ["a"]


def test_judge_summary_excludes_failures_from_mean(db):
    """失败该条排除，不记 0 —— 记 0 会让限流伪装成质量差。"""
    compile_id = run_store.create_compile_run(
        db, run_id="r", datasets=[], seed=1, qa_limit=1, negatives_ratio=1.0
    )
    query_id = run_store.create_query_run(
        db, name="q", compile_id=compile_id, score_threshold=None, concurrency=1, model_configs={}
    )
    eval_id = run_store.create_eval_run(
        db, name="e", query_id=query_id, ks=[2], metrics=["faithfulness"], judge_provider_id=None
    )
    run_store.record_judge_verdict(db, eval_id, sample_id="a", score=1.0, failure_kind=None, detail=None)
    run_store.record_judge_verdict(db, eval_id, sample_id="b", score=0.5, failure_kind=None, detail=None)
    run_store.record_judge_verdict(
        db, eval_id, sample_id="c", score=None, failure_kind="rate_limit", detail=None
    )
    db.commit()

    summary = run_store.judge_summary(db, eval_id)
    assert summary["mean"] == pytest.approx(0.75)
    assert summary["scored"] == 2
    assert summary["failure_rate"] == pytest.approx(1 / 3)
    assert summary["failures_by_kind"] == {"rate_limit": 1}


def test_compile_ready_requires_quality_gate(db):
    """质量闸门取不到值时不算通过：空集合上的 all() 是 True，那会让半成品过闸。"""
    compile_id = run_store.create_compile_run(
        db, run_id="r", datasets=["d"], seed=1, qa_limit=1, negatives_ratio=1.0
    )
    run_store.replace_compile_subset(db, compile_id, "d", [], [{"doc_id": "0", "is_gold": True}])
    db.commit()
    assert run_store.compile_ready(db, compile_id)["ready"] is False

    run_store.record_page(db, compile_id, "d", "0", page_id="p0", error=None)
    run_store.update_compile_run(
        db,
        compile_id,
        space_id="s",
        status=run_store.STATUS_SUCCEEDED,
        quality_json='{"passed": false}',
    )
    db.commit()
    assert run_store.compile_ready(db, compile_id)["ready"] is False

    run_store.update_compile_run(db, compile_id, quality_json='{"passed": true}')
    db.commit()
    assert run_store.compile_ready(db, compile_id)["ready"] is True


def test_provider_api_key_roundtrip(db):
    provider_id = config_store.upsert_provider(
        db,
        role="judge",
        label="default",
        base_url="https://x/v1",
        model="m",
        api_key="secret",
    )
    db.commit()
    assert config_store.get_provider(db, provider_id)["api_key"] == "secret"

    with pytest.raises(ValueError):
        config_store.upsert_provider(
            db,
            role="bogus",
            label="x",
            base_url="u",
            model="m",
            api_key="",
        )


def test_provider_id_updates_in_place(db):
    provider_id = config_store.upsert_provider(
        db, role="judge", label="default", base_url="https://x/v1", model="m", api_key="k"
    )
    config_store.upsert_provider(
        db,
        role="judge",
        label="renamed",
        base_url="https://y/v1",
        model="m2",
        api_key="k",
        provider_id=provider_id,
    )
    db.commit()
    rows = config_store.list_providers(db, "judge")
    assert [(r["id"], r["label"], r["model"]) for r in rows] == [(provider_id, "renamed", "m2")]


def test_connection_updates_only_given_fields(db):
    config_store.update_connection(db, base_url="http://a", email="e@x")
    db.commit()
    row = config_store.get_connection_row(db)
    assert row["base_url"] == "http://a"
    assert row["password"] == ""

    config_store.update_connection(db, password="p")
    db.commit()
    row = config_store.get_connection_row(db)
    assert row["base_url"] == "http://a"
    assert row["password"] == "p"

    with pytest.raises(ValueError):
        config_store.update_connection(db, bogus="x")


# localhost 在 Windows 上先解析到 ::1，每个请求都要先等它被拒（实测 2s）。
def test_localhost_is_rewritten_to_ipv4():
    clean = config_store.sanitize_connection(
        {
            "base_url": "http://localhost:3000",
            "database_url": "postgresql://u:p@localhost:5432/db",
        }
    )
    assert clean["base_url"] == "http://127.0.0.1:3000"
    # 凭据与端口不能在改写中丢掉。
    assert clean["database_url"] == "postgresql://u:p@127.0.0.1:5432/db"

    # 显式写 IPv6 的是特意要 IPv6；别的主机名不动。
    untouched = config_store.sanitize_connection(
        {"base_url": "http://[::1]:3000", "database_url": "postgresql://localhostish/db"}
    )
    assert untouched["base_url"] == "http://[::1]:3000"
    assert untouched["database_url"] == "postgresql://localhostish/db"


def test_normalize_hosts_migrates_existing_row(db):
    # 绕过 sanitize 直接写入旧值，模拟本次改动之前存下的配置。
    config_store.update_connection(db, base_url="http://localhost:3000")
    db.commit()

    assert config_store.normalize_hosts(db) == ["base_url"]
    db.commit()
    assert config_store.get_connection_row(db)["base_url"] == "http://127.0.0.1:3000"
    # 已经是 IPv4 的不再改写，重复调用不产生变更。
    assert config_store.normalize_hosts(db) == []
