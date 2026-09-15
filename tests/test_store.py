"""存储层：schema、级联清理、续跑依据、judge 汇总口径。"""

from __future__ import annotations

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


def test_foreign_keys_cascade(db, sample_dataset, compile_id, query_id, eval_id):
    """删除编译时级联删除下游记录与产物。"""
    compile_store.replace_compile_subset(
        db, compile_id, "d", ["d:1"], [{"doc_id": "0", "is_gold": True}]
    )
    attribution_id = attribution_store.create_attribution_run(
        db, name="a", eval_id=eval_id, metric="recall@2", sample_limit=1, provider_id=None
    )
    db.commit()

    compile_store.delete_compile_run(db, compile_id)
    db.commit()

    assert query_store.get_query_run(db, query_id) is None
    assert eval_store.get_eval_run(db, eval_id) is None
    assert attribution_store.get_attribution_run(db, attribution_id) is None
    assert compile_store.compile_docs(db, compile_id) == []


def test_dataset_delete_refuses_when_compiled(db, sample_dataset, compile_id):
    """禁止删除已被编译引用的数据集。"""
    compile_store.replace_compile_subset(db, compile_id, "d", ["d:1"], [])
    db.commit()

    with pytest.raises(ValueError, match="编译层"):
        data_store.delete_dataset(db, "d")


def test_pending_query_samples_drives_resume(db, sample_dataset, query_id):
    """续跑仅处理固化选择中尚无响应的样本。"""
    query_store.freeze_query_samples(
        db, query_id, [{"sample_id": "d:1", "dataset": "d"}, {"sample_id": "d:2", "dataset": "d"}]
    )
    db.commit()
    assert len(query_store.pending_query_samples(db, query_id)) == 2

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
    db.commit()
    pending = query_store.pending_query_samples(db, query_id)
    assert [p["sample_id"] for p in pending] == ["d:2"]

    # 固化选择是幂等的：再冻结一次不会让待办回退。
    query_store.freeze_query_samples(db, query_id, [{"sample_id": "d:1", "dataset": "d"}])
    db.commit()
    assert len(query_store.pending_query_samples(db, query_id)) == 1


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
    db.commit()

    assert query_store.delete_failed_responses(db, query_id) == 2
    db.commit()
    assert [r["sample_id"] for r in query_store.responses_of(db, query_id)] == ["a"]


def test_judge_summary_excludes_failures_from_mean(db, eval_id):
    """失败条目不计入评分均值。"""
    eval_store.record_judge_verdict(
        db, eval_id, sample_id="a", score=1.0, failure_kind=None, detail=None
    )
    eval_store.record_judge_verdict(
        db, eval_id, sample_id="b", score=0.5, failure_kind=None, detail=None
    )
    eval_store.record_judge_verdict(
        db, eval_id, sample_id="c", score=None, failure_kind="rate_limit", detail=None
    )
    db.commit()

    summary = eval_store.judge_summary(db, eval_id)
    assert summary["mean"] == pytest.approx(0.75)
    assert summary["scored"] == 2
    assert summary["failure_rate"] == pytest.approx(1 / 3)
    assert summary["failures_by_kind"] == {"rate_limit": 1}


def test_compile_ready_requires_quality_gate(db, compile_id):
    """编译成功且质量检查通过后才允许查询。"""
    compile_store.replace_compile_subset(
        db, compile_id, "d", [], [{"doc_id": "0", "is_gold": True}]
    )
    db.commit()
    assert compile_store.compile_ready(db, compile_id)["ready"] is False

    compile_store.record_page(db, compile_id, "d", "0", page_id="p0", error=None)
    compile_store.update_compile_run(
        db,
        compile_id,
        space_id="s",
        status=run_store.STATUS_SUCCEEDED,
        quality_json='{"passed": false}',
    )
    db.commit()
    assert compile_store.compile_ready(db, compile_id)["ready"] is False

    compile_store.update_compile_run(db, compile_id, quality_json='{"passed": true}')
    db.commit()
    assert compile_store.compile_ready(db, compile_id)["ready"] is True


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
    config_store.update_connection(db, base_url="http://a", email="e@x", password="")
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
