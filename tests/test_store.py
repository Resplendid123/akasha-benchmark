"""数据层：迁移、身份哈希、以及几条「错了不报错」的性质。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from akasha_benchmark.store import connect, identity, repo
from akasha_benchmark.store.migrate import migrate
from akasha_benchmark.store.reindex import PROTECTED_TABLES, REBUILDABLE_TABLES

MODEL_CONFIGS = {
    "configs": [
        {
            "feature": "embedding",
            "provider": "openai-compatible",
            "model": "text-embedding-3-large",
            "baseUrl": "https://example.test/v1",
            "apiKeySet": True,
            "parameters": {"dimension": 1024},
        },
        {
            "feature": "compiler",
            "provider": "openai-compatible",
            "model": "gemini-3.6-flash",
            "baseUrl": "https://example.test/v1",
            "apiKeySet": True,
            "parameters": None,
        },
        {
            "feature": "answer",
            "provider": "openai-compatible",
            "model": "gemini-3.6-flash",
            "baseUrl": "https://example.test/v1",
            "apiKeySet": True,
            "parameters": None,
        },
    ]
}


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    path = tmp_path / "t.db"
    migrate(path, verbose=False)
    connection = connect(path)
    yield connection
    connection.close()


def _dataset(connection: sqlite3.Connection, name: str, provides: list[str]) -> None:
    repo.upsert_dataset(
        connection,
        name=name,
        adapter="A",
        adapter_version="1",
        provides=provides,
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


def _layer(connection: sqlite3.Connection, label: str = "L") -> int:
    return repo.create_index_layer(
        connection,
        label=label,
        subset_hash="sh",
        config_hash="h",
        seed=1,
        qa_limit=100,
        negatives_ratio=1.0,
        narrativeqa_docs=2,
    )


def _subset(connection: sqlite3.Connection, layer_id: int, dataset: str, doc_ids: list[str]) -> None:
    repo.upsert_index_layer_dataset(
        connection,
        layer_id,
        dataset,
        strategy="s",
        qa_count=0,
        corpus_count=len(doc_ids),
        gold_doc_count=0,
        negative_doc_count=0,
        strata={},
        normalized_qa_sha256="0" * 64,
        normalized_corpus_sha256="1" * 64,
    )
    repo.replace_subset(
        connection,
        layer_id,
        dataset,
        [],
        [
            {"doc_id": d, "md_text": f"# {d}\n\nbody\n", "md_sha256": f"{i}" * 64, "is_gold": False}
            for i, d in enumerate(doc_ids)
        ],
    )


# --- 迁移 -------------------------------------------------------------------


def test_first_migration_does_not_leave_an_empty_backup(tmp_path: Path):
    """空库上跑首个迁移不该留下 ``.pre-001``。"""
    path = tmp_path / "fresh.db"
    migrate(path, verbose=False)
    assert not list(tmp_path.glob("*.pre-*"))


def test_migration_refuses_to_run_when_an_applied_file_was_edited(tmp_path: Path):
    """已应用的迁移被改过就报错，不能带着分叉的 schema 继续跑。"""
    migrations = tmp_path / "m"
    migrations.mkdir()
    (migrations / "001_x.sql").write_text("CREATE TABLE a (id INTEGER);", encoding="utf-8")
    db_path = tmp_path / "d.db"
    migrate(db_path, migrations, verbose=False)

    (migrations / "001_x.sql").write_text(
        "CREATE TABLE a (id INTEGER, extra TEXT);", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="modified after it ran"):
        migrate(db_path, migrations, verbose=False)


def test_migration_runs_on_a_populated_database_and_backs_it_up(tmp_path: Path):
    """迁移必须能在**有数据**的库上跑，不能只在空库验证过。"""
    migrations = tmp_path / "m"
    migrations.mkdir()
    source = Path(__file__).resolve().parents[1] / "migrations" / "001_initial.sql"
    (migrations / "001_initial.sql").write_text(source.read_text(encoding="utf-8"), "utf-8")

    db_path = tmp_path / "d.db"
    migrate(db_path, migrations, verbose=False)
    connection = connect(db_path)
    _dataset(connection, "hotpotqa", ["gold_docs"])
    layer_id = _layer(connection)
    _subset(connection, layer_id, "hotpotqa", ["d1"])
    annotation_id = repo.add_annotation(
        connection,
        level="sample",
        target_id="hotpotqa:s1",
        author_kind="human",
        author="me",
        labels=["retrieval_miss"],
        note=None,
        source="human",
        confidence=None,
    )
    connection.commit()
    connection.close()

    # 已有数据的库应用后续结构变更时需要备份。
    (migrations / "002_probe.sql").write_text(
        "ALTER TABLE index_layer ADD COLUMN probe TEXT;", encoding="utf-8"
    )
    assert migrate(db_path, migrations, verbose=False) == ["002_probe"]
    assert (tmp_path / "d.db.pre-002_probe").is_file()

    connection = connect(db_path, read_only=True)
    try:
        # 数据与不可重建的标注都必须活着。
        assert repo.subset_docs(connection, layer_id, "hotpotqa")[0]["doc_id"] == "d1"
        assert repo.annotations_for(connection, "sample", "hotpotqa:s1")[0]["id"] == annotation_id
    finally:
        connection.close()


def test_renormalizing_does_not_empty_existing_layers(db: sqlite3.Connection):
    """重跑归一化不能清空已有索引层的 QA 归属。"""
    _dataset(db, "hotpotqa", ["gold_docs"])
    rows = [
        {
            "sample_id": f"hotpotqa:s{i}",
            "dataset_sample_id": f"s{i}",
            "question": f"q{i}",
            "answers": ["a"],
            "gold_doc_ids": ["d1"],
            "metadata": {},
        }
        for i in range(3)
    ]
    repo.replace_samples(db, "hotpotqa", rows)
    layer_id = _layer(db)
    _subset(db, layer_id, "hotpotqa", ["d1"])
    db.execute(
        "INSERT INTO subset_sample (index_layer_id, dataset, sample_id) VALUES (?,?,?)",
        (layer_id, "hotpotqa", "hotpotqa:s0"),
    )
    db.commit()
    assert len(repo.subset_samples(db, layer_id, "hotpotqa")) == 1

    # 再跑一遍归一化，内容完全相同。
    repo.replace_samples(db, "hotpotqa", rows)
    db.commit()
    assert len(repo.subset_samples(db, layer_id, "hotpotqa")) == 1, (
        "re-normalising wiped the layer's sample membership"
    )

    # 内容变了也一样：sample_id 不变就不该触发 cascade。
    repo.replace_samples(
        db, "hotpotqa", [{**r, "question": f"changed {r['question']}"} for r in rows]
    )
    db.commit()
    kept = repo.subset_samples(db, layer_id, "hotpotqa")
    assert len(kept) == 1
    assert kept[0]["question"] == "changed q0"


def test_samples_dropped_upstream_do_cascade(db: sqlite3.Connection):
    """反面：上游真的删掉一条样本时，cascade **应该**发生。"""
    _dataset(db, "hotpotqa", ["gold_docs"])
    rows = [
        {
            "sample_id": f"hotpotqa:s{i}",
            "dataset_sample_id": f"s{i}",
            "question": f"q{i}",
            "answers": ["a"],
            "gold_doc_ids": ["d1"],
            "metadata": {},
        }
        for i in range(2)
    ]
    repo.replace_samples(db, "hotpotqa", rows)
    layer_id = _layer(db)
    _subset(db, layer_id, "hotpotqa", ["d1"])
    for i in range(2):
        db.execute(
            "INSERT INTO subset_sample (index_layer_id, dataset, sample_id) VALUES (?,?,?)",
            (layer_id, "hotpotqa", f"hotpotqa:s{i}"),
        )
    db.commit()

    # 上游只剩一条。
    repo.replace_samples(db, "hotpotqa", rows[:1])
    db.commit()
    remaining = {s["sample_id"] for s in repo.subset_samples(db, layer_id, "hotpotqa")}
    assert remaining == {"hotpotqa:s0"}


def test_foreign_keys_are_enforced_on_every_connection(db: sqlite3.Connection):
    """外键必须逐连接开。不开的话孤儿行会静默堆积，几个月后才发现。"""
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO subset_sample (index_layer_id, dataset, sample_id) VALUES (999,'x','y')"
        )


# --- 续跑判据 ---------------------------------------------------------------


def test_pending_imports_keeps_colliding_doc_ids_of_two_datasets_apart(db: sqlite3.Connection):
    """doc_id 跨数据集会撞，续跑必须按 (层, 数据集, doc_id) 三项判。"""
    _dataset(db, "hotpotqa", ["gold_docs"])
    _dataset(db, "musique", ["gold_docs"])
    layer_id = _layer(db)
    _subset(db, layer_id, "hotpotqa", ["d1", "d2"])
    _subset(db, layer_id, "musique", ["d1", "d2"])

    repo.record_page(
        db,
        layer_id,
        "hotpotqa",
        doc_id="d1",
        page_id="p1",
        space_id="sp",
        title=None,
        md_sha256="0" * 64,
    )
    db.commit()

    assert [d["doc_id"] for d in repo.pending_imports(db, layer_id, "hotpotqa")] == ["d2"]
    # musique 的 d1 与 hotpotqa 的 d1 同名但是不同文档，必须仍然待导入。
    assert [d["doc_id"] for d in repo.pending_imports(db, layer_id, "musique")] == ["d1", "d2"]


def test_recording_the_same_page_twice_raises_instead_of_overwriting(db: sqlite3.Connection):
    """重复导入意味着 Akasha 里多了一个没人引用的重复 page —— 要报错，不要静默覆盖。"""
    _dataset(db, "hotpotqa", ["gold_docs"])
    layer_id = _layer(db)
    _subset(db, layer_id, "hotpotqa", ["d1"])
    for _ in range(1):
        repo.record_page(
            db,
            layer_id,
            "hotpotqa",
            doc_id="d1",
            page_id="p1",
            space_id="sp",
            title=None,
            md_sha256="0" * 64,
        )
    with pytest.raises(sqlite3.IntegrityError):
        repo.record_page(
            db,
            layer_id,
            "hotpotqa",
            doc_id="d1",
            page_id="p2",
            space_id="sp",
            title=None,
            md_sha256="0" * 64,
        )


def _query_layer(db: sqlite3.Connection, layer_id: int) -> int:
    return repo.create_query_layer(
        db,
        index_layer_id=layer_id,
        label="Q",
        config_hash="qh",
        score_threshold=None,
        concurrency=1,
        request_interval_seconds=0.5,
        model_configs=MODEL_CONFIGS,
        model_configs_match_index=True,
        allow_config_drift=False,
    )


def _response(db: sqlite3.Connection, qid: int, sample_id: str, status: int, at: str) -> None:
    repo.record_response(
        db,
        qid,
        sample_id=sample_id,
        dataset="hotpotqa",
        question="Q",
        requested_at=at,
        latency_ms=100,
        http_status=status,
        error=None,
        response={"answerMode": "knowledge"} if status == 200 else None,
    )


def test_completed_sample_ids_includes_failures_so_reruns_do_not_burn_llm_calls(
    db: sqlite3.Connection,
):
    """失败行也算已完成：重跑一次要烧 LLM 调用，而失败率本身是结果的一部分。"""
    _dataset(db, "hotpotqa", ["gold_docs"])
    layer_id = _layer(db)
    qid = _query_layer(db, layer_id)
    _response(db, qid, "hotpotqa:ok", 200, "2026-09-10T09:00:00Z")
    _response(db, qid, "hotpotqa:bad", 500, "2026-09-10T09:01:00Z")
    db.commit()
    assert repo.completed_sample_ids(db, qid) == {"hotpotqa:ok", "hotpotqa:bad"}


def test_retrying_failures_requires_deleting_them_first(db: sqlite3.Connection):
    """失败行恢复策略：只能显式删除后重跑，删了多少行是可见的。"""
    _dataset(db, "hotpotqa", ["gold_docs"])
    layer_id = _layer(db)
    qid = _query_layer(db, layer_id)
    _response(db, qid, "hotpotqa:ok", 200, "2026-09-10T09:00:00Z")
    _response(db, qid, "hotpotqa:bad", 0, "2026-09-10T09:01:00Z")
    db.commit()

    with pytest.raises(sqlite3.IntegrityError):
        _response(db, qid, "hotpotqa:bad", 200, "2026-09-10T09:02:00Z")
    db.rollback()

    assert repo.delete_failed_responses(db, qid) == 1
    db.commit()
    assert repo.completed_sample_ids(db, qid) == {"hotpotqa:ok"}


def test_request_window_covers_all_accumulated_sessions(db: sqlite3.Connection):
    """时间窗从行里现算，覆盖累积的全部会话。"""
    _dataset(db, "hotpotqa", ["gold_docs"])
    layer_id = _layer(db)
    qid = _query_layer(db, layer_id)
    _response(db, qid, "hotpotqa:a", 200, "2026-09-10T09:00:00Z")
    db.commit()
    # 第二次会话，中间隔了一天。
    _response(db, qid, "hotpotqa:b", 200, "2026-09-11T10:00:00Z")
    db.commit()

    assert repo.request_window(db, qid) == ("2026-09-10T09:00:00Z", "2026-09-11T10:00:00Z")


# --- 质量闸门 ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "gates", "expected"),
    [
        (
            "四项全 0",
            {
                "missingChunkPageCount": 0,
                "missingEmbeddingPageCount": 0,
                "missingSourcePageCount": 0,
                "stalePageCount": 0,
            },
            True,
        ),
        (
            "一项非 0",
            {
                "missingChunkPageCount": 3,
                "missingEmbeddingPageCount": 0,
                "missingSourcePageCount": 0,
                "stalePageCount": 0,
            },
            False,
        ),
        # 这一条是那次线上事故的形态：响应信封没剥，四项全读成 None,
        # 而 all(value == 0) 对空值集合返回 True —— 闸门假通过。
        ("字段全缺", {}, False),
        (
            "缺一项",
            {
                "missingChunkPageCount": 0,
                "missingEmbeddingPageCount": 0,
                "missingSourcePageCount": 0,
            },
            False,
        ),
    ],
)
def test_quality_gate_treats_missing_counts_as_failure(
    db: sqlite3.Connection, label: str, gates: dict[str, int], expected: bool
):
    """取不到值必须判失败。假通过会带着半成品索引跑出一堆没意义的指标。"""
    _dataset(db, "hotpotqa", ["gold_docs"])
    layer_id = _layer(db)
    passed, _ = repo.record_quality_gate(db, layer_id, gates=gates, report={})
    assert passed is expected, label


# --- 身份哈希 ---------------------------------------------------------------


def test_config_hash_ignores_dict_order():
    """同一份配置不该因为字段顺序不同而算成两个层。"""
    left = identity.config_hash({"a": 1, "b": {"c": 2, "d": 3}})
    right = identity.config_hash({"b": {"d": 3, "c": 2}, "a": 1})
    assert left == right


def _swap(feature: str, model: str) -> dict[str, Any]:
    return {
        "configs": [
            {**c, "model": model} if c["feature"] == feature else c
            for c in MODEL_CONFIGS["configs"]
        ]
    }


def test_index_layer_hash_changes_with_embedding_but_not_with_answer_model():
    """索引层只吃实际文档集 + compiler + embedding。"""
    base = identity.index_layer_hash(subset_hash="docs-abc", model_configs=MODEL_CONFIGS)

    assert identity.index_layer_hash(subset_hash="docs-abc", model_configs=_swap("answer", "x")) == base
    assert (
        identity.index_layer_hash(subset_hash="docs-abc", model_configs=_swap("embedding", "x"))
        != base
    )
    assert (
        identity.index_layer_hash(subset_hash="docs-abc", model_configs=_swap("compiler", "x"))
        != base
    )
    # 文档集变了也必须换身份 —— 这是内容寻址的那一半。
    assert identity.index_layer_hash(subset_hash="docs-xyz", model_configs=MODEL_CONFIGS) != base


def test_subset_hash_is_content_addressed_over_the_actual_documents(db: sqlite3.Connection):
    """``subset_hash`` 必须由实际文档算出，不是由抽样配置算出。"""
    _dataset(db, "hotpotqa", ["gold_docs"])
    left = _layer(db, "L1")
    right = _layer(db, "L2")
    _subset(db, left, "hotpotqa", ["d1", "d2"])
    _subset(db, right, "hotpotqa", ["d1", "d2"])
    db.commit()

    # 同一批文档 -> 同一个哈希，与 label、与建层顺序都无关。
    assert repo.recompute_subset_hash(db, left) == repo.recompute_subset_hash(db, right)

    # 文档集不同 -> 哈希必须不同。
    _subset(db, right, "hotpotqa", ["d1", "d3"])
    db.commit()
    assert repo.recompute_subset_hash(db, left) != repo.recompute_subset_hash(db, right)


def test_query_layer_hash_ignores_concurrency_but_tracks_score_threshold():
    """并发度只影响跑多快，不影响跑出什么，所以不进哈希。"""
    base = identity.query_layer_hash(
        index_config_hash="i", score_threshold=None, model_configs=MODEL_CONFIGS
    )
    assert (
        identity.query_layer_hash(
            index_config_hash="i", score_threshold=0.6, model_configs=MODEL_CONFIGS
        )
        != base
    )


def test_judge_hash_never_depends_on_the_api_key():
    """：judge 的身份只吃 base_url + model，绝不吃 api_key。"""
    left = identity.judge_hash(base_url="https://x/v1", model="m")
    right = identity.judge_hash(base_url="https://x/v1", model="m", params={})
    assert left == right
    # 函数签名里根本没有 api_key 这个参数 —— 这就是判据。
    assert "api_key" not in identity.judge_hash.__code__.co_varnames


def test_embedding_drift_is_detectable(db: sqlite3.Connection):
    """embedding 漂移必须可判定：它是唯一会静默失效的那个。"""
    other = {
        "configs": [
            {**c, "model": "changed"} if c["feature"] == "embedding" else c
            for c in MODEL_CONFIGS["configs"]
        ]
    }
    assert identity.embedding_matches(MODEL_CONFIGS, MODEL_CONFIGS)
    assert not identity.embedding_matches(MODEL_CONFIGS, other)
    # compiler 换了只是不可比，不是静默失效，所以两者要能分开判。
    assert identity.compiler_matches(MODEL_CONFIGS, other)


# --- reindex 的保护名单 -----------------------------------------------------


def test_reindex_never_lists_unrebuildable_tables_as_rebuildable():
    """标注与 judge 判决没有上游可重算，一个粗心的 DELETE 就没了（）。"""
    assert not REBUILDABLE_TABLES & set(PROTECTED_TABLES)
    for table in PROTECTED_TABLES:
        assert table not in REBUILDABLE_TABLES


def test_clearing_eval_results_keeps_judge_verdicts_and_annotations(db: sqlite3.Connection):
    """重跑确定性指标不该顺手清掉要花钱的 judge 判决和无法重算的标注。"""
    _dataset(db, "hotpotqa", ["gold_docs"])
    layer_id = _layer(db)
    qid = _query_layer(db, layer_id)
    eid = repo.create_eval_layer(
        db, query_layer_id=qid, label="E", config_hash="eh", ks=[10], metrics=["recall"]
    )
    repo.record_sample_metrics(db, eid, "hotpotqa:s1", "hotpotqa", {"recall@10": 1.0})
    repo.record_judge_verdict(
        db,
        eid,
        sample_id="hotpotqa:s1",
        metric="faithfulness",
        score=0.8,
        failure_kind=None,
        reasoning={"claims": []},
        raw_response="{}",
        provider_hash="ph",
        prompt_version="v1",
    )
    repo.add_annotation(
        db,
        level="sample",
        target_id="hotpotqa:s1",
        author_kind="human",
        author="me",
        labels=["answer_form"],
        note=None,
        source="human",
        confidence=None,
    )
    db.commit()

    repo.clear_eval_results(db, eid)
    db.commit()

    assert repo.sample_metrics_of(db, eid, "hotpotqa:s1") == {}
    assert len(repo.judge_verdicts(db, eid, "faithfulness")) == 1
    assert len(repo.annotations_for(db, "sample", "hotpotqa:s1")) == 1


@pytest.mark.parametrize("change", ["schema", "ledger"])
def test_baseline_rejects_incomplete_or_changed_legacy_database(tmp_path: Path, change: str):
    from akasha_benchmark.store.migrate import LEGACY_CHECKSUMS

    path = tmp_path / "legacy.db"
    migrate(path, verbose=False)
    connection = connect(path)
    connection.execute("DELETE FROM schema_migration")
    connection.executemany(
        "INSERT INTO schema_migration VALUES (?, '2026-01-01T00:00:00Z', ?)",
        LEGACY_CHECKSUMS.items(),
    )
    if change == "schema":
        connection.execute("ALTER TABLE connection ADD COLUMN unexpected TEXT")
    else:
        connection.execute("DELETE FROM schema_migration WHERE version='006_drop_slug_prefix'")
    connection.commit()
    before = list(connection.execute("SELECT * FROM schema_migration"))
    with pytest.raises(RuntimeError, match="differs|first be upgraded"):
        migrate(path, verbose=False)
    assert list(connection.execute("SELECT * FROM schema_migration")) == before
    assert not list(tmp_path.glob("*.pre-*"))
    connection.close()
