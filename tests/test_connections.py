"""连接配置（单例）与那两道防静默失效的闸门。"""

from __future__ import annotations

from pathlib import Path

import pytest
from akasha_benchmark.store import connect, repo
from akasha_benchmark.store.migrate import migrate
from fastapi.testclient import TestClient

from akasha_platform.main import create_app
from akasha_platform.settings import Settings

DATASET = "hotpotqa"


@pytest.fixture
def seeded(tmp_path: Path):
    """一个已入库的层：落在 ws-a，有 space 与 page_map。"""
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
        corpus_rows=2,
        dedup_stats={},
        gold_count_distribution={},
        unique_question_texts=1,
    )
    repo.update_connection(connection, base_url="http://prod:3000", email="prod@example.com")

    layer_id = repo.create_index_layer(
        connection,
        label="L",
        subset_hash="sh",
        seed=1,
        qa_limit=1,
        negatives_ratio=1.0,
        narrativeqa_docs=1,
    )
    repo.upsert_index_layer_dataset(
        connection,
        layer_id,
        DATASET,
        strategy="s",
        qa_count=1,
        corpus_count=2,
        gold_doc_count=1,
        negative_doc_count=1,
        strata={},
        normalized_qa_sha256="0" * 64,
        normalized_corpus_sha256="1" * 64,
    )
    repo.set_space(
        connection, layer_id, DATASET, space_id="space-a", space_slug="benchhotpotqaL",
        space_reused=False,
    )
    for doc_id in ("d1", "d2"):
        repo.record_page(
            connection,
            layer_id,
            DATASET,
            doc_id=doc_id,
            page_id=f"page-{doc_id}",
            space_id="space-a",
            title=doc_id,
            md_sha256="a" * 64,
        )
    repo.update_index_layer(
        connection,
        layer_id,
        workspace_id="ws-a",
        workspace_name="prod",
        ingested_at="2026-01-01T00:00:00Z",
        connection_json=repo.dumps({"base_url": "http://prod:3000", "email": "prod@example.com"}),
    )
    connection.commit()

    api = TestClient(create_app(Settings(db_path=db)))
    yield api, connection, layer_id
    connection.close()


# --- 单例 -------------------------------------------------------------------


def test_the_connection_is_a_singleton_enforced_by_the_schema(tmp_path: Path):
    """只有一份配置，而且这一点写在 schema 里而不是只写在代码里。"""
    import sqlite3

    db = tmp_path / "t.db"
    migrate(db, verbose=False)
    connection = connect(db)

    # 迁移保证它总是存在 —— 读配置不必处理空表。
    assert repo.get_connection_row(connection) is not None
    assert connection.execute("SELECT COUNT(*) FROM connection").fetchone()[0] == 1

    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        connection.execute(
            "INSERT INTO connection (id, created_at, updated_at) VALUES (2, 'x', 'x')"
        )
    connection.close()


def test_there_is_no_way_to_add_or_remove_a_connection(seeded):
    """API 上没有新增与删除 —— 只有读和改。"""
    api, _, _ = seeded
    assert api.post("/api/connection", json={"base_url": "http://x"}).status_code == 405
    assert api.delete("/api/connection").status_code == 405


# --- 闸门 A：workspace 不匹配（红线）---------------------------------------


def test_workspace_mismatch_uses_the_server_resolved_value(seeded):
    """判据是**登录后 users/me 解析出的** workspace，不是任何配置项。"""
    _, connection, layer_id = seeded
    # 登录到同一个 workspace：放行。
    assert repo.workspace_mismatch(connection, layer_id, "ws-a") is None

    # 登录到另一个：拒绝。
    message = repo.workspace_mismatch(connection, layer_id, "ws-b")
    assert message is not None
    # 报错要说后果，不只说「不匹配」—— 否则读者不知道为什么这是致命的。
    assert "ws-a" in message and "ws-b" in message
    assert "dangling" in message


def test_readiness_does_not_claim_to_check_the_workspace(seeded):
    """``index_layer_readiness`` 是纯库函数，登不了 Akasha。"""
    _, connection, layer_id = seeded
    readiness = repo.index_layer_readiness(connection, layer_id)
    assert readiness["workspace_recorded"] == "ws-a"
    assert not any("workspace" in reason for reason in readiness["reasons"])


def test_a_layer_without_spaces_is_not_workspace_checked(seeded):
    """还没入库过的层没有 space，跑在哪个 workspace 上都行。"""
    _, connection, _ = seeded
    fresh = repo.create_index_layer(
        connection, label="F", subset_hash="s2", seed=1, qa_limit=1,
        negatives_ratio=1.0, narrativeqa_docs=1,
    )
    connection.commit()
    assert repo.workspace_mismatch(connection, fresh, "ws-anything") is None


def test_a_layer_without_a_recorded_workspace_is_not_compared(seeded):
    """reindex 导进来的历史层可能没记 workspace_id，那时无从比较。"""
    _, connection, layer_id = seeded
    repo.update_index_layer(connection, layer_id, workspace_id=None)
    connection.commit()
    assert repo.workspace_mismatch(connection, layer_id, "ws-b") is None


# --- 闸门 B：ensure_space 校验 space 身份 ----------------------------------


class _FakeClient:
    """只实现 ensure_space 用到的两个方法。"""

    def __init__(self, spaces: list[dict[str, str]]) -> None:
        self.spaces = spaces
        self.created: list[str] = []

    def list_spaces(self, page: int = 1, limit: int = 100) -> dict:
        return {"items": self.spaces, "meta": {"hasNextPage": False}}

    def create_space(self, name: str, slug: str, description: str = "") -> dict:
        self.created.append(slug)
        return {"id": f"new-{slug}", "slug": slug}


@pytest.mark.parametrize(
    "spaces,message",
    [
        ([{"id": "space-OTHER", "slug": "benchhotpotqaL"}], "Same slug, different space"),
        ([], "not visible under the current connection"),
    ],
    ids=["different-space", "missing-space"],
)
def test_ensure_space_rejects_invalid_recorded_identity(spaces, message):
    from akasha_benchmark.ingest import ensure_space

    client = _FakeClient(spaces)
    with pytest.raises(RuntimeError, match=message):
        ensure_space(client, DATASET, "L", "bench", "space-a")
    assert client.created == []


def test_ensure_space_reuses_the_recorded_space_when_it_matches():
    from akasha_benchmark.ingest import ensure_space

    client = _FakeClient([{"id": "space-a", "slug": "benchhotpotqaL"}])
    space = ensure_space(client, DATASET, "L", "bench", "space-a")
    assert space["id"] == "space-a"
    assert space["reused"] is True


def test_ensure_space_creates_one_when_nothing_was_recorded():
    """首次入库：没有已记的 space，正常创建。"""
    from akasha_benchmark.ingest import ensure_space

    client = _FakeClient([])
    space = ensure_space(client, DATASET, "L", "bench", None)
    assert space["reused"] is False
    assert client.created == ["benchhotpotqaL"]


# --- 清掉入库产物 -----------------------------------------------------------


def test_discarding_ingest_needs_an_explicit_confirmation(seeded):
    """重新入库要烧掉整批编译时间，不该被一次误点触发。"""
    api, _, layer_id = seeded
    response = api.post(f"/api/layers/index/{layer_id}/discard-ingest", json={})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "confirm=true" in detail
    # 代价要说出来：按约 40 秒/篇估的重编译时间。
    assert "h of compilation" in detail


def test_discarding_ingest_keeps_the_subset_and_the_remote_spaces(seeded):
    """清产物只清入库结果。"""
    api, connection, layer_id = seeded
    repo.replace_subset(
        connection, layer_id, DATASET, [],
        [{"doc_id": "d1", "md_text": "x", "md_sha256": "a" * 64, "is_gold": True}],
    )
    connection.commit()

    response = api.post(f"/api/layers/index/{layer_id}/discard-ingest", json={"confirm": True})
    assert response.status_code == 200
    assert response.json()["discarded"]["page_map"] == 2

    assert repo.page_map_counts(connection, layer_id) == {}
    assert repo.spaces_of(connection, layer_id) == {}
    # 子集还在。
    assert len(repo.subset_docs(connection, layer_id, DATASET, with_text=False)) == 1
    # 层退回未入库。
    layer = repo.get_index_layer(connection, layer_id)
    assert layer["ingested_at"] is None


def test_discarding_clears_the_identity_that_no_longer_holds(seeded):
    """config_hash、模型快照与入库身份也要清掉。"""
    api, connection, layer_id = seeded
    repo.seal_index_layer(connection, layer_id, "hash-abc")
    repo.update_index_layer(connection, layer_id, model_configs_json='{"x":1}')
    connection.commit()

    api.post(f"/api/layers/index/{layer_id}/discard-ingest", json={"confirm": True})
    layer = repo.get_index_layer(connection, layer_id)
    assert layer["config_hash"] is None
    assert layer["model_configs_json"] is None
    assert layer["workspace_id"] is None
    assert layer["connection_json"] is None


def test_discarding_a_layer_that_was_never_ingested_is_rejected(seeded):
    api, connection, _ = seeded
    fresh = repo.create_index_layer(
        connection, label="F", subset_hash="s2", seed=1, qa_limit=1,
        negatives_ratio=1.0, narrativeqa_docs=1,
    )
    connection.commit()
    response = api.post(f"/api/layers/index/{fresh}/discard-ingest", json={"confirm": True})
    assert response.status_code == 409
    assert "no ingest to discard" in response.json()["detail"]


# --- 重抽子集（另一个静默失效）---------------------------------------------


def test_resampling_an_ingested_layer_is_refused(seeded):
    """已入库的层拒绝重抽子集。"""
    from akasha_benchmark.subset import LayerAlreadyIngested, ensure_layer

    _, connection, _ = seeded
    with pytest.raises(LayerAlreadyIngested, match="Re-sampling"):
        ensure_layer(
            connection,
            label="L",
            seed=99,
            qa_limit=50,
            negatives_ratio=1.0,
            narrativeqa_docs=1,
            datasets=[DATASET],
        )


def test_a_layer_that_was_never_ingested_can_be_resampled(seeded):
    """没入库过的层可以随便重抽 —— 那时什么都还没建。"""
    from akasha_benchmark.subset import ensure_layer

    _, connection, _ = seeded
    repo.create_index_layer(
        connection, label="F", subset_hash="s2", seed=1, qa_limit=1,
        negatives_ratio=1.0, narrativeqa_docs=1,
    )
    connection.commit()
    layer_id, created = ensure_layer(
        connection, label="F", seed=7, qa_limit=9, negatives_ratio=1.0,
        narrativeqa_docs=1, datasets=[DATASET],
    )
    assert created is False  # 同 label 复用，不多出一层


def test_readiness_compares_document_sets_not_counts(seeded):
    """兜底：假设重抽还是发生了，readiness 必须发现。"""
    _, connection, layer_id = seeded
    repo.record_quality_gate(
        connection,
        layer_id,
        gates={
            k: 0
            for k in (
                "missingChunkPageCount",
                "missingEmbeddingPageCount",
                "missingSourcePageCount",
                "stalePageCount",
            )
        },
        report={},
    )
    repo.record_compile_run(
        connection, layer_id, accepted_run_count=1, coalesced_run_count=0,
        status_counts={"succeeded": 1}, terminal={"succeeded": 1}, timed_out=False,
        requested_at="x", finished_at="y",
    )
    # 子集与 page_map 一致（d1/d2 两篇都导过）时应当通过。
    repo.replace_subset(
        connection, layer_id, DATASET, [],
        [
            {"doc_id": d, "md_text": d, "md_sha256": "a" * 64, "is_gold": True}
            for d in ("d1", "d2")
        ],
    )
    connection.commit()
    assert repo.index_layer_readiness(connection, layer_id)["ready"] is True

    # 换成**同样两篇但不同 doc_id** —— 条数相等，集合不同。
    repo.replace_subset(
        connection, layer_id, DATASET, [],
        [
            {"doc_id": d, "md_text": d, "md_sha256": "b" * 64, "is_gold": True}
            for d in ("x1", "x2")
        ],
    )
    connection.commit()

    readiness = repo.index_layer_readiness(connection, layer_id)
    assert readiness["ready"] is False
    reason = readiness["reasons"][0]
    assert "re-sampled after ingest" in reason
    # 两个方向都要报：子集里没导过的，与导过但已不在子集里的。
    assert "never imported" in reason and "no longer in it" in reason


# --- 改配置的影响 -----------------------------------------------------------


def test_put_connection_is_a_pure_write(seeded):
    """PUT /api/connection 只写,不附带副作用提示。
    workspace 漂移的拦阻交回给 ingest / query 登录那一步。
    """
    api, _, _ = seeded
    body = api.put("/api/connection", json={"base_url": "http://elsewhere:3000"}).json()
    assert "warnings" not in body
    assert body["connection"]["base_url"] == "http://elsewhere:3000"

    # 改速率同样干净。
    quiet = api.put("/api/connection", json={"concurrency": 4}).json()
    assert "warnings" not in quiet


def test_the_connection_view_exposes_ingested_layers(seeded):
    """配置页要能看到「改这个会影响哪些层」，而不是改完才知道。"""
    api, _, layer_id = seeded
    body = api.get("/api/connection").json()
    assert [row["id"] for row in body["ingested_layers"]] == [layer_id]
    assert body["ingested_layers"][0]["workspace_id"] == "ws-a"


def test_the_layer_view_reports_the_ingest_identity_not_the_current_config(seeded):
    """层上显示的是**入库时**的身份，不是现在的配置。"""
    api, _, layer_id = seeded
    api.put("/api/connection", json={"base_url": "http://changed:3000"})

    layers = {entry["id"]: entry for entry in api.get("/api/layers").json()["index_layers"]}
    identity = layers[layer_id]["ingest_identity"]
    assert identity["base_url"] == "http://prod:3000"
    assert identity["workspace_id"] == "ws-a"


# --- 迁移 -------------------------------------------------------------------


def test_baseline_preserves_populated_legacy_database(tmp_path: Path):
    """Adopting the final schema must preserve credentials and ingest products."""
    from akasha_benchmark.store.migrate import LEGACY_CHECKSUMS

    db = tmp_path / "t.db"
    migrate(db, verbose=False)
    connection = connect(db)
    repo.upsert_dataset(
        connection,
        name=DATASET,
        adapter="A",
        adapter_version="1",
        provides=["gold_docs"],
        identity_rules={},
        qa_path="q",
        qa_sha256="0" * 64,
        qa_rows=1,
        corpus_path="c",
        corpus_sha256="1" * 64,
        corpus_rows=2,
        dedup_stats={},
        gold_count_distribution={},
        unique_question_texts=1,
    )
    connection.execute("DELETE FROM schema_migration")
    connection.executemany(
        "INSERT INTO schema_migration VALUES (?, '2026-01-01T00:00:00Z', ?)",
        LEGACY_CHECKSUMS.items(),
    )
    connection.execute(
        "UPDATE connection SET base_url=?, email=?, password=?, concurrency=?, "
        "request_interval_seconds=? WHERE id=1",
        ("http://legacy:3000", "legacy@example.com", "legacy-pw", 3, 1.5),
    )
    ingested = repo.create_index_layer(
        connection, label="old", subset_hash="s1", seed=1, qa_limit=1,
        negatives_ratio=1.0, narrativeqa_docs=1,
    )
    repo.upsert_index_layer_dataset(
        connection, ingested, DATASET, strategy="s", qa_count=1, corpus_count=2,
        gold_doc_count=1, negative_doc_count=1, strata={},
        normalized_qa_sha256="0" * 64, normalized_corpus_sha256="1" * 64,
    )
    repo.set_space(
        connection, ingested, DATASET, space_id="space-legacy", space_slug="sl",
        space_reused=False,
    )
    for doc_id in ("d1", "d2"):
        repo.record_page(
            connection, ingested, DATASET, doc_id=doc_id, page_id=f"p-{doc_id}",
            space_id="space-legacy", title=doc_id, md_sha256="a" * 64,
        )
    query_layer_id = repo.create_query_layer(
        connection, index_layer_id=ingested, label="Q", config_hash="qh",
        score_threshold=None, concurrency=1, request_interval_seconds=0.5,
        model_configs=None, model_configs_match_index=True, allow_config_drift=False,
    )
    repo.update_index_layer(connection, ingested, ingested_at="2026-01-01T00:00:00Z")
    connection.commit()
    connection.close()

    assert migrate(db, verbose=False) == ["001_initial"]
    assert migrate(db, verbose=False) == []
    assert db.with_name("t.db.pre-baseline").is_file()

    connection = connect(db)
    # 产物一行不丢。
    assert repo.page_map_counts(connection, ingested) == {DATASET: 2}
    assert repo.spaces_of(connection, ingested) == {DATASET: "space-legacy"}
    assert repo.get_query_layer(connection, query_layer_id) is not None

    # 保留密钥与非默认数值。
    row = repo.get_connection_row(connection)
    assert row["base_url"] == "http://legacy:3000"
    assert row["password"] == "legacy-pw"
    assert row["concurrency"] == 3
    assert row["request_interval_seconds"] == 1.5
    # 单例，且没有 label / workspace_id 这些后来删掉的列。
    assert connection.execute("SELECT COUNT(*) FROM connection").fetchone()[0] == 1
    assert "workspace_id" not in row.keys()
    assert "label" not in row.keys()

    # 死重的列都删掉了。
    layer_columns = {r[1] for r in connection.execute("PRAGMA table_info(index_layer)")}
    assert "connection_id" not in layer_columns
    assert "connection_sealed_at" not in layer_columns
    query_columns = {r[1] for r in connection.execute("PRAGMA table_info(query_layer)")}
    assert "connection_id" not in query_columns

    assert connection.execute("SELECT version FROM schema_migration").fetchall()[0][0] == "001_initial"
    connection.close()
