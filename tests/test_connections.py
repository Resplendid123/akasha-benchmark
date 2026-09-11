"""连接配置（单例）与那两道防静默失效的闸门。

**这一组的核心是一个不报错的失效**。把连接改到另一个 workspace 后重跑 ingest：

1. ``_find_space`` 走 ``list_spaces``，那是按当前 workspace 过滤的
2. 新 workspace 下找不到同 slug 的 space -> ``ensure_space`` 建一个新的
3. ``set_space`` 覆盖库里的 space_id
4. ``page_map`` 里的 page_id 还指向旧 workspace 的页

之后查询照常跑，每条都召回不到 —— 看起来像「这批语料检索效果差」，
而不像一个配置错误。所以这两道闸门必须**拒绝执行**，不是警告。

连接只有一份（``CHECK (id = 1)``），只能改。历史记录靠
``index_layer.connection_json`` 与 ``workspace_id`` —— 它们记的是入库时的值。
"""

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

    api = TestClient(create_app(Settings(db_path=db, web_dist=tmp_path / "none")))
    yield api, connection, layer_id
    connection.close()


# --- 单例 -------------------------------------------------------------------


def test_the_connection_is_a_singleton_enforced_by_the_schema(tmp_path: Path):
    """只有一份配置，而且这一点写在 schema 里而不是只写在代码里。

    只靠代码约束的话，任何一处漏判就能插进第二行，而那时「哪一行是真的」
    就成了个没有答案的问题。
    """
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
    """判据是**登录后 users/me 解析出的** workspace，不是任何配置项。

    workspace 由服务端决定（自建部署走 ``workspaceRepo.findFirst()``），客户端
    选不了。所以配置上没有这一项，比对的另一头是层入库时记下的值。
    """
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
    """``index_layer_readiness`` 是纯库函数，登不了 Akasha。

    所以它**不做** workspace 比对 —— 只把层记下的那个值报出来，让调用方拿服务端
    的值去判。假装在这里判过会让人以为离线就能发现问题。
    """
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
    """reindex 导进来的历史层可能没记 workspace_id，那时无从比较。

    但 ``ensure_space`` 的 space 身份校验仍然拦得住 —— 那一道更强，
    它比的是 space_id 而不是 workspace。
    """
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


def test_ensure_space_rejects_a_different_space_with_the_same_slug():
    """**slug 相同不等于同一个 space。**

    ``list_spaces`` 按当前 workspace 过滤，所以换了 workspace 之后同一个 slug 会
    解析到另一个 space。照旧复用它就等于把语料导进了错的地方，
    而库里记的 page_id 全部悬空。
    """
    from akasha_benchmark.ingest import ensure_space

    client = _FakeClient([{"id": "space-OTHER", "slug": "benchhotpotqaL"}])
    with pytest.raises(RuntimeError, match="Same slug, different space"):
        ensure_space(client, DATASET, "L", "bench", "space-a")


def test_ensure_space_rejects_a_missing_space_when_one_was_recorded():
    """库里记了 space 但现在看不到它 —— 那意味着换了 workspace。"""
    from akasha_benchmark.ingest import ensure_space

    client = _FakeClient([])
    with pytest.raises(RuntimeError, match="not visible under the current connection"):
        ensure_space(client, DATASET, "L", "bench", "space-a")
    # 不能悄悄建一个新的。
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
    """清产物只清入库结果。

    子集是离线抽的，与连接无关，所以留着 —— 否则要重抽，而重抽可能得到另一批
    样本（seed 相同才不会）。远端的 space 也不删：我们不删别人的数据。
    """
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
    """config_hash、模型快照与入库身份也要清掉。

    它们是「这一层在那个部署上编译出来的东西」的身份。留着会让下一次入库看起来
    像是复用了一个已经封好的层，而那个身份已经不成立了。
    """
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
    """已入库的层拒绝重抽子集。

    ``replace_subset`` 删 ``subset_doc`` 但**不动 page_map**，所以重抽之后两者
    指向不同的文档集 —— 而条数往往仍然相等（同一个 qa_limit 抽出来的语料规模
    差不多），于是那种状态看起来是正常的。接下来查询打在装着旧文档的 Space 上、
    指标按新 gold 算，每条检索数都是 0，看起来像检索烂到极点。

    这一道是「编辑抽样参数」那个入口的前提：不拦住的话那就是一个会静默毁层的
    按钮。
    """
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
    """兜底：假设重抽还是发生了，readiness 必须发现。

    **按集合比，不按条数比。** 条数相等而集合不同是可能的，而那正是最危险的
    形态 —— 它看起来完全正常。
    """
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


def test_changing_the_identity_warns_about_ingested_layers(seeded):
    """改 base_url / email 可能落到另一个 workspace。

    不阻止（可能是在修一个填错的值），但必须说出影响 —— 改完之后那些层在
    ingest/query 时被拦住，不解释的话看起来像个 bug。
    """
    api, _, _ = seeded
    body = api.put("/api/connection", json={"base_url": "http://elsewhere:3000"}).json()
    assert body["warnings"]
    assert "already-ingested" in body["warnings"][0]

    # 改速率不影响任何东西，不该报警。
    quiet = api.put("/api/connection", json={"concurrency": 4}).json()
    assert quiet["warnings"] == []


def test_the_connection_view_exposes_ingested_layers(seeded):
    """配置页要能看到「改这个会影响哪些层」，而不是改完才知道。"""
    api, _, layer_id = seeded
    body = api.get("/api/connection").json()
    assert [row["id"] for row in body["ingested_layers"]] == [layer_id]
    assert body["ingested_layers"][0]["workspace_id"] == "ws-a"


def test_the_layer_view_reports_the_ingest_identity_not_the_current_config(seeded):
    """层上显示的是**入库时**的身份，不是现在的配置。

    配置改过之后这两者会不同，而层的 page_map 属于前者 —— 显示后者会让人以为
    那一层跑在新配置上。
    """
    api, _, layer_id = seeded
    api.put("/api/connection", json={"base_url": "http://changed:3000"})

    layers = {entry["id"]: entry for entry in api.get("/api/layers").json()["index_layers"]}
    identity = layers[layer_id]["ingest_identity"]
    assert identity["base_url"] == "http://prod:3000"
    assert identity["workspace_id"] == "ws-a"


# --- 迁移 -------------------------------------------------------------------


def test_the_migration_backfills_without_losing_ingest_products(tmp_path: Path):
    """连接那几次迁移的验收标准：**已入库的产物一行都不能丢。**

    真库里那 1722 行 page_map 是真实 Akasha 实例里的页，重挣一遍约 19 小时编译。
    所以这条用例把「迁移前有数据」这个场景摆出来：先在 001+002 的 schema 上造出
    一层已入库的数据与一份 app_config，再跑其余的，然后逐项核对。

    只在空库上验证过的迁移，第一次真用就炸 —— 而这一类事故的代价是那 19 小时。
    """
    from akasha_benchmark.store.migrate import discover

    db = tmp_path / "t.db"
    migrations = Path(__file__).resolve().parents[1] / "migrations"
    # 先只跑到 002（连接那一组之前）。按序号切而不是按文件名硬编码。
    early = tmp_path / "early"
    early.mkdir()
    for path in discover(migrations):
        if int(path.stem.split("_", 1)[0]) > 2:
            continue
        (early / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    migrate(db, early, verbose=False)

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
    # 002 时代的配置形态：散在 app_config 的键值里。
    repo.set_app_config(
        connection,
        {
            "base_url": "http://legacy:3000",
            "email": "legacy@example.com",
            "password": "legacy-pw",
            "concurrency": 3,
            "request_interval_seconds": 1.5,
        },
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

    # 现在跑其余的迁移。
    migrate(db, migrations, verbose=False)

    connection = connect(db)
    # 产物一行不丢。
    assert repo.page_map_counts(connection, ingested) == {DATASET: 2}
    assert repo.spaces_of(connection, ingested) == {DATASET: "space-legacy"}
    assert repo.get_query_layer(connection, query_layer_id) is not None

    # app_config 的配置迁进了 connection，**含密钥与非默认的数值**。
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

    # app_config 清空了：配置全在 connection 里。
    assert repo.get_app_config(connection) == {}
    connection.close()
