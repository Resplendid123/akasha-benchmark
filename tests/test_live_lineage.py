"""在线血缘回归：核对已保存案例的逐跳证据。

运行：AKASHA_LIVE=1 uv run pytest tests/test_live_lineage.py -v
需要评测库中配置只读 database_url，并已有包含 CASE_SAMPLE 的评测结果。
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from akasha_benchmark.config import load_config
from akasha_benchmark.store import DEFAULT_DB_PATH, connect, repo
from akasha_platform.main import create_app
from akasha_platform.settings import load_settings

# 已保存的案例：bridge / hard，gold 2 篇，recall@5 = 0.5。
CASE_SAMPLE = "hotpotqa:5ae4f2595542990ba0bbb1a8"
GOLD_MISSED = "6365"  # Dee Does Broadway —— 到 k=20 都没出现
GOLD_HIT = "6369"  # Cyndi Lauper —— 召回第 1 名

pytestmark = pytest.mark.skipif(
    os.environ.get("AKASHA_LIVE") != "1",
    reason="需要 AKASHA_LIVE=1、只读数据库与装好 run001 的库",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not load_config().database_url:
        pytest.skip("没有配置 database_url")
    if not DEFAULT_DB_PATH.is_file():
        pytest.skip(f"没有 {DEFAULT_DB_PATH}")
    return TestClient(create_app(load_settings()))


@pytest.fixture(scope="module")
def eval_layer_id() -> int:
    connection = connect(DEFAULT_DB_PATH, read_only=True)
    try:
        for layer in repo.list_eval_layers(connection):
            rows = repo.sample_evals(connection, int(layer["id"]))
            if any(row["sample_id"] == CASE_SAMPLE for row in rows):
                return int(layer["id"])
    finally:
        connection.close()
    pytest.skip(f"库里没有含 {CASE_SAMPLE} 的评测层")


def test_malformed_page_id_is_rejected_before_reaching_postgres(client: TestClient):
    """非 UUID 的 page_id 要回 400。"""
    response = client.get("/api/lineage/not-a-uuid")
    assert response.status_code == 400
    assert "not a valid page id" in response.json()["detail"]


def test_lineage_reproduces_the_case_study(client: TestClient, eval_layer_id: int):
    """六跳链路给出的结论必须与 案例手查的结论一致。"""
    response = client.get(
        f"/api/layers/eval/{eval_layer_id}/samples/{CASE_SAMPLE}/lineage"
    )
    assert response.status_code == 200
    body = response.json()

    # 这条样本是 knowledge 回答，所以它的低分是**真的漏 gold**，
    # 不是生成端回落 —— 四条 recall@5 < 1.0 里只有这一条是这样。
    assert body["answer_mode"] == "knowledge"
    assert body["metrics"]["recall@5"] == pytest.approx(0.5)

    by_doc = {entry["doc_id"]: entry for entry in body["gold"]}
    assert {GOLD_MISSED, GOLD_HIT} <= set(by_doc)

    missed = by_doc[GOLD_MISSED]
    assert missed["page_id"], "gold 6365 必须在 page_map 里"

    # 6365 编成 3 个 artifact（Dee Does Broadway / Dee Snider /
    # Source Summary: …）。artifact 数变了说明编译行为变了，那时这条案例的
    # 全部结论都要重新核。
    titles = {a["title"] for a in missed["lineage"]["artifacts"]}
    assert len(titles) == 3, titles
    assert any("Dee Does Broadway" in t for t in titles)
    assert any("Dee Snider" in t for t in titles)
    assert any(t.startswith("Source Summary") for t in titles)

    # 实体被物化成独立 artifact，canonical_key 是合并的键。
    keys = {a["canonical_key"] for a in missed["lineage"]["artifacts"]}
    assert "dee_does_broadway" in keys
    assert "dee_snider" in keys


def test_diff_shows_the_compiler_dropped_the_query_terms(
    client: TestClient, eval_layer_id: int
):
    """**这一条是整个平台存在的理由。"""
    body = client.get(
        f"/api/layers/eval/{eval_layer_id}/samples/{CASE_SAMPLE}/lineage"
    ).json()
    missed = {entry["doc_id"]: entry for entry in body["gold"]}[GOLD_MISSED]
    diff = missed["diff"]

    # 问题里的实词落在了「编译丢掉」的集合里。这就是那条断链的证据。
    lost = set(diff["question_terms_lost"])
    assert {"grammy", "emmy"} <= lost, diff["question_terms_lost"]

    # 编译产物的正文里确实没有这两个词，原文里有。
    assert "Grammy" in diff["source"]["text"]
    assert "Grammy" not in diff["compiled"]["text"]

    # 跨全库验证：编译**不是压缩而是扩写**（中位 2.19 倍，仅 2.7% 净压缩）。
    # 所以丢修饰语是改写策略，不是空间不足 —— 这个比值让读者自己看到这一点。
    assert diff["diff"]["expansion_ratio"] > 1.0

    assert "词法召回" in missed["diff"]["verdict"] or lost


def test_the_bridging_graph_edge_does_not_exist(client: TestClient, eval_layer_id: int):
    """桥接关系没升格成图边 —— 这是那条案例里第三条召回路径断掉的原因。"""
    body = client.get(
        f"/api/layers/eval/{eval_layer_id}/samples/{CASE_SAMPLE}/lineage"
    ).json()
    missed = {entry["doc_id"]: entry for entry in body["gold"]}[GOLD_MISSED]

    edges = missed["lineage"]["edges"]
    # 边只在这两个实体之间打转，没有一条指向 Cyndi Lauper。
    involved = {edge["from_title"] for edge in edges} | {edge["to_title"] for edge in edges}
    assert not any("Cyndi Lauper" in title for title in involved), involved
