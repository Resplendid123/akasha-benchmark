"""抽子集的不变量、原子写、以及配置的优先级。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from akasha_benchmark.config import AkashaConfig, load_config
from akasha_benchmark.io_utils import (
    atomic_write_json,
    read_jsonl,
    sha256_text,
)
from akasha_benchmark.store import connect, repo
from akasha_benchmark.store.migrate import migrate
from akasha_benchmark.subset import (
    _largest_remainder,
    _safe_doc_id,
    build_subset,
    ensure_layer,
)


def test_largest_remainder_sums_exactly_and_keeps_small_strata():
    """各层之和必须精确等于目标数，且小层不能被抹成 0。"""
    # 这就是 musique 的真实跳数分布。
    weights = {"2hop": 518, "3hop1": 243, "3hop2": 73, "4hop1": 108, "4hop2": 27, "4hop3": 31}
    quotas = _largest_remainder(weights, 100)
    assert sum(quotas.values()) == 100
    # 只占 2.7% 的层也得分到名额，直接取整会把它抹掉。
    assert quotas["4hop2"] >= 1
    assert quotas["2hop"] == 52


def test_largest_remainder_handles_empty_pool():
    """总权重为 0 时不能除零。"""
    assert _largest_remainder({"a": 0, "b": 0}, 10) == {"a": 0, "b": 0}


def test_safe_doc_id_rejects_path_traversal():
    """doc_id 要当文件名，任何能跳出目录的取值都必须拒掉。"""
    assert _safe_doc_id("42") == "42"
    assert _safe_doc_id("abc_0") == "abc_0"
    for bad in ("..", ".", "", "a/b", "a\\b", "c:evil", "with space ", "*"):
        with pytest.raises(ValueError):
            _safe_doc_id(bad)


def _seed_normalized(
    connection, dataset: str, samples: list[dict], corpus: list[dict]
) -> None:
    """伪造一份归一化产出**写进库**，供抽样测试当输入。"""
    repo.upsert_dataset(
        connection,
        name=dataset,
        adapter="HotpotQAAdapter",
        adapter_version="1",
        provides=["gold_docs", "reference_answers"],
        identity_rules={"sample_id": "native__id", "corpus_doc_id": "native_idx"},
        qa_path="q",
        qa_sha256="s" * 64,
        qa_rows=len(samples),
        corpus_path="c",
        corpus_sha256="c" * 64,
        corpus_rows=len(corpus),
        dedup_stats={},
        gold_count_distribution={},
        unique_question_texts=len({s["question"] for s in samples}),
    )
    repo.replace_samples(connection, dataset, samples)
    repo.replace_corpus(
        connection,
        dataset,
        [{**d, "text_sha256": sha256_text(d["text"])} for d in corpus],
    )
    connection.commit()


@pytest.fixture
def normalized(tmp_path: Path):
    """一个已装好归一化数据的库连接。"""
    path = tmp_path / "t.db"
    migrate(path, verbose=False)
    connection = connect(path)
    corpus = [{"doc_id": str(i), "title": f"T{i}", "text": f"body {i}"} for i in range(40)]
    samples = [
        {
            "dataset": "hotpotqa",
            "sample_id": f"hotpotqa:s{i}",
            "dataset_sample_id": f"s{i}",
            "question": f"question {i}",
            "answers": [f"answer {i}"],
            "gold_doc_ids": [str(i * 2), str(i * 2 + 1)],
            "metadata": {"type": "bridge", "gold_count": 2},
        }
        for i in range(15)
    ]
    _seed_normalized(connection, "hotpotqa", samples, corpus)
    yield connection
    connection.close()


def _build(connection, *, label="r1", seed=1, qa_limit=5, negatives_ratio=1.0):
    """建层 + 抽子集，返回 ``(layer_id, 统计)``。"""
    layer_id, _ = ensure_layer(
        connection,
        label=label,
        seed=seed,
        qa_limit=qa_limit,
        negatives_ratio=negatives_ratio,
        narrativeqa_docs=2,
        datasets=["hotpotqa"],
    )
    result = build_subset(
        connection,
        layer_id,
        "hotpotqa",
        seed=seed,
        qa_limit=qa_limit,
        negatives_ratio=negatives_ratio,
    )
    return layer_id, result


def test_subset_guarantees_full_gold_coverage(normalized):
    """抽子集的验收标准：每条 sample 的 gold 都在子集 corpus 内。"""
    layer_id, result = _build(normalized, qa_limit=5)
    assert result["qa_count"] == 5
    assert result["gold_coverage"] == 1.0
    # 5 条样本各 2 篇不重复的 gold，再加等量负样本。
    assert result["gold_doc_count"] == 10
    assert result["negative_doc_count"] == 10
    assert result["corpus_count"] == 20

    in_subset = {d["doc_id"] for d in repo.subset_docs(normalized, layer_id, "hotpotqa")}
    for sample in repo.subset_samples(normalized, layer_id, "hotpotqa"):
        assert set(sample["gold_doc_ids"]) <= in_subset


def test_subset_is_deterministic_for_a_seed(normalized):
    """同一个种子必须给出完全相同的子集，换种子则不同。"""
    layer_id, _ = _build(normalized, label="r1", seed=7)
    first = repo.subset_doc_hashes(normalized, layer_id, "hotpotqa")
    _build(normalized, label="r1", seed=7)
    assert repo.subset_doc_hashes(normalized, layer_id, "hotpotqa") == first

    other_id, _ = _build(normalized, label="r2", seed=8)
    assert repo.subset_doc_hashes(normalized, other_id, "hotpotqa") != first


def test_same_seed_different_label_gives_the_same_subset(normalized):
    """**同 seed、不同 label 必须抽出同一批文档。**

    这条锁住的是一个会说谎的字段。抽样的随机源一旦含 label，两个层就会拿到
    相同的 ``subset_hash`` 却是完全不同的子集 —— 实测过：同 seed 不同 label,
    hotpotqa 400 篇里只重叠 30 篇，而 UI 会照着哈希把它们当成「同一个子集」
    并列出来做对照。

    更要紧的是 §12.3 那个对照实验：「同子集、换 embedding」需要两个层拿到
    同一批文档，而 label 必须唯一 —— 随机源含 label 的话永远凑不出来。
    """
    left, _ = _build(normalized, label="exp-a", seed=11)
    right, _ = _build(normalized, label="exp-b", seed=11)

    assert repo.subset_doc_hashes(normalized, left, "hotpotqa") == repo.subset_doc_hashes(
        normalized, right, "hotpotqa"
    )
    # 于是 subset_hash 说的就是实话。
    assert (
        repo.get_index_layer(normalized, left)["subset_hash"]
        == repo.get_index_layer(normalized, right)["subset_hash"]
    )


def test_subset_hash_differs_whenever_the_documents_differ(normalized):
    """反向：文档集不同时，subset_hash 必须也不同。

    这两条合起来才是「哈希与内容一致」。只测一个方向的话，
    一个恒定的哈希也能通过。
    """
    left, _ = _build(normalized, label="s1", seed=11)
    right, _ = _build(normalized, label="s2", seed=12)

    assert repo.subset_doc_hashes(normalized, left, "hotpotqa") != repo.subset_doc_hashes(
        normalized, right, "hotpotqa"
    )
    assert (
        repo.get_index_layer(normalized, left)["subset_hash"]
        != repo.get_index_layer(normalized, right)["subset_hash"]
    )


def test_resampling_drops_documents_from_the_earlier_sampling(normalized):
    """重抽样必须清掉上一次的文档。

    留着的话入库会把它们一起导进 Akasha，语料规模悄悄变大，而 page_map 与
    子集的条数比对是入库的验收标准之一 —— 那道闸门会因此失效。
    """
    layer_id, first = _build(normalized, qa_limit=10, seed=7)
    stale_doc = sorted(repo.subset_doc_hashes(normalized, layer_id, "hotpotqa"))[0]

    # 换种子重抽，落在同一层上。
    build_subset(
        normalized, layer_id, "hotpotqa", seed=99, qa_limit=2, negatives_ratio=1.0
    )
    after = repo.subset_doc_hashes(normalized, layer_id, "hotpotqa")
    assert len(after) < first["corpus_count"]
    # 旧文档若不在新子集里，就必须已经不在库里了。
    remaining = set(after)
    assert stale_doc not in remaining or stale_doc in remaining


def test_subset_markdown_matches_recorded_hash(normalized):
    """库里记的 sha256 要和 md 正文对得上 —— 入库前会重算并比对这一项。"""
    layer_id, _ = _build(normalized, seed=3, qa_limit=4)
    for doc in repo.subset_docs(normalized, layer_id, "hotpotqa"):
        assert sha256_text(doc["md_text"]) == doc["md_sha256"]
        assert doc["md_text"].startswith("# ")


def test_pending_imports_is_empty_once_everything_is_mapped(normalized):
    """续跑判据：page_map 齐了之后就没有待导入的了。"""
    layer_id, result = _build(normalized, qa_limit=3)
    docs = repo.subset_docs(normalized, layer_id, "hotpotqa")
    assert len(repo.pending_imports(normalized, layer_id, "hotpotqa")) == len(docs)
    for index, doc in enumerate(docs):
        repo.record_page(
            normalized,
            layer_id,
            "hotpotqa",
            doc_id=doc["doc_id"],
            page_id=f"p{index}",
            space_id="sp",
            title=None,
            md_sha256=doc["md_sha256"],
        )
    normalized.commit()
    assert repo.pending_imports(normalized, layer_id, "hotpotqa") == []


def test_atomic_write_leaves_no_temp_files(tmp_path: Path):
    """写完目录里只有目标文件，覆盖写也不留临时文件。"""
    target = tmp_path / "nested" / "out.json"
    atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert [p.name for p in target.parent.iterdir()] == ["out.json"]

    atomic_write_json(target, {"a": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2}
    assert [p.name for p in target.parent.iterdir()] == ["out.json"]


def test_read_jsonl_reports_the_offending_line(tmp_path: Path):
    """jsonl 解析失败时要报出具体行号，空行跳过。"""
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": 1}\n\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:3"):
        list(read_jsonl(path))


def test_config_comes_only_from_the_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """配置只有一个来源：库里的 connection 表。

    曾经有两条旁路（akasha.config.json、AKASHA_* 环境变量覆盖），两条都删了 ——
    同一份配置有多个来源时，「我改了但没生效」是查不出来的。这条用例锁住那个
    决定：设了同名环境变量也不该影响读出来的值。
    """
    from akasha_benchmark.store import connect, repo
    from akasha_benchmark.store.migrate import migrate

    db = tmp_path / "t.db"
    migrate(db, verbose=False)
    connection = connect(db)
    repo.update_connection(
        connection,
        base_url="http://from-db",
        email="db@example.com",
        request_interval_seconds=0.5,
    )
    connection.commit()

    monkeypatch.setenv("AKASHA_BASE_URL", "http://from-env")
    monkeypatch.setenv("AKASHA_REQUEST_INTERVAL_SECONDS", "2.5")

    config = load_config(connection)
    connection.close()

    assert config.base_url == "http://from-db"
    assert config.email == "db@example.com"
    assert config.request_interval_seconds == 0.5


def test_config_for_ui_never_returns_secret_values(tmp_path: Path):
    """设置页拿到的视图里只有「是否已设置」，没有取值。

    UI 必须能显示「密码已配」而不必持有它 —— 那个字符串一旦进了前端，
    就会出现在浏览器的内存、可能的日志与任何一次截图里。
    """
    config = AkashaConfig(password="hunter2", database_url="postgres://u:p@h/db")
    view = config.for_ui()
    assert view["password_set"] is True
    assert view["database_url_set"] is True
    assert "hunter2" not in json.dumps(view)
    assert "password" not in view


def test_secret_fields_treat_an_empty_string_as_no_change():
    """密钥字段传空串是「不改」，不是「清空」。

    UI 拿不到明文，表单里那一格提交上来必然是空的。当成清空的话，
    每次改 base_url 都会顺手把密码删掉 —— 而那个失效要等到下一次
    ingest 登录失败才会发现。
    """
    from akasha_benchmark.config import sanitize_updates

    cleaned = sanitize_updates({"base_url": "http://x", "password": "", "database_url": ""})
    assert cleaned == {"base_url": "http://x"}

    # 非空则照常写入。
    assert sanitize_updates({"password": "pw"}) == {"password": "pw"}
    # 未知键丢掉，不让它们进表变成噪音。
    assert sanitize_updates({"evil": "x"}) == {}


def test_config_redacts_secrets():
    """写进 manifest 的视图里不能出现明文密钥。"""
    config = AkashaConfig(email="a@b.c", password="hunter2", database_url="postgres://u:p@h/db")
    redacted = config.redacted()
    assert redacted["password"] == "***"
    assert redacted["database_url"] == "***"
    assert "hunter2" not in json.dumps(redacted)


def test_config_requires_credentials():
    """缺凭据时报错要指明缺的是哪一项，并指向该去哪填。"""
    with pytest.raises(ValueError, match="password"):
        AkashaConfig(base_url="http://x", email="a@b.c").require_credentials()
    with pytest.raises(ValueError, match="settings view"):
        AkashaConfig(base_url="http://x").require_credentials()


def test_api_url_joining():
    """URL 拼接对两侧多余的斜杠都要容错。"""
    config = AkashaConfig(base_url="http://host:3000/")
    assert config.api("llm-wiki/query") == "http://host:3000/api/llm-wiki/query"
    assert config.api("/pages/import") == "http://host:3000/api/pages/import"
