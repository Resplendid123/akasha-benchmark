"""抽子集的不变量、原子写、以及配置的优先级。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from akasha_benchmark.config import AkashaConfig, load_config
from akasha_benchmark.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_text,
)
from akasha_benchmark.subset import _largest_remainder, _safe_doc_id, build_subset


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


def _write_normalized(data: Path, dataset: str, samples: list[dict], corpus: list[dict]) -> None:
    """伪造一份归一化产出，供抽样测试当输入。"""
    out = data / "normalized" / dataset
    atomic_write_jsonl(out / "samples.jsonl", samples)
    atomic_write_jsonl(out / "corpus.jsonl", corpus)
    atomic_write_json(
        out / "manifest.json",
        {
            "outputs": {"samples": {"sha256": "s"}, "corpus": {"sha256": "c"}},
            "identity_rules": {"sample_id": "native__id", "corpus_doc_id": "native_idx"},
        },
    )


@pytest.fixture
def normalized(tmp_path: Path) -> Path:
    data = tmp_path / "data"
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
    _write_normalized(data, "hotpotqa", samples, corpus)
    return data


def test_subset_guarantees_full_gold_coverage(normalized: Path):
    """抽子集的验收标准：每条 sample 的 gold 都在子集 corpus 内。"""
    manifest = build_subset(
        "hotpotqa", run_id="r1", seed=1, qa_limit=5, negatives_ratio=1.0, data_dir=normalized
    )
    assert manifest["qa_count"] == 5
    assert manifest["gold_coverage"] == 1.0
    # 5 条样本各 2 篇不重复的 gold，再加等量负样本。
    assert manifest["gold_doc_count"] == 10
    assert manifest["negative_doc_count"] == 10
    assert manifest["corpus_count"] == 20

    subset = normalized / "subsets" / "r1" / "hotpotqa"
    samples = list(read_jsonl(subset / "samples.jsonl"))
    on_disk = {p.stem for p in (subset / "corpus").glob("*.md")}
    for sample in samples:
        assert set(sample["gold_doc_ids"]) <= on_disk


def test_subset_is_deterministic_for_a_seed(normalized: Path):
    """同一个种子必须给出完全相同的子集，换种子则不同。"""
    first = build_subset("hotpotqa", run_id="r1", seed=7, qa_limit=5, data_dir=normalized)
    second = build_subset("hotpotqa", run_id="r1", seed=7, qa_limit=5, data_dir=normalized)
    assert first["corpus_md_sha256"] == second["corpus_md_sha256"]

    different = build_subset("hotpotqa", run_id="r1", seed=8, qa_limit=5, data_dir=normalized)
    assert different["corpus_md_sha256"] != first["corpus_md_sha256"]


def test_subset_removes_stale_markdown_from_an_earlier_sampling(normalized: Path):
    """上次抽样残留的 md 必须清掉，否则入库会把它一起导进库。"""
    build_subset("hotpotqa", run_id="r1", seed=7, qa_limit=10, data_dir=normalized)
    corpus_dir = normalized / "subsets" / "r1" / "hotpotqa" / "corpus"
    stray = corpus_dir / "9999.md"
    stray.write_text("# stale\n\nleftover\n", encoding="utf-8")

    manifest = build_subset("hotpotqa", run_id="r1", seed=7, qa_limit=10, data_dir=normalized)
    assert manifest["stale_md_removed"] == 1
    assert not stray.exists()


def test_subset_markdown_matches_recorded_hash(normalized: Path):
    """manifest 里记的 sha256 要和磁盘上的 md 对得上。"""
    manifest = build_subset("hotpotqa", run_id="r1", seed=3, qa_limit=4, data_dir=normalized)
    corpus_dir = normalized / "subsets" / "r1" / "hotpotqa" / "corpus"
    for doc_id, expected in manifest["corpus_md_sha256"].items():
        actual = sha256_text((corpus_dir / f"{doc_id}.md").read_text(encoding="utf-8"))
        assert actual == expected


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


def test_config_env_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """环境变量优先于配置文件，且按字段类型正确转换。"""
    config_path = tmp_path / "akasha.config.json"
    atomic_write_json(
        config_path,
        {"base_url": "http://from-file", "email": "file@example.com", "custom_key": 1},
    )
    monkeypatch.setenv("AKASHA_BASE_URL", "http://from-env")
    monkeypatch.setenv("AKASHA_REQUEST_INTERVAL_SECONDS", "2.5")

    config = load_config(config_path)
    assert config.base_url == "http://from-env"
    assert config.email == "file@example.com"
    assert config.request_interval_seconds == 2.5
    # 未知键隔离保留，不静默丢弃 —— 拼错的键名要能被发现。
    assert config.extra == {"custom_key": 1}


def test_config_redacts_secrets():
    """写进 manifest 的视图里不能出现明文密钥。"""
    config = AkashaConfig(email="a@b.c", password="hunter2", database_url="postgres://u:p@h/db")
    redacted = config.redacted()
    assert redacted["password"] == "***"
    assert redacted["database_url"] == "***"
    assert "hunter2" not in json.dumps(redacted)


def test_config_requires_credentials():
    """缺凭据时报错要指明该设哪个环境变量。"""
    with pytest.raises(ValueError, match="AKASHA_PASSWORD"):
        AkashaConfig(base_url="http://x", email="a@b.c").require_credentials()


def test_api_url_joining():
    """URL 拼接对两侧多余的斜杠都要容错。"""
    config = AkashaConfig(base_url="http://host:3000/")
    assert config.api("llm-wiki/query") == "http://host:3000/api/llm-wiki/query"
    assert config.api("/pages/import") == "http://host:3000/api/pages/import"
