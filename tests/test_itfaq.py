"""itfaq：身份规则、无 gold 的指标省略、full_corpus 抽样与本地数据集的下载跳过。"""

from __future__ import annotations

import threading

import pytest

from akasha_benchmark.datasets import (
    CorpusDoc,
    DataDependency,
    SubsetStrategy,
    all_adapters,
    get_adapter,
)
from akasha_benchmark.metrics import registry
from akasha_benchmark.stages import compile, download, normalize
from akasha_benchmark.store import compile_store, data_store
from akasha_benchmark.task import TaskContext


def _context(connection, params: dict) -> TaskContext:
    connection.execute(
        "INSERT INTO task (id, stage, status, params_json, created_at) "
        "VALUES (1, 'download', 'running', '{}', 'now')"
    )
    connection.commit()
    return TaskContext(
        task_id=1,
        stage="download",
        params=params,
        connection=connection,
        pause_event=threading.Event(),
    )


def _ready_status() -> list[dict[str, object]]:
    return [
        {"dataset": "itfaq", "kind": k, "file": f, "present": True, "error": None}
        for k, f in (("qa", "itfaq.json"), ("corpus", "itfaq_corpus.json"))
    ]


def _missing_status() -> list[dict[str, object]]:
    return [
        {"dataset": "itfaq", "kind": k, "file": f, "present": False, "error": None}
        for k, f in (("qa", "itfaq.json"), ("corpus", "itfaq_corpus.json"))
    ]

# ------------------------------------------------------------ 归一化与身份


def test_sample_and_doc_ids_come_from_native_fields(itfaq_normalized):
    """doc_id 取自语料的 id 字段，不是行号 —— doc_001 当导入文件名才追得回来。"""
    samples = data_store.samples_of(itfaq_normalized, "itfaq")
    assert [s["sample_id"] for s in samples] == [
        "itfaq:qa_001",
        "itfaq:qa_002",
        "itfaq:qa_003",
    ]
    assert samples[0]["dataset_sample_id"] == "qa_001"
    # answer 是单串，归一化成单元素参考答案集。
    assert samples[0]["answers"] == ["走 IT 设备申请流程。"]
    assert samples[0]["gold_doc_ids"] == []
    assert samples[0]["metadata"]["answer_chars"] == len("走 IT 设备申请流程。")

    docs = data_store.corpus_of(itfaq_normalized, "itfaq")
    assert [d["doc_id"] for d in docs] == ["doc_001", "doc_002"]


def test_normalize_validation_passes_without_gold(itfaq_normalized):
    """无 gold 不是缺陷：验收只查悬空 gold，空 gold 集不该报问题。"""
    assert normalize.validate_dataset(itfaq_normalized, "itfaq") == []


def test_parse_row_rejects_blank_answer():
    adapter = get_adapter("itfaq")
    with pytest.raises(ValueError, match="answer"):
        adapter.parse_row({"id": "qa_1", "question": "q", "answer": "   "}, 0, None)


def test_parse_row_rejects_missing_id():
    adapter = get_adapter("itfaq")
    with pytest.raises(ValueError, match="id"):
        adapter.parse_row({"question": "q", "answer": "a"}, 0, None)


# ------------------------------------------------------------ 指标省略


def test_retrieval_family_is_omitted_not_zeroed():
    """无 gold：检索与引用两族必须省略而不是记 0，judge 与 QA 族仍成立。"""
    adapter = get_adapter("itfaq")
    assert DataDependency.GOLD_DOCS not in adapter.provides
    omitted = {d.name for d in registry.omitted(adapter.provides)}
    assert {"recall", "ndcg", "mrr", "citation_recall", "citation_precision"} <= omitted
    # 有参考答案，所以 QA 族与 answer_correctness 算得出来。
    assert not ({"em", "f1", "answer_correctness", "faithfulness"} & omitted)


def test_requesting_retrieval_metric_raises():
    adapter = get_adapter("itfaq")
    with pytest.raises(registry.DependencyError):
        registry.require("itfaq", adapter.provides, "recall")


# ------------------------------------------------------------ full_corpus 抽样


def test_full_corpus_keeps_every_doc_while_limiting_qa(itfaq_normalized):
    """语料整份导入，QA 按 qa_limit 抽 —— 没有「问题→文档」映射就不能抽语料。"""
    assert get_adapter("itfaq").subset_strategy is SubsetStrategy.FULL_CORPUS
    compile_id = compile_store.create_compile_run(
        itfaq_normalized, run_id="r", datasets=["itfaq"], seed=7, qa_limit=1, negatives_ratio=1.0
    )
    itfaq_normalized.commit()
    stats = compile.build_subset(
        itfaq_normalized, compile_id, "itfaq", seed=7, qa_limit=1, negatives_ratio=1.0
    )

    assert stats["strategy"] == "full_corpus"
    assert stats["samples"] == 1
    # qa_limit 只压缩 QA，语料仍是全部 2 篇。
    assert stats["docs"] == 2
    assert stats["gold"] == 0 and stats["negatives"] == 0
    docs = compile_store.compile_docs(itfaq_normalized, compile_id)
    assert {d["doc_id"] for d in docs} == {"doc_001", "doc_002"}
    assert not any(d["is_gold"] for d in docs)


def test_negatives_ratio_does_not_change_corpus(itfaq_normalized):
    """全量导入下 negatives_ratio 失效，两次取值必须给出同一份语料。"""

    def docs_for(ratio: float, run_id: str) -> set[str]:
        compile_id = compile_store.create_compile_run(
            itfaq_normalized,
            run_id=run_id,
            datasets=["itfaq"],
            seed=7,
            qa_limit=2,
            negatives_ratio=ratio,
        )
        itfaq_normalized.commit()
        compile.build_subset(
            itfaq_normalized, compile_id, "itfaq", seed=7, qa_limit=2, negatives_ratio=ratio
        )
        return {d["doc_id"] for d in compile_store.compile_docs(itfaq_normalized, compile_id)}

    assert docs_for(0.0, "a") == docs_for(3.0, "b") == {"doc_001", "doc_002"}


def test_subset_is_seed_stable(itfaq_normalized):
    def samples_for(run_id: str) -> list[str]:
        compile_id = compile_store.create_compile_run(
            itfaq_normalized,
            run_id=run_id,
            datasets=["itfaq"],
            seed=11,
            qa_limit=2,
            negatives_ratio=1.0,
        )
        itfaq_normalized.commit()
        compile.build_subset(
            itfaq_normalized, compile_id, "itfaq", seed=11, qa_limit=2, negatives_ratio=1.0
        )
        return [s["sample_id"] for s in compile_store.compile_samples(itfaq_normalized, compile_id)]

    assert samples_for("a") == samples_for("b")


# ------------------------------------------------------------ 语料渲染


def test_markdown_does_not_repeat_existing_heading(itfaq_normalized):
    """itfaq 语料自带 H1，再加一遍会让标题重复进 chunk 与 embedding。"""
    markdown = compile.markdown_of(itfaq_normalized, "itfaq", "doc_001")
    assert markdown.startswith("# 设备申请说明")
    assert markdown.count("# 设备申请说明") == 1


def test_markdown_still_adds_heading_when_absent():
    """正文没有 H1 时照旧补上 —— 四组现有数据集走的是这条。"""
    doc = CorpusDoc(doc_id="0", title="Rita Moreno", text="Rita Moreno won a Grammy.")
    assert doc.to_markdown() == "# Rita Moreno\n\nRita Moreno won a Grammy.\n"


def test_markdown_heading_match_is_exact_first_line():
    """musique 有以 `# ` 开头的表格片段，首行不等于 `# {title}`，必须照旧加标题。"""
    doc = CorpusDoc(doc_id="0", title="North Korea", text="# Name Took office Left office")
    assert doc.to_markdown().startswith("# North Korea\n\n# Name Took office")


# ------------------------------------------------------------ 本地数据集不下载


def test_itfaq_is_not_downloadable():
    assert get_adapter("itfaq").downloadable is False
    assert all(a.downloadable for a in all_adapters() if a.name != "itfaq")


def _no_endpoint(monkeypatch, tmp_path) -> None:
    """站点解析一旦被调用就让测试失败。"""

    def boom(_ctx):
        raise AssertionError("本地数据集不该连下载源")

    monkeypatch.setattr(download, "_resolve_endpoint", boom)
    monkeypatch.setattr(download, "DEFAULT_DATASET_DIR", tmp_path)


def test_download_skips_endpoint_for_local_dataset(db, monkeypatch, tmp_path):
    _no_endpoint(monkeypatch, tmp_path)
    monkeypatch.setattr(download, "file_status", _ready_status)
    download.run(_context(db, {"datasets": ["itfaq"]}))


def test_download_reports_missing_local_files(db, monkeypatch, tmp_path):
    """本地文件缺失时报错，而不是让用户去点下载。"""
    _no_endpoint(monkeypatch, tmp_path)
    monkeypatch.setattr(download, "file_status", _missing_status)

    with pytest.raises(RuntimeError, match="校验失败"):
        download.run(_context(db, {"datasets": ["itfaq"]}))
