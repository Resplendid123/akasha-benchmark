"""itfaq：身份规则、无 gold 的指标省略、full_corpus 抽样与本地数据集的下载跳过。"""

from __future__ import annotations

import threading

import pytest

from akasha_benchmark.datasets import (
    DataDependency,
    SubsetStrategy,
    all_adapters,
    get_adapter,
)
from akasha_benchmark.metrics import registry
from akasha_benchmark.stages import compile, download
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



def test_sample_and_doc_ids_come_from_native_fields(itfaq_normalized):
    samples = data_store.samples_of(itfaq_normalized, "itfaq")
    assert [s["sample_id"] for s in samples] == [
        "itfaq:qa_001",
        "itfaq:qa_002",
        "itfaq:qa_003",
    ]
    assert samples[0]["dataset_sample_id"] == "qa_001"
    assert samples[0]["answers"] == ["走 IT 设备申请流程。"]
    assert samples[0]["gold_doc_ids"] == []
    assert samples[0]["metadata"]["answer_chars"] == len("走 IT 设备申请流程。")

    docs = data_store.corpus_of(itfaq_normalized, "itfaq")
    assert [d["doc_id"] for d in docs] == ["doc_001", "doc_002"]


@pytest.mark.parametrize(
    "row,field",
    [
        ({"id": "qa_1", "question": "q", "answer": "   "}, "answer"),
        ({"question": "q", "answer": "a"}, "id"),
    ],
)
def test_parse_row_rejects_missing_required_values(row, field):
    adapter = get_adapter("itfaq")
    with pytest.raises(ValueError, match=field):
        adapter.parse_row(row, 0, None)




def test_retrieval_family_is_omitted_not_zeroed():
    adapter = get_adapter("itfaq")
    assert DataDependency.GOLD_DOCS not in adapter.provides
    omitted = {d.name for d in registry.omitted(adapter.provides)}
    assert {"recall", "ndcg", "mrr", "citation_recall", "citation_precision"} <= omitted
    assert not ({"em", "f1", "answer_correctness", "faithfulness"} & omitted)




def test_full_corpus_keeps_every_doc_while_limiting_qa(itfaq_normalized):
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
    assert stats["docs"] == 2
    assert stats["gold"] == 0 and stats["negatives"] == 0
    docs = compile_store.compile_docs(itfaq_normalized, compile_id)
    assert {d["doc_id"] for d in docs} == {"doc_001", "doc_002"}
    assert not any(d["is_gold"] for d in docs)




def test_markdown_does_not_repeat_existing_heading(itfaq_normalized):
    markdown = compile.markdown_of(itfaq_normalized, "itfaq", "doc_001")
    assert markdown.startswith("# 设备申请说明")
    assert markdown.count("# 设备申请说明") == 1




def test_itfaq_is_not_downloadable():
    assert get_adapter("itfaq").downloadable is False
    assert all(a.downloadable for a in all_adapters() if a.name != "itfaq")


def _no_endpoint(monkeypatch, tmp_path) -> None:

    def boom(_ctx):
        raise AssertionError("本地数据集不该连下载源")

    monkeypatch.setattr(download, "_resolve_endpoint", boom)
    monkeypatch.setattr(download, "DEFAULT_DATASET_DIR", tmp_path)


def test_download_skips_endpoint_for_local_dataset(db, monkeypatch, tmp_path):
    _no_endpoint(monkeypatch, tmp_path)
    monkeypatch.setattr(download, "file_status", _ready_status)
    download.run(_context(db, {"datasets": ["itfaq"]}))


def test_download_reports_missing_local_files(db, monkeypatch, tmp_path):
    _no_endpoint(monkeypatch, tmp_path)
    monkeypatch.setattr(download, "file_status", _missing_status)

    with pytest.raises(RuntimeError, match="校验失败"):
        download.run(_context(db, {"datasets": ["itfaq"]}))
