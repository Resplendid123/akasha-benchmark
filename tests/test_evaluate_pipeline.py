"""End-to-end evaluate on synthetic responses.

Ingest and query need a live Akasha, so this fabricates their outputs — a page_map
and a response jsonl shaped like a real ``/api/llm-wiki/query`` body — and drives
the evaluator over them. That covers the plumbing they feed into: page_id ->
doc_id translation, the question-text cross-check, the answerMode split, and
report rendering.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from akasha_benchmark.evaluate import evaluate_dataset, reports_dir, run
from akasha_benchmark.io_utils import atomic_write_json, atomic_write_jsonl, load_json

DATASET = "hotpotqa"
RUN_ID = "testrun"


def _sample(sample_id: str, question: str, answer: str, gold: list[str], meta: dict) -> dict:
    """造一条规范化样本。"""
    return {
        "dataset": DATASET,
        "sample_id": f"{DATASET}:{sample_id}",
        "dataset_sample_id": sample_id,
        "question": question,
        "answers": [answer],
        "gold_doc_ids": gold,
        "metadata": meta,
    }


def _knowledge_response(retrieved: list[str], cited: list[str], answer: str, snippets: list[dict]):
    """造一个 answerMode=knowledge 的响应体，字段形状照服务端类型定义。"""
    return {
        "answer": answer,
        "answerMode": "knowledge",
        "citations": [{"sourcePageId": p, "title": p, "url": "", "images": []} for p in cited],
        "citationEvidence": [{"sourcePageId": p, "excerpts": ["evidence"]} for p in cited],
        "retrievedSources": [{"sourcePageId": p, "title": p, "url": ""} for p in retrieved],
        "snippets": snippets,
        "warnings": [],
        "retrievalReasons": ["semantic"],
        "budget": {"maxContextLength": 12000, "usedContextLength": 500},
        "completenessNotice": None,
    }


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """搭出子集、入库、查询的产出，让评测有完整输入可读。"""
    data = tmp_path / "data"

    subset = data / "subsets" / RUN_ID / DATASET
    atomic_write_jsonl(
        subset / "samples.jsonl",
        [
            _sample("s1", "Who directed the film?", "Kubrick", ["d1", "d2"], {"type": "bridge"}),
            _sample("s2", "Which city is larger?", "Berlin", ["d3", "d4"], {"type": "comparison"}),
            _sample("s3", "What year did it open?", "1984", ["d5", "d6"], {"type": "bridge"}),
        ],
    )

    # 入库的产出：page id 反查回语料 doc_id。
    atomic_write_jsonl(
        data / "ingest" / RUN_ID / "page_map.jsonl",
        [
            {"dataset": DATASET, "doc_id": f"d{i}", "page_id": f"p{i}", "md_sha256": "x"}
            for i in range(1, 8)
        ],
    )
    atomic_write_json(
        data / "ingest" / RUN_ID / "manifest.json",
        {"spaces": {DATASET: {"id": "space-uuid"}}, "model_configs": {"embedding": "m1"}},
    )

    responses = data / "responses" / RUN_ID
    rows = [
        # 两篇 gold 排在第 1、2 位，只引用了一篇；d2 仅由 graph-neighbor 提供。
        {
            "sample_id": f"{DATASET}:s1",
            "dataset": DATASET,
            "question": "Who directed the film?",
            "requested_at": "2026-09-08T00:00:00Z",
            "latency_ms": 900,
            "http_status": 200,
            "error": None,
            "response": _knowledge_response(
                retrieved=["p1", "p2", "p7"],
                cited=["p1"],
                answer="Kubrick",
                snippets=[
                    {
                        "id": "c1",
                        "title": "t",
                        "text": "x",
                        "retrievalReasons": ["semantic"],
                        "sourceWindows": [{"sourcePageId": "p1"}],
                    },
                    {
                        "id": "c2",
                        "title": "t",
                        "text": "x",
                        "retrievalReasons": ["graph-neighbor"],
                        "sourceWindows": [{"sourcePageId": "p2"}],
                    },
                ],
            ),
        },
        # no_match：retrievedSources 按设计就是空的，答案也是错的。
        {
            "sample_id": f"{DATASET}:s2",
            "dataset": DATASET,
            "question": "Which city is larger?",
            "requested_at": "2026-09-08T00:00:01Z",
            "latency_ms": 400,
            "http_status": 200,
            "error": None,
            "response": {
                "answer": "I could not find this in the knowledge base.",
                "answerMode": "no_match",
                "citations": [],
                "citationEvidence": [],
                "retrievedSources": [],
                "snippets": [],
                "warnings": [],
                "retrievalReasons": [],
                "budget": {},
                "completenessNotice": None,
            },
        },
        # 一条记录在案的 HTTP 失败。它仍然要算作一行。
        {
            "sample_id": f"{DATASET}:s3",
            "dataset": DATASET,
            "question": "What year did it open?",
            "requested_at": "2026-09-08T00:00:02Z",
            "latency_ms": 0,
            "http_status": 500,
            "error": "server error",
            "response": {"message": "Internal server error"},
        },
    ]
    responses.mkdir(parents=True, exist_ok=True)
    (responses / f"{DATASET}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n"
    )
    atomic_write_json(
        responses / "manifest.json",
        {"model_configs": {"embedding": "m1"}, "model_configs_match_ingest": True,
         "total_failures": 1, "score_threshold": None},
    )
    return data


def test_evaluate_splits_knowledge_only_from_all_samples(workspace: Path):
    """全样本与 knowledge 切片必须分开报，两者差值就是生成端拒答的规模。"""
    summary, rows = evaluate_dataset(DATASET, RUN_ID, workspace, (2, 5))
    assert summary["responses_evaluated"] == 3
    assert summary["http_failures"] == 1
    assert summary["knowledge_answer_count"] == 1

    # s1 两篇 gold 都在前 2 位；s2/s3 贡献 0。全样本 1/3，knowledge 切片 1.0。
    assert summary["retrieval"]["recall@2"] == pytest.approx(1 / 3)
    assert summary["retrieval_knowledge_only"]["recall@2"] == pytest.approx(1.0)
    assert summary["retrieval"]["full_coverage@2"] == pytest.approx(1 / 3)

    # F1：只有 s1 答对，且答得与参考逐词相同，所以它那条是 1.0。
    # 这个 fixture 的 s1 是短跨度答案，所以 EM 跟 F1 一样是 1/3 ——
    # 真实运行里 EM 会是 0（散文答案），见 metrics/qa.py。
    assert summary["qa"]["em"] == pytest.approx(1 / 3)
    assert summary["qa"]["em_knowledge_only"] == pytest.approx(1.0)
    assert summary["qa"]["f1"] == pytest.approx(1 / 3)
    assert summary["qa"]["f1_knowledge_only"] == pytest.approx(1.0)
    assert summary["answer_mode_distribution"]["no_match"] == pytest.approx(1 / 3)

    # p7 -> d7 能反查到、只是不是 gold；这里不该有反查不到的 page。
    assert summary["unmapped_page_ids"] == []

    # d2 是图扩展带来的，语义召回没找到它。
    assert summary["multihop"]["graph_exclusive_gold_count"] == pytest.approx(1 / 3)

    by_id = {row["sample_id"]: row for row in rows}
    assert by_id[f"{DATASET}:s3"]["ok"] is False
    assert by_id[f"{DATASET}:s2"]["answer_mode"] == "no_match"


def test_stratification_splits_by_question_type(workspace: Path):
    """hotpotqa 按题型分层，各层的指标独立计算。"""
    summary, _ = evaluate_dataset(DATASET, RUN_ID, workspace, (2,))
    strata = summary["stratified"]["strata"]
    assert summary["stratified"]["key"] == "type"
    assert strata["bridge"]["count"] == 2
    assert strata["comparison"]["count"] == 1
    assert strata["bridge"]["retrieval"]["recall@2"] == pytest.approx(0.5)
    assert strata["comparison"]["retrieval"]["recall@2"] == 0.0


def test_question_text_mismatch_is_fatal(workspace: Path):
    """ID 对得上但 question 变了，说明产物来自不同的数据快照，必须报错。"""
    path = workspace / "responses" / RUN_ID / f"{DATASET}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["question"] = "A different question entirely"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="question text differs"):
        evaluate_dataset(DATASET, RUN_ID, workspace, (2,))


def test_duplicate_sample_id_is_fatal(workspace: Path):
    """一个 sample_id 只能有一行，重复会让指标被算两次。"""
    path = workspace / "responses" / RUN_ID / f"{DATASET}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("".join(f"{line}\n" for line in [*lines, lines[0]]), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate sample_id"):
        evaluate_dataset(DATASET, RUN_ID, workspace, (2,))


def test_run_writes_all_three_artifacts(workspace: Path):
    """metrics.json / per_sample.jsonl / report.md 三样都要产出。"""
    assert run(RUN_ID, [DATASET], workspace, (2, 5)) == 0
    out = reports_dir(RUN_ID, workspace)
    assert (out / "metrics.json").is_file()
    assert (out / "per_sample.jsonl").is_file()

    report = (out / "report.md").read_text(encoding="utf-8")
    # 那条架构说明必须出现在每份报告里，不能只写在计划文档里。
    assert "无效的" in report
    assert "仅 knowledge" in report
    # EM 报出来了，但「为什么预期是 0」必须同时在报告里 ——
    # 否则熟悉 hotpotqa 的读者看到 0.0000 会判断系统坏了。
    assert "answer EM：" in report
    assert "Exact Match 预期就是 0.0000" in report
    assert "答案**形状**的探针" in report

    metrics = load_json(out / "metrics.json")
    assert metrics["datasets"][0]["dataset"] == DATASET
    assert "em" in metrics["datasets"][0]["qa"]


def test_report_tables_have_matching_column_counts(workspace: Path):
    """每张表的表头、分隔行与数据行列数必须一致。

    删掉 EM 列时只改表头不改数据行（或反之）不会报错，只会让 Markdown 表格错行，
    而错行的表格照样是「一份报告」—— 看起来正常，读出来的数却对错了列。
    """
    assert run(RUN_ID, [DATASET], workspace, (2, 5)) == 0
    report = (reports_dir(RUN_ID, workspace) / "report.md").read_text(encoding="utf-8")

    lines = report.splitlines()
    tables = 0
    for index, line in enumerate(lines):
        # 分隔行（| --- | --- |）定位一张表：它上面是表头，下面是数据行。
        if not line.startswith("|") or set(line.replace("|", "").replace("-", "").strip()):
            continue
        tables += 1
        width = line.count("|")
        assert lines[index - 1].count("|") == width, f"header/separator mismatch:\n{lines[index-1]}\n{line}"
        for row in lines[index + 1 :]:
            if not row.startswith("|"):
                break
            assert row.count("|") == width, f"row has {row.count('|')} pipes, header has {width}:\n{row}"
    assert tables >= 2, f"expected the retrieval and stratified tables, found {tables}"
