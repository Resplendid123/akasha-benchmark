"""End-to-end evaluate on synthetic responses, driven off the database.

Ingest and query need a live Akasha, so this fabricates their outputs — a page_map
and response rows shaped like a real ``/api/llm-wiki/query`` body — writes them into
the store, and drives the evaluator over them. That covers the plumbing they feed
into: page_id -> doc_id translation, the question-text cross-check, the answerMode
split, dependency-driven omission, and report rendering.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from akasha_benchmark.evaluate import (
    ensure_eval_layer,
    evaluate_dataset,
    export_report,
    reports_dir,
    run,
)
from akasha_benchmark.io_utils import load_json, sha256_text
from akasha_benchmark.store import connect, repo
from akasha_benchmark.store.migrate import migrate

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


def _response_rows() -> list[dict]:
    """三条响应：knowledge 命中、no_match、HTTP 失败。"""
    return [
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


LAYER = "testlayer"
QUERY_LAYER = "testlayer-query"


def _seed(connection: sqlite3.Connection) -> tuple[int, int]:
    """把子集、page_map 与响应装进库，返回 ``(index_layer_id, query_layer_id)``。"""
    samples = [
        _sample("s1", "Who directed the film?", "Kubrick", ["d1", "d2"], {"type": "bridge"}),
        _sample("s2", "Which city is larger?", "Berlin", ["d3", "d4"], {"type": "comparison"}),
        _sample("s3", "What year did it open?", "1984", ["d5", "d6"], {"type": "bridge"}),
    ]
    repo.upsert_dataset(
        connection,
        name=DATASET,
        adapter="HotpotQAAdapter",
        adapter_version="1",
        provides=["gold_docs", "reference_answers"],
        identity_rules={},
        qa_path="q",
        qa_sha256="0" * 64,
        qa_rows=len(samples),
        corpus_path="c",
        corpus_sha256="1" * 64,
        corpus_rows=7,
        dedup_stats={},
        gold_count_distribution={"2": 3},
        unique_question_texts=3,
    )
    repo.replace_samples(connection, DATASET, samples)
    repo.replace_corpus(
        connection,
        DATASET,
        [
            {"doc_id": f"d{i}", "title": f"T{i}", "text": f"body {i}", "text_sha256": "x" * 64}
            for i in range(1, 8)
        ],
    )
    layer_id = repo.create_index_layer(
        connection,
        label=LAYER,
        subset_hash="sh",
        config_hash="ih",
        seed=1,
        qa_limit=3,
        negatives_ratio=1.0,
        narrativeqa_docs=2,
    )
    repo.upsert_index_layer_dataset(
        connection,
        layer_id,
        DATASET,
        strategy="uniform_qa_then_gold_corpus",
        qa_count=len(samples),
        corpus_count=7,
        gold_doc_count=6,
        negative_doc_count=1,
        strata={},
        normalized_qa_sha256="0" * 64,
        normalized_corpus_sha256="1" * 64,
    )
    docs = []
    for i in range(1, 8):
        markdown = f"# T{i}\n\nbody {i}\n"
        docs.append(
            {
                "doc_id": f"d{i}",
                "md_text": markdown,
                "md_sha256": sha256_text(markdown),
                "is_gold": i <= 6,
            }
        )
    repo.replace_subset(connection, layer_id, DATASET, [s["sample_id"] for s in samples], docs)
    # 入库的产出：page id 反查回语料 doc_id。
    for i in range(1, 8):
        repo.record_page(
            connection,
            layer_id,
            DATASET,
            doc_id=f"d{i}",
            page_id=f"p{i}",
            space_id="space-uuid",
            title=f"T{i}",
            md_sha256=docs[i - 1]["md_sha256"],
        )
    # 前置闸门：质量四项全 0、编译已终态，否则 readiness 会拦住查询阶段。
    repo.record_quality_gate(
        connection,
        layer_id,
        gates={
            "missingChunkPageCount": 0,
            "missingEmbeddingPageCount": 0,
            "missingSourcePageCount": 0,
            "stalePageCount": 0,
        },
        report={},
    )
    repo.update_index_layer(connection, layer_id, quality_passed=1)
    repo.record_compile_run(
        connection,
        layer_id,
        accepted_run_count=1,
        coalesced_run_count=0,
        status_counts={"succeeded": 1},
        terminal={"succeeded": 1},
        timed_out=False,
        requested_at="2026-09-08T00:00:00Z",
        finished_at="2026-09-08T00:10:00Z",
    )
    repo.set_space(
        connection, layer_id, DATASET, space_id="space-uuid", space_slug="sp", space_reused=False
    )

    query_layer_id = repo.create_query_layer(
        connection,
        index_layer_id=layer_id,
        label=QUERY_LAYER,
        config_hash="qh",
        score_threshold=None,
        concurrency=1,
        request_interval_seconds=0.5,
        model_configs={"configs": [{"feature": "embedding", "model": "m1"}]},
        model_configs_match_index=True,
        allow_config_drift=False,
    )
    for row in _response_rows():
        repo.record_response(
            connection,
            query_layer_id,
            sample_id=row["sample_id"],
            dataset=row["dataset"],
            question=row["question"],
            requested_at=row["requested_at"],
            latency_ms=row["latency_ms"],
            http_status=row["http_status"],
            error=row["error"],
            response=row["response"],
        )
    connection.commit()
    return layer_id, query_layer_id


class Workspace:
    """库连接 + 路径。sqlite3.Connection 挂不了自定义属性，所以包一层。"""

    def __init__(self, connection: sqlite3.Connection, db_path: Path, data_dir: Path) -> None:
        self.connection = connection
        self.db_path = db_path
        self.data_dir = data_dir


@pytest.fixture
def workspace(tmp_path: Path):
    """一个装好三层输入的库。"""
    path = tmp_path / "t.db"
    migrate(path, verbose=False)
    connection = connect(path)
    _seed(connection)
    yield Workspace(connection, path, tmp_path / "out")
    connection.close()


def _ids(connection: sqlite3.Connection) -> tuple[int, int, int]:
    """``(eval_layer_id, query_layer_id, index_layer_id)``，评测层按需新建。"""
    query_layer = repo.query_layer_by_label(connection, QUERY_LAYER)
    eval_layer_id, _ = ensure_eval_layer(
        connection,
        query_layer_id=int(query_layer["id"]),
        label=f"{QUERY_LAYER}-eval",
        ks=(2, 5),
        metrics=["recall"],
    )
    return eval_layer_id, int(query_layer["id"]), int(query_layer["index_layer_id"])


def test_evaluate_splits_knowledge_only_from_all_samples(workspace):
    """全样本与 knowledge 切片必须分开报，两者差值就是生成端拒答的规模。"""
    connection = workspace.connection
    eval_id, query_id, index_id = _ids(connection)
    summary = evaluate_dataset(connection, eval_id, query_id, index_id, DATASET, (2, 5))

    assert summary["responses_evaluated"] == 3
    assert summary["http_failures"] == 1
    assert summary["knowledge_answer_count"] == 1
    assert summary["has_gold"] is True

    # s1 两篇 gold 都在前 2 位；s2/s3 贡献 0。全样本 1/3，knowledge 切片 1.0。
    assert summary["overall"]["recall@2"] == pytest.approx(1 / 3)
    assert summary["knowledge_only"]["recall@2"] == pytest.approx(1.0)
    assert summary["overall"]["full_coverage@2"] == pytest.approx(1 / 3)

    # F1：只有 s1 答对，且答得与参考逐词相同，所以它那条是 1.0。
    # 这个 fixture 的 s1 是短跨度答案，所以 EM 跟 F1 一样是 1/3 ——
    # 真实运行里 EM 会是 0（散文答案），见 metrics/qa.py。
    assert summary["overall"]["em"] == pytest.approx(1 / 3)
    assert summary["knowledge_only"]["em"] == pytest.approx(1.0)
    assert summary["overall"]["f1"] == pytest.approx(1 / 3)
    assert summary["knowledge_only"]["f1"] == pytest.approx(1.0)
    assert summary["answer_mode_distribution"]["no_match"] == pytest.approx(1 / 3)

    # p7 -> d7 能反查到、只是不是 gold；这里不该有反查不到的 page。
    assert summary["unmapped_page_ids"] == []
    # d2 是图扩展带来的，语义召回没找到它。
    assert summary["overall"]["graph_exclusive_gold_count"] == pytest.approx(1 / 3)

    rows = {r["sample_id"]: r for r in repo.sample_evals(connection, eval_id)}
    assert rows[f"{DATASET}:s3"]["ok"] == 0
    assert rows[f"{DATASET}:s2"]["answer_mode"] == "no_match"
    # 扁平指标表要能按名字直接查，UI 的筛选排序靠它。
    assert repo.sample_metrics_of(connection, eval_id, f"{DATASET}:s1")["recall@2"] == 1.0


def test_metric_summary_stores_both_scopes(workspace):
    """两份口径都要落库：只存一份的话，差值就再也算不出来了。"""
    connection = workspace.connection
    eval_id, query_id, index_id = _ids(connection)
    evaluate_dataset(connection, eval_id, query_id, index_id, DATASET, (2,))

    scopes = {row["scope"] for row in repo.metric_summaries(connection, eval_id, DATASET)}
    assert {"overall", "knowledge_only"} <= scopes
    assert any(s.startswith("stratum:") for s in scopes)


def test_stratification_splits_by_question_type(workspace):
    """hotpotqa 按题型分层，各层的指标独立计算。"""
    connection = workspace.connection
    eval_id, query_id, index_id = _ids(connection)
    summary = evaluate_dataset(connection, eval_id, query_id, index_id, DATASET, (2,))

    strata = summary["stratified"]["strata"]
    assert summary["stratified"]["key"] == "type"
    assert strata["bridge"]["count"] == 2
    assert strata["comparison"]["count"] == 1
    assert strata["bridge"]["metrics"]["recall@2"] == pytest.approx(0.5)
    assert strata["comparison"]["metrics"]["recall@2"] == 0.0


def test_question_text_mismatch_is_fatal(workspace):
    """ID 对得上但 question 变了，说明子集被原地重建过，必须报错。

    外键保证了样本归属，抓不到这一种 —— 所以文本比对不能因为有外键就省掉。
    """
    connection = workspace.connection
    eval_id, query_id, index_id = _ids(connection)
    connection.execute(
        "UPDATE query_response SET question = ? WHERE sample_id = ?",
        ("A different question entirely", f"{DATASET}:s1"),
    )
    connection.commit()

    with pytest.raises(ValueError, match="question text differs"):
        evaluate_dataset(connection, eval_id, query_id, index_id, DATASET, (2,))


def test_duplicate_sample_id_is_impossible_by_construction(workspace):
    """一个 sample_id 只能有一行 —— 主键保证，不靠运行时检查。

    原来是追加 JSONL，重复行只能靠 evaluate 里的 seen 集合拦；现在
    ``(query_layer_id, sample_id)`` 是主键，重复插入直接违反约束。
    """
    connection = workspace.connection
    query_layer = repo.query_layer_by_label(connection, QUERY_LAYER)
    with pytest.raises(sqlite3.IntegrityError):
        repo.record_response(
            connection,
            int(query_layer["id"]),
            sample_id=f"{DATASET}:s1",
            dataset=DATASET,
            question="Who directed the film?",
            requested_at="2026-09-08T00:00:09Z",
            latency_ms=1,
            http_status=200,
            error=None,
            response={"answerMode": "knowledge"},
        )
    connection.rollback()


def test_unknown_sample_id_is_fatal(workspace):
    """响应里有子集之外的 sample_id，说明查询层挂的不是这个子集。"""
    connection = workspace.connection
    eval_id, query_id, index_id = _ids(connection)
    connection.execute(
        "INSERT INTO query_response (query_layer_id, sample_id, dataset, question,"
        " requested_at, latency_ms, http_status, error, response_json, answer_mode)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (query_id, f"{DATASET}:stranger", DATASET, "Who?", "2026-09-08T00:00:09Z",
         1, 200, None, "{}", "knowledge"),
    )
    connection.commit()

    with pytest.raises(ValueError, match="not in the subset"):
        evaluate_dataset(connection, eval_id, query_id, index_id, DATASET, (2,))


def test_run_writes_metrics_to_the_database_and_exports_on_request(workspace):
    """指标进库；导出是可选的一步，产出那三个文件。"""
    connection = workspace.connection
    assert (
        run(QUERY_LAYER, [DATASET], workspace.db_path, (2, 5), export=True,
            data_dir=workspace.data_dir)
        == 0
    )

    eval_layer = repo.eval_layer_by_label(connection, f"{QUERY_LAYER}-eval")
    assert eval_layer is not None
    eval_id = int(eval_layer["id"])
    assert len(repo.sample_evals(connection, eval_id)) == 3
    assert repo.dataset_evals(connection, eval_id)[0]["responses_evaluated"] == 3

    out = reports_dir(f"{QUERY_LAYER}-eval", workspace.data_dir)
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
    assert "em" in metrics["datasets"][0]["overall"]


def test_rerunning_evaluate_does_not_double_count(workspace):
    """同 label 重跑是「重算这一层」，不是「结果翻倍」。"""
    connection = workspace.connection
    assert run(QUERY_LAYER, [DATASET], workspace.db_path, (2,)) == 0
    eval_id = int(repo.eval_layer_by_label(connection, f"{QUERY_LAYER}-eval")["id"])
    first = repo.metric_summaries(connection, eval_id, DATASET)

    assert run(QUERY_LAYER, [DATASET], workspace.db_path, (2,)) == 0
    assert repo.metric_summaries(connection, eval_id, DATASET) == first
    assert len(repo.sample_evals(connection, eval_id)) == 3


def test_only_selected_metrics_reach_the_database(workspace):
    """勾了的才写进 ``sample_metric`` —— 报告页的列以勾选为准。

    过滤在**写库之前**做而不是计算之前：逐样本的检索族指标是一次算出来的一组,
    拆开单算不会更快。而 ``detail`` 保留全部明细 —— 那是归因要读的原始链路，
    与「这一轮报哪些指标」是两件事。
    """
    connection = workspace.connection
    assert (
        run(QUERY_LAYER, [DATASET], workspace.db_path, (2, 5), metrics=["recall", "f1"]) == 0
    )
    eval_id = int(repo.eval_layer_by_label(connection, f"{QUERY_LAYER}-eval")["id"])

    stored = {
        row["metric"]
        for row in connection.execute(
            "SELECT DISTINCT metric FROM sample_metric WHERE eval_layer_id = ?", (eval_id,)
        )
    }
    assert stored == {"recall@2", "recall@5", "f1"}
    # 没勾的确定性指标一条都不该在。
    assert not {"em", "ndcg@2", "mrr", "citation_precision"} & stored

    # 勾选写进层的身份里，前端据此决定显示哪些列。
    layer = repo.get_eval_layer(connection, eval_id)
    assert repo.loads(layer["metrics_json"]) == ["f1", "recall"]

    # detail 仍是完整的：归因判 compiled_away / citation_dropped 要读它。
    detail = repo.loads(repo.sample_evals(connection, eval_id)[0]["detail_json"])
    assert "retrieval" in detail and "attribution" in detail


def test_deselected_metrics_are_reported_separately_from_undefined_ones(workspace):
    """两种「没有值」分开陈述。

    缺依赖是「这个数据集永远算不了」，没勾选是「这一轮没要」。混成一句话的话,
    读者会把后者当成前者，进而以为其他组也缺 gold 标注。
    """
    connection = workspace.connection
    assert run(QUERY_LAYER, [DATASET], workspace.db_path, (2,), metrics=["f1"]) == 0
    eval_id = int(repo.eval_layer_by_label(connection, f"{QUERY_LAYER}-eval")["id"])
    row = repo.dataset_evals(connection, eval_id)[0]

    # hotpotqa 有 gold，所以没有「算不了」的项 —— omitted 只收那一类。
    assert repo.loads(row["omitted_metrics_json"]) == []
    reason = row["omission_reason"]
    assert "not selected for this run" in reason
    assert "recall" in reason
    assert "GOLD_DOCS" not in reason


def test_selecting_nothing_still_means_everything(workspace):
    """不传勾选就是全量 —— 命令行那条路不能因为加了勾选而变。"""
    connection = workspace.connection
    assert run(QUERY_LAYER, [DATASET], workspace.db_path, (2,)) == 0
    eval_id = int(repo.eval_layer_by_label(connection, f"{QUERY_LAYER}-eval")["id"])
    stored = {
        row["metric"]
        for row in connection.execute(
            "SELECT DISTINCT metric FROM sample_metric WHERE eval_layer_id = ?", (eval_id,)
        )
    }
    assert {"em", "f1", "recall@2", "mrr", "citation_precision"} <= stored


def test_report_tables_have_matching_column_counts(workspace):
    """每张表的表头、分隔行与数据行列数必须一致。

    删掉某一列时只改表头不改数据行（或反之）不会报错，只会让 Markdown 表格错行，
    而错行的表格照样是「一份报告」—— 看起来正常，读出来的数却对错了列。
    """
    assert (
        run(QUERY_LAYER, [DATASET], workspace.db_path, (2, 5), export=True,
            data_dir=workspace.data_dir)
        == 0
    )
    report = (
        reports_dir(f"{QUERY_LAYER}-eval", workspace.data_dir) / "report.md"
    ).read_text(encoding="utf-8")

    lines = report.splitlines()
    tables = 0
    for index, line in enumerate(lines):
        # 分隔行（| --- | --- |）定位一张表：它上面是表头，下面是数据行。
        if not line.startswith("|") or set(line.replace("|", "").replace("-", "").strip()):
            continue
        tables += 1
        width = line.count("|")
        assert lines[index - 1].count("|") == width, (
            f"header/separator mismatch:\n{lines[index - 1]}\n{line}"
        )
        for row in lines[index + 1 :]:
            if not row.startswith("|"):
                break
            assert row.count("|") == width, (
                f"row has {row.count('|')} pipes, header has {width}:\n{row}"
            )
    assert tables >= 2, f"expected the retrieval and stratified tables, found {tables}"
