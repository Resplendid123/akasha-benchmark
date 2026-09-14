"""Akasha 客户端契约，以及编译/查询阶段的闸门。

用假客户端跑，不需要真实部署。这些闸门拦的都是不报错的失败：
非 owner 静默丢 chunk、换 embedding 让旧 chunk 召回不到、闸门读到空值假通过。
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from akasha_benchmark import model_configs
from akasha_benchmark.akasha_client import unwrap_envelope
from akasha_benchmark.stages import compile, query
from akasha_benchmark.store import config_store, dumps, run_store
from akasha_benchmark.task import TaskContext


def context(connection, params: dict[str, Any]) -> TaskContext:
    connection.execute(
        "INSERT OR IGNORE INTO task (id, stage, status, params_json, created_at) "
        "VALUES (1, 'test', 'running', '{}', 'now')"
    )
    connection.commit()
    return TaskContext(
        task_id=1,
        stage="test",
        params=params,
        connection=connection,
        pause_event=threading.Event(),
    )


CONFIGS = {
    "configs": [
        {"feature": "compiler", "provider": "p", "model": "c1", "baseUrl": "u", "parameters": {}},
        {"feature": "embedding", "provider": "p", "model": "e1", "baseUrl": "u", "parameters": {}},
        {"feature": "answer", "provider": "p", "model": "a1", "baseUrl": "u", "parameters": {}},
        {"feature": "image", "provider": "p", "model": "i1", "baseUrl": "u", "parameters": {}},
    ]
}


class FakeClient:
    """只实现阶段用到的那几个方法。"""

    def __init__(
        self,
        config,
        *,
        role: str = "owner",
        configs: Any = None,
        retrieved: list[str] | None = None,
        accepted_runs: int = 1,
        status_counts: dict[str, int] | None = None,
        page_log_items: list[dict[str, Any]] | None = None,
        run_items: list[dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.role = role
        # accepted_runs=0 且 status_counts 为空 = Akasha 没接编译请求。
        self.accepted_runs = accepted_runs
        self.status_counts = (
            status_counts if status_counts is not None else {"succeeded": 1}
        )
        # 逐页编译日志。闸门失败时阶段会读它问原因。
        self.page_log_items: list[dict[str, Any]] = page_log_items or []
        # 编译 Run 明细，用于估算每篇耗时。
        self.run_items: list[dict[str, Any]] = (
            run_items
            if run_items is not None
            else [
                {
                    "runDurationMs": 8000,
                    "progress": {"text": {"expected": 4, "succeeded": 4, "failed": 0}},
                }
            ]
        )
        self.configs = configs if configs is not None else CONFIGS
        # query 时回哪些 page_id。空表示回空 retrievedSources（生成端拒答的形状）。
        self.retrieved = retrieved or []
        self.imported: list[str] = []
        self.quality: dict[str, Any] = {
            "summary": {
                "missingChunkPageCount": 0,
                "missingEmbeddingPageCount": 0,
                "missingSourcePageCount": 0,
                "stalePageCount": 0,
            }
        }
        self.queries: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def close(self):
        pass

    def login(self):
        pass

    def current_user(self):
        return {"user": {"id": "u", "role": self.role}, "workspace": {"id": "w", "name": "W"}}

    def get_model_configs(self):
        return self.configs

    def create_space(self, name, slug, description=""):
        return {"id": "space-1", "slug": slug}

    def import_page_text(self, filename, markdown, space_id):
        self.imported.append(filename)
        return {"id": f"page-{len(self.imported)}"}

    def compile_spaces(self, space_ids):
        return {"acceptedRunCount": self.accepted_runs, "coalescedRunCount": 0}

    def run_diagnostics_summary(self, space_ids):
        return {"statusCounts": dict(self.status_counts)}

    def run_diagnostics(self, space_ids, *, limit=50):
        return {"items": list(self.run_items)}

    def quality_diagnostics(self, space_ids):
        return self.quality

    def page_log(self, space_ids, *, limit=100):
        return {"items": list(self.page_log_items)}

    def query(self, question, space_ids, score_threshold=None):
        from akasha_benchmark.akasha_client import Response

        self.queries.append(question)
        return Response(
            status=200,
            body={
                "answerMode": "knowledge",
                "answer": "a",
                "retrievedSources": [{"sourcePageId": page} for page in self.retrieved],
                "citations": [{"sourcePageId": page} for page in self.retrieved],
                "citationEvidence": [{"excerpts": ["x"]} for _ in self.retrieved],
                "snippets": [],
            },
            latency_ms=10,
        )


@pytest.fixture
def ready_connection(normalized):
    config_store.update_connection(
        normalized, base_url="http://x", email="e@x", password="p"
    )
    normalized.commit()
    return normalized


# ------------------------------------------------------------ 客户端契约


def test_envelope_is_unwrapped_only_when_it_is_an_envelope():
    """全局拦截器给每个端点套了 {data, success, status}。不剥会静默读空。"""
    assert unwrap_envelope({"data": {"id": 1}, "success": True, "status": 200}) == {"id": 1}
    # login 的 handler 没有返回值，信封里没有 data 键。
    assert unwrap_envelope({"success": True, "status": 201}) is None
    # 正常载荷里恰好有一个叫 data 的字段时不能动。
    payload = {"data": 1, "success": True, "status": 200, "extra": "x"}
    assert unwrap_envelope(payload) == payload
    assert unwrap_envelope([1, 2]) == [1, 2]


def test_embedding_drift_is_detected():
    changed = {
        "configs": [
            {**entry, "model": "e2"} if entry["feature"] == "embedding" else entry
            for entry in CONFIGS["configs"]
        ]
    }
    assert model_configs.matches(CONFIGS, changed, "compiler") is True
    assert model_configs.matches(CONFIGS, changed, "embedding") is False
    assert model_configs.drift(changed, CONFIGS) == {
        "compiler": False,
        "embedding": True,
        "answer": False,
        "image": False,
    }


def test_normalize_model_configs_is_order_stable():
    reversed_configs = {"configs": list(reversed(CONFIGS["configs"]))}
    assert model_configs.normalize(CONFIGS) == model_configs.normalize(reversed_configs)


# ------------------------------------------------------------ 编译闸门


def test_compile_requires_owner(ready_connection, monkeypatch):
    """非 owner 会在授权闸门静默丢弃 chunk，症状看起来像召回质量差。"""
    monkeypatch.setattr(
        compile, "AkashaClient", lambda config: FakeClient(config, role="member")
    )
    ctx = context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2})
    with pytest.raises(RuntimeError, match="owner"):
        compile.run(ctx)


def test_compile_fails_when_quality_gate_reports_nothing(ready_connection, monkeypatch):
    """四项计数取不到值时不算通过 —— all() 对空集合返回 True，那会让半成品过闸。"""

    def factory(config):
        client = FakeClient(config)
        client.quality = {}
        return client

    monkeypatch.setattr(compile, "AkashaClient", factory)
    ctx = context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2})
    with pytest.raises(RuntimeError, match="质量闸门"):
        compile.run(ctx)

    run = run_store.list_compile_runs(ready_connection)[0]
    assert run["status"] == run_store.STATUS_FAILED


def test_compile_records_pace_estimate(ready_connection, monkeypatch):
    """每篇耗时按 Run 墙钟时长 ÷ 页数估算。"""

    def factory(config):
        return FakeClient(
            config,
            run_items=[
                {"runDurationMs": 8000, "progress": {"text": {"expected": 4}}},
            ],
        )

    monkeypatch.setattr(compile, "AkashaClient", factory)
    ctx = context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2, "run_id": "p1"})
    compile.run(ctx)

    from akasha_benchmark.store import loads

    run = run_store.compile_run_by_run_id(ready_connection, "p1")
    pace = loads(run["pace_json"])
    assert pace["pages"] == 4
    assert pace["total_ms"] == 8000
    assert pace["per_page_ms"] == 2000


def test_compile_pace_survives_missing_diagnostics(ready_connection, monkeypatch):
    """诊断拿不到就不记 pace，编译照样成功：它是展示用的估算，不是闸门。"""

    def factory(config):
        return FakeClient(config, run_items=[])

    monkeypatch.setattr(compile, "AkashaClient", factory)
    ctx = context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2, "run_id": "p2"})
    compile.run(ctx)

    run = run_store.compile_run_by_run_id(ready_connection, "p2")
    assert run["status"] == run_store.STATUS_SUCCEEDED
    assert run["pace_json"] is None


def test_compile_reports_page_failure_reason(ready_connection, monkeypatch):
    """闸门只报后果，报错里要带上逐页日志给出的 errorCode。"""
    pages = [
        {
            "status": "failed",
            "errorCode": "provider_error",
            "errorSummary": "Knowledge compiler provider request failed.",
            "title": f"Doc {index}",
        }
        for index in range(8)
    ]

    def factory(config):
        client = FakeClient(config, page_log_items=pages)
        client.quality = {"summary": {"missingChunkPageCount": 8, "missingSourcePageCount": 8}}
        return client

    monkeypatch.setattr(compile, "AkashaClient", factory)
    ctx = context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2})
    with pytest.raises(RuntimeError, match="provider_error") as caught:
        compile.run(ctx)

    assert "8 篇" in str(caught.value)


def test_compile_fails_when_no_run_was_accepted(ready_connection, monkeypatch):
    """一个编译 Run 都没有算没编译，不算编译好了（两者的 active 都是 0）。"""

    def factory(config):
        return FakeClient(config, accepted_runs=0, status_counts={})

    monkeypatch.setattr(compile, "AkashaClient", factory)
    ctx = context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2})
    with pytest.raises(RuntimeError, match="编译没有启动"):
        compile.run(ctx)

    run = run_store.list_compile_runs(ready_connection)[0]
    assert run["status"] == run_store.STATUS_FAILED


def test_compile_waits_when_runs_are_still_active(ready_connection, monkeypatch):
    """有 Run 在跑就得等 —— 别把「进行中」当成「没有 Run」提前放行。"""
    seen: list[dict[str, int]] = []

    class Slow(FakeClient):
        def run_diagnostics_summary(self, space_ids):
            seen.append({})
            # 第一次回「编译中」，第二次回终态。
            counts = {"compiling": 1} if len(seen) == 1 else {"succeeded": 1}
            return {"statusCounts": counts}

    monkeypatch.setattr(compile, "AkashaClient", lambda config: Slow(config))
    monkeypatch.setattr(compile.time, "sleep", lambda seconds: None)
    ctx = context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2, "run_id": "r0"})
    compile.run(ctx)

    assert len(seen) == 2
    run = run_store.compile_run_by_run_id(ready_connection, "r0")
    assert run["status"] == run_store.STATUS_SUCCEEDED


def test_compile_succeeds_and_records_pages(ready_connection, monkeypatch):
    clients: list[FakeClient] = []

    def factory(config):
        client = FakeClient(config)
        clients.append(client)
        return client

    monkeypatch.setattr(compile, "AkashaClient", factory)
    ctx = context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2, "run_id": "r1"})
    compile.run(ctx)

    run = run_store.compile_run_by_run_id(ready_connection, "r1")
    assert run["status"] == run_store.STATUS_SUCCEEDED
    assert run["space_id"] == "space-1"
    docs = run_store.compile_docs(ready_connection, int(run["id"]))
    assert docs and all(doc["page_id"] for doc in docs)
    assert len(clients[0].imported) == len(docs)
    assert run_store.compile_ready(ready_connection, int(run["id"]))["ready"] is True


def test_compile_resume_skips_imported_docs(ready_connection, monkeypatch):
    clients: list[FakeClient] = []

    def factory(config):
        client = FakeClient(config)
        clients.append(client)
        return client

    monkeypatch.setattr(compile, "AkashaClient", factory)
    params = {"datasets": ["hotpotqa"], "qa_limit": 2, "run_id": "r1"}
    compile.run(context(ready_connection, params))
    first = len(clients[0].imported)

    # 同 run_id 再跑一次：子集与已导入的文档都跳过。
    compile.run(context(ready_connection, params))
    assert clients[1].imported == []
    assert first > 0


# ------------------------------------------------------------ 查询闸门


def _compiled(connection, monkeypatch) -> int:
    monkeypatch.setattr(compile, "AkashaClient", lambda config: FakeClient(config))
    compile.run(
        context(connection, {"datasets": ["hotpotqa"], "qa_limit": 2, "run_id": "r1"})
    )
    return int(run_store.compile_run_by_run_id(connection, "r1")["id"])


def test_query_refuses_when_embedding_changed(ready_connection, monkeypatch):
    """这一项不可绕过：旧 chunk 永远召回不到，指标会全错但不报错。"""
    compile_id = _compiled(ready_connection, monkeypatch)
    changed = {
        "configs": [
            {**entry, "model": "e2"} if entry["feature"] == "embedding" else entry
            for entry in CONFIGS["configs"]
        ]
    }
    monkeypatch.setattr(
        query, "AkashaClient", lambda config: FakeClient(config, configs=changed)
    )
    with pytest.raises(RuntimeError, match="embedding"):
        query.run(context(ready_connection, {"compile_id": compile_id}))


def test_query_refuses_on_workspace_mismatch(ready_connection, monkeypatch):
    """换了账号/部署之后，这次编译的 page_id 在这里解析不到。

    查询打上去不报错，只会每条都召回不到 —— 一份 recall 全 0 的报告，
    看起来像检索烂到极点而不像配置指向了别处。
    """
    compile_id = _compiled(ready_connection, monkeypatch)

    class OtherWorkspace(FakeClient):
        def current_user(self):
            return {
                "user": {"id": "u", "role": "owner"},
                "workspace": {"id": "w-other", "name": "Other"},
            }

    monkeypatch.setattr(query, "AkashaClient", lambda config: OtherWorkspace(config))
    with pytest.raises(RuntimeError, match="workspace"):
        query.run(context(ready_connection, {"compile_id": compile_id}))
    # 拒绝执行时不该留下一条查询记录。
    assert run_store.list_query_runs(ready_connection, compile_id) == []


def test_compile_resume_refuses_on_workspace_mismatch(ready_connection, monkeypatch):
    """续跑往一个解析不到的空间导入，会让已记下的 page_id 全部失效。"""
    compile_id = _compiled(ready_connection, monkeypatch)
    # 造一篇没导入成功的文档，让续跑确实有活要干。
    ready_connection.execute(
        "UPDATE compile_doc SET page_id = NULL WHERE compile_id = ?", (compile_id,)
    )
    ready_connection.commit()

    class OtherWorkspace(FakeClient):
        def current_user(self):
            return {
                "user": {"id": "u", "role": "owner"},
                "workspace": {"id": "w-other", "name": "Other"},
            }

    clients: list[FakeClient] = []

    def factory(config):
        client = OtherWorkspace(config)
        clients.append(client)
        return client

    monkeypatch.setattr(compile, "AkashaClient", factory)
    with pytest.raises(RuntimeError, match="workspace"):
        compile.run(
            context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2, "run_id": "r1"})
        )
    # 必须在发出任何写入之前拦住。
    assert clients[-1].imported == []


def test_workspace_mismatch_tolerates_missing_record(ready_connection, monkeypatch):
    """编译时没记下 workspace（历史数据）时不拦：没有可比的东西。"""
    compile_id = _compiled(ready_connection, monkeypatch)
    ready_connection.execute(
        "UPDATE compile_run SET workspace_id = NULL WHERE id = ?", (compile_id,)
    )
    ready_connection.commit()
    assert run_store.workspace_mismatch(ready_connection, compile_id, "w-other") is None


def test_query_refuses_unready_compile(ready_connection, monkeypatch):
    compile_id = run_store.create_compile_run(
        ready_connection, run_id="half", datasets=["hotpotqa"], seed=1, qa_limit=1, negatives_ratio=1.0
    )
    ready_connection.commit()
    monkeypatch.setattr(query, "AkashaClient", lambda config: FakeClient(config))
    with pytest.raises(ValueError, match="不能用于查询"):
        query.run(context(ready_connection, {"compile_id": compile_id}))


def test_query_records_responses_and_resumes(ready_connection, monkeypatch):
    compile_id = _compiled(ready_connection, monkeypatch)
    clients: list[FakeClient] = []

    def factory(config):
        client = FakeClient(config)
        clients.append(client)
        return client

    monkeypatch.setattr(query, "AkashaClient", factory)
    params = {"compile_id": compile_id, "name": "q1"}
    query.run(context(ready_connection, params))

    run = run_store.query_run_by_name(ready_connection, "q1")
    assert run["status"] == run_store.STATUS_SUCCEEDED
    responses = run_store.responses_of(ready_connection, int(run["id"]))
    assert len(responses) == 2
    assert all(r["answer_mode"] == "knowledge" for r in responses)

    # 续跑：已有响应的样本不再发请求。
    query.run(context(ready_connection, params))
    assert clients[-1].queries == []


def test_query_uses_frozen_selection(ready_connection, monkeypatch):
    """固化选择让续跑不受后续抽样改动影响。"""
    compile_id = _compiled(ready_connection, monkeypatch)
    monkeypatch.setattr(query, "AkashaClient", lambda config: FakeClient(config))
    query.run(
        context(ready_connection, {"compile_id": compile_id, "name": "q1", "sample_limit": 1})
    )
    run = run_store.query_run_by_name(ready_connection, "q1")
    assert len(run_store.query_samples(ready_connection, int(run["id"]))) == 1


def test_query_rejects_name_from_another_compile(ready_connection, monkeypatch):
    compile_id = _compiled(ready_connection, monkeypatch)
    other = run_store.create_compile_run(
        ready_connection, run_id="other", datasets=[], seed=1, qa_limit=1, negatives_ratio=1.0
    )
    run_store.create_query_run(
        ready_connection,
        name="taken",
        compile_id=other,
        score_threshold=None,
        concurrency=1,
        model_configs=CONFIGS,
    )
    ready_connection.commit()
    monkeypatch.setattr(query, "AkashaClient", lambda config: FakeClient(config))
    with pytest.raises(ValueError, match="另一次编译"):
        query.run(context(ready_connection, {"compile_id": compile_id, "name": "taken"}))


def test_compile_snapshot_is_frozen_on_the_run(ready_connection, monkeypatch):
    """run_id 上固化模型快照 —— 查询前拿现在的配置与它比对靠这一份。"""
    compile_id = _compiled(ready_connection, monkeypatch)
    run = run_store.get_compile_run(ready_connection, compile_id)
    assert run["model_configs_json"] == dumps(CONFIGS)
