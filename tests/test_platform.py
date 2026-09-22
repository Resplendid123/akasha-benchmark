"""平台层：任务生命周期（暂停/继续/清理）、审计日志留存、路由与鉴权。"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from akasha_benchmark.datasets import DATASET_NAMES
from akasha_benchmark.stages import STAGES, StageSpec
from akasha_benchmark.store import (
    attribution_store,
    compile_store,
    config_store,
    connect,
    eval_store,
    query_store,
    run_store,
    task_store,
)
from akasha_platform.main import create_app
from akasha_benchmark.metrics.interpretation import build_metric_evidence
from akasha_platform.settings import Settings
from akasha_platform.tasks import TaskRejected, TaskRunner


def test_citation_precision_evidence_matches_scoring_scope():
    evidence = build_metric_evidence(
        ["citation_precision"],
        [5],
        {"citation_precision": 0.5},
        {"gold_doc_ids": ["g1", "g2"], "reference_answers": []},
        {
            "citations": [
                {"sourcePageId": "p-gold", "title": "Gold"},
                {"sourcePageId": "p-other", "title": "Other"},
                {"sourcePageId": "p-gold", "title": "duplicate"},
                {"sourcePageId": "p-missing", "title": "Missing"},
            ],
            "snippets": [
                {
                    "id": "s1",
                    "title": "Gold chunk",
                    "text": "gold evidence text",
                    "retrievalReasons": ["semantic"],
                    "sourceWindows": [{"sourcePageId": "p-gold"}],
                }
            ],
            "citationEvidence": [
                {
                    "sourcePageId": "p-gold",
                    "excerpts": [{"text": "exact cited sentence"}],
                }
            ],
        },
        {"p-gold": "g1", "p-other": "d2"},
        {
            "g1": {"doc_id": "g1", "title": "Gold document"},
            "g2": {"doc_id": "g2", "title": "Uncited gold"},
            "d2": {"doc_id": "d2", "title": "Other document"},
        },
        [],
    )["citation_precision"]

    assert evidence["formula"] == "实际引用中命中 Gold（1）/ 实际引用（2）"
    assert [(doc["doc_id"], doc["is_gold"]) for doc in evidence["documents"]] == [
        ("g1", True),
        ("d2", False),
        (None, False),
    ]
    assert evidence["snippets"][0]["text"] == "gold evidence text"
    assert evidence["citation_excerpts"][0]["excerpts"] == ["exact cited sentence"]
    assert [(doc["doc_id"], doc["cited"]) for doc in evidence["gold_documents"]] == [
        ("g1", True),
        ("g2", False),
    ]


@pytest.fixture
def settings(db_path) -> Settings:
    return Settings(db_path=db_path)


@pytest.fixture
def client(settings) -> TestClient:
    return TestClient(create_app(settings))


class _AkashaStub:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def login(self):
        pass


def _compile_run(connection, **overrides) -> int:
    params = {
        "run_id": "r",
        "datasets": [],
        "seed": 1,
        "qa_limit": 1,
        "negatives_ratio": 1.0,
    }
    return compile_store.create_compile_run(connection, **(params | overrides))


# ------------------------------------------------------------ 任务生命周期


def _register(monkeypatch, name: str, run) -> None:
    monkeypatch.setitem(
        STAGES, name, StageSpec(label=name, run=run, params={"marker": str})
    )


def _wait(settings, task_id: int, statuses: set[str], timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        connection = connect(settings.db_path, read_only=True)
        try:
            task = task_store.get_task(connection, task_id)
        finally:
            connection.close()
        if task and task["status"] in statuses:
            return task
        time.sleep(0.02)
    raise AssertionError(f"任务 #{task_id} 未在 {timeout}s 内进入 {statuses}，当前 {task}")


def test_task_succeeds_and_logs_are_kept_after_cleanup(settings, monkeypatch):
    """审计日志只追加：清理任务记录之后它仍然查得到。"""

    def stage(ctx):
        ctx.log("干了点事")
        ctx.progress(1, 1, "完成")

    _register(monkeypatch, "unit", stage)
    runner = TaskRunner(settings)
    task = runner.start("unit", {"marker": "x"})
    finished = _wait(settings, int(task["id"]), {task_store.SUCCEEDED})
    assert finished["status"] == task_store.SUCCEEDED

    runner.cleanup(int(task["id"]))
    connection = connect(settings.db_path, read_only=True)
    try:
        assert task_store.get_task(connection, int(task["id"])) is None
        messages = [entry["message"] for entry in task_store.audit_logs(connection)]
    finally:
        connection.close()
    assert any("干了点事" in message for message in messages)


def test_failure_is_recorded_on_the_task(settings, monkeypatch):
    def stage(ctx):
        raise RuntimeError("炸了")

    _register(monkeypatch, "unit", stage)
    runner = TaskRunner(settings)
    task = runner.start("unit", {})
    failed = _wait(settings, int(task["id"]), {task_store.FAILED})
    assert "炸了" in (failed["error"] or "")


def test_failed_query_can_create_retry_task_without_original_task(settings, db, monkeypatch):
    compile_id = _compile_run(db, run_id="retry-query-compile")
    query_id = query_store.create_query_run(
        db, name="retry-query", compile_id=compile_id,
        concurrency=3, model_configs={},
    )
    query_store.record_response(
        db, query_id, sample_id="failed", dataset="hotpotqa", question="q",
        http_status=500, latency_ms=1, error="failed", response=None,
    )
    db.commit()

    runner = TaskRunner(settings)
    monkeypatch.setattr(runner, "_spawn", lambda *args: None)
    task = runner.retry_failed_query(query_id)

    assert task["target_kind"] == "query"
    assert task["target_id"] == query_id
    assert task["params"]["retry_failed"] is True
    assert task["params"]["concurrency"] == 3
    assert task["progress_total"] == 1
    assert query_store.get_query_run(db, query_id)["status"] == "paused"


def test_pause_stops_at_checkpoint_and_resume_continues(settings, monkeypatch):
    """暂停是协作式的：停在已落库的位置上，继续时接着跑而不是重来。"""
    entered = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def stage(ctx):
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            release.wait(5)
            # 第一次进来时会被请求暂停，checkpoint 在这里抛出。
            ctx.checkpoint()
            raise AssertionError("checkpoint 没有拦住暂停")
        ctx.log("续跑完成")

    _register(monkeypatch, "unit", stage)
    runner = TaskRunner(settings)
    task = runner.start("unit", {})
    task_id = int(task["id"])
    assert entered.wait(5)

    runner.pause(task_id)
    release.set()
    paused = _wait(settings, task_id, {task_store.PAUSED})
    assert paused["status"] == task_store.PAUSED

    runner.resume(task_id)
    done = _wait(settings, task_id, {task_store.SUCCEEDED})
    assert done["status"] == task_store.SUCCEEDED
    assert len(calls) == 2


@pytest.mark.parametrize("action", ["pause", "cleanup"])
def test_compile_task_action_cancels_remote_bullmq_run(settings, db, monkeypatch, action):
    """暂停和清理通过 Akasha 控制面取消精确 Run，不暂停共享 BullMQ 队列。"""
    calls: list[tuple[str, str]] = []

    class FakeAkasha(_AkashaStub):
        def cancel_compile_run(self, run_id, reason):
            calls.append((run_id, reason))
            return {"disposition": "cancelled", "runId": run_id, "status": "cancelled", "removedJobCount": 2}

    monkeypatch.setattr("akasha_platform.tasks.AkashaClient", FakeAkasha)
    task_id = task_store.create_task(
        db,
        stage="compile",
        params={"remote_compile_run_ids": ["run-1"]},
    )
    task_store.transition(
        db, task_id, task_store.RUNNING if action == "pause" else task_store.PAUSED
    )
    db.commit()

    runner = TaskRunner(settings)
    getattr(runner, action)(task_id)

    assert [run_id for run_id, _ in calls] == ["run-1"]
    assert calls[0][1].startswith("Akasha-Benchmark task")


def test_pause_rejects_invalid_remote_cancel_response(settings, db, monkeypatch):
    class FakeAkasha(_AkashaStub):
        def cancel_compile_run(self, run_id, reason):
            return {"disposition": "not_found", "runId": run_id}

    monkeypatch.setattr("akasha_platform.tasks.AkashaClient", FakeAkasha)
    task_id = task_store.create_task(
        db, stage="compile", params={"remote_compile_run_ids": ["run-1"]}
    )
    task_store.transition(db, task_id, task_store.RUNNING)
    db.commit()

    with pytest.raises(TaskRejected, match="取消.*无效结果"):
        TaskRunner(settings).pause(task_id)
    assert task_store.get_task(db, task_id)["status"] == task_store.RUNNING


def test_pause_discovers_and_cancels_active_run_before_id_is_saved(
    settings, db, compile_id, monkeypatch
):
    cancelled: list[str] = []

    class FakeAkasha(_AkashaStub):
        def run_diagnostics(self, space_ids, *, limit=50):
            return {"items": [{"runId": "active-run", "status": "compiling"}]}

        def cancel_compile_run(self, run_id, reason):
            cancelled.append(run_id)
            return {"disposition": "cancelled", "runId": run_id, "status": "cancelled"}

    monkeypatch.setattr("akasha_platform.tasks.AkashaClient", FakeAkasha)
    compile_store.update_compile_run(db, compile_id, space_id="benchmark-space")
    task_id = task_store.create_task(db, stage="compile", params={})
    task_store.set_task_target(db, task_id, "compile", compile_id)
    task_store.transition(db, task_id, task_store.RUNNING)
    db.commit()

    TaskRunner(settings).pause(task_id)
    assert cancelled == ["active-run"]
    assert task_store.get_task(db, task_id)["status"] == task_store.PAUSED


@pytest.mark.parametrize("status", ["cancelled", "succeeded"])
def test_pause_distinguishes_cancelled_from_naturally_terminal_remote_run(
    settings, db, monkeypatch, status
):
    class FakeAkasha(_AkashaStub):
        def cancel_compile_run(self, run_id, reason):
            return {"disposition": "already_terminal", "runId": run_id, "status": status}

    monkeypatch.setattr("akasha_platform.tasks.AkashaClient", FakeAkasha)
    task_id = task_store.create_task(
        db, stage="compile", params={"remote_compile_run_ids": ["run-1"]}
    )
    task_store.transition(db, task_id, task_store.RUNNING)
    db.commit()

    result = TaskRunner(settings).pause(task_id)
    assert result["status"] == (
        task_store.PAUSED if status == "cancelled" else task_store.RUNNING
    )


def test_running_task_cannot_be_cleaned_up(settings, monkeypatch):
    """在跑的任务不许删 —— 那会留下一个没人认领的线程还在写库。"""
    release = threading.Event()
    entered = threading.Event()

    def stage(ctx):
        entered.set()
        release.wait(5)

    _register(monkeypatch, "unit", stage)
    runner = TaskRunner(settings)
    task = runner.start("unit", {})
    assert entered.wait(5)
    with pytest.raises(TaskRejected):
        runner.cleanup(int(task["id"]))
    release.set()
    _wait(settings, int(task["id"]), {task_store.SUCCEEDED})


def test_same_stage_does_not_run_twice(settings, monkeypatch):
    release = threading.Event()
    entered = threading.Event()

    def stage(ctx):
        entered.set()
        release.wait(5)

    _register(monkeypatch, "unit", stage)
    runner = TaskRunner(settings)
    runner.start("unit", {})
    assert entered.wait(5)
    with pytest.raises(TaskRejected, match="正在运行"):
        runner.start("unit", {})
    release.set()


def test_compile_resume_obeys_same_stage_serialization(settings, db, monkeypatch):
    paused_id = task_store.create_task(db, stage="compile", params={})
    task_store.transition(db, paused_id, task_store.PAUSED)
    task_id = task_store.create_task(db, stage="compile", params={})
    task_store.transition(db, task_id, task_store.RUNNING)
    db.commit()

    runner = TaskRunner(settings)
    monkeypatch.setattr(runner, "_spawn", lambda *args: None)
    with pytest.raises(TaskRejected, match="正在运行"):
        runner.resume(paused_id)
    assert task_store.get_task(db, paused_id)["status"] == task_store.PAUSED


def test_unknown_stage_and_bad_params_are_rejected(settings):
    runner = TaskRunner(settings)
    with pytest.raises(TaskRejected, match="未知阶段"):
        runner.start("nope", {})
    with pytest.raises(TaskRejected):
        runner.start("query", {"compile_id": "abc"})


def test_recover_marks_orphaned_tasks_paused(settings):
    """后端重启后线程没了，留着 running 会让界面显示一个不存在的任务。"""
    connection = connect(settings.db_path)
    try:
        task_id = task_store.create_task(connection, stage="compile", params={})
        task_store.transition(connection, task_id, task_store.RUNNING)
        connection.commit()
    finally:
        connection.close()

    assert TaskRunner(settings).recover() == 1
    connection = connect(settings.db_path, read_only=True)
    try:
        assert task_store.get_task(connection, task_id)["status"] == task_store.PAUSED
    finally:
        connection.close()


# ------------------------------------------------------------ 路由


def test_health(client):
    assert client.get("/api/health").json()["ok"] is True


def test_datasets_route_reports_missing_files(client):
    body = client.get("/api/datasets").json()
    # 跟注册表比，而不是写死条数 —— 加一组适配器不该让这条测试失败。
    assert {d["name"] for d in body["datasets"]} == set(DATASET_NAMES)
    entry = next(d for d in body["datasets"] if d["name"] == "hotpotqa")
    assert entry["normalized"] is False
    assert "sample_id" in entry["identity_rules"]


def test_metrics_route_narrows_by_dataset(client):
    body = client.get("/api/metrics?datasets=narrativeqa").json()
    assert "recall" not in body["computable_for_all"]
    # faithfulness 不需要任何标注，所以它对 narrativeqa 也成立。
    assert "faithfulness" in body["computable_for_all"]

    both = client.get("/api/metrics?datasets=hotpotqa,narrativeqa").json()
    assert "recall" in both["computable_for_some"]
    assert "recall" not in both["computable_for_all"]


def test_connection_put_only_updates_given_fields(client):
    client.put("/api/connection", json={"base_url": "http://x", "email": "a@b", "password": ""})
    body = client.get("/api/connection").json()
    assert body["base_url"] == "http://x"
    assert body["password"] == ""

    # 只提交 password，base_url 不该被动。
    client.put("/api/connection", json={"password": "p"})
    body = client.get("/api/connection").json()
    assert body["base_url"] == "http://x"
    assert body["password"] == "p"

    assert client.put("/api/connection", json={"timeout_seconds": "many"}).status_code == 422


def test_provider_api_key_never_leaves_the_backend(client):
    client.put(
        "/api/providers/judge",
        json={"label": "d", "base_url": "https://x/v1", "model": "m", "api_key": "secret"},
    )
    providers = client.get("/api/providers?role=judge").json()
    assert providers[0]["api_key_set"] is True
    assert "api_key" not in providers[0]

    # 密钥留空表示保留原值。
    client.put(
        "/api/providers/judge",
        json={"label": "d", "base_url": "https://x/v1", "model": "m2", "api_key": ""},
    )
    connection = connect(client.app.state.settings.db_path, read_only=True)
    try:
        from akasha_benchmark.store import config_store

        stored = config_store.list_model_providers(connection, "judge")[0]
    finally:
        connection.close()
    assert stored["api_key"] == "secret"
    assert stored["model"] == "m2"


def test_provider_probe_reports_failure_as_data(client, monkeypatch):
    """探测调不通时回 200 + ok=false，不是 500。

    调不通是这个接口要报告的结果，不是它自己的故障 —— 报 500 会让前端把
    「模型端点有问题」显示成「平台出错了」。
    """
    from akasha_benchmark.judge.client import JudgeReply

    client.put(
        "/api/providers/judge",
        json={"label": "d", "base_url": "https://x/v1", "model": "m", "api_key": "k"},
    )
    provider_id = client.get("/api/providers?role=judge").json()[0]["id"]

    monkeypatch.setattr(
        "akasha_benchmark.judge.client.JudgeClient.complete",
        lambda self, system, user: JudgeReply(
            content=None, failure_kind="http:401", raw='{"error":"bad key"}', status=401
        ),
    )
    body = client.post(f"/api/providers/{provider_id}/probe")
    assert body.status_code == 200
    payload = body.json()
    assert payload["ok"] is False
    assert payload["failure"] == "http:401"
    assert payload["status"] == 401
    assert "bad key" in payload["detail"]
    # 探测的响应也不能带出密钥。
    assert payload["provider"]["api_key_set"] is True
    assert "api_key" not in payload["provider"]


def test_provider_probe_returns_the_reply(client, monkeypatch):
    from akasha_benchmark.judge.client import JudgeReply

    client.put(
        "/api/providers/attribution",
        json={"label": "d", "base_url": "https://x/v1", "model": "m", "api_key": "k"},
    )
    provider_id = client.get("/api/providers?role=attribution").json()[0]["id"]

    sent: list[tuple[str, str]] = []

    def fake(self, system, user):
        sent.append((system, user))
        return JudgeReply(content="Hi there", failure_kind=None, raw=None, status=200)

    monkeypatch.setattr("akasha_benchmark.judge.client.JudgeClient.complete", fake)
    payload = client.post(f"/api/providers/{provider_id}/probe").json()
    assert payload["ok"] is True
    assert payload["reply"] == "Hi there"
    # 发的是一句 hi。user 消息里必须带 json —— JudgeClient 固定要求 json_object
    # 输出，而有些 provider 规定消息里出现 "json" 才允许这个格式。
    assert len(sent) == 1
    _, user = sent[0]
    assert user.startswith("hi")
    assert "json" in user.lower()


def test_provider_probe_reports_missing_key(client):
    """没配密钥时也回 200 —— 那是配置问题，同样是要报告的结果。"""
    from akasha_benchmark.store import config_store

    connection = connect(client.app.state.settings.db_path)
    try:
        provider_id = config_store.upsert_model_provider(
            connection,
            purpose="judge",
            label="nokey",
            base_url="https://x/v1",
            model="m",
            api_key="",
        )
        connection.commit()
    finally:
        connection.close()

    payload = client.post(f"/api/providers/{provider_id}/probe").json()
    assert payload["ok"] is False
    assert payload["failure"] == "config"
    assert "api key" in payload["detail"]


def test_provider_probe_404s_on_unknown_endpoint(client):
    assert client.post("/api/providers/9999/probe").status_code == 404


def test_provider_does_not_expose_or_use_concurrency(client):
    client.put(
        "/api/providers/judge",
        json={"label": "d", "base_url": "https://x/v1", "model": "m", "concurrency": 4},
    )
    provider = client.get("/api/providers?role=judge").json()[0]
    assert "concurrency" not in provider

    from akasha_benchmark.judge.providers import resolve_provider

    connection = connect(client.app.state.settings.db_path)
    try:
        provider_id = config_store.upsert_model_provider(
            connection,
            purpose="judge",
            label="k",
            base_url="https://x/v1",
            model="m",
            api_key="key",
        )
        connection.commit()
        resolved = resolve_provider(connection, provider_id, "judge")
    finally:
        connection.close()
    assert not hasattr(resolved, "concurrency")


def test_akasha_models_are_independent_and_hide_keys(client):
    created = client.put(
        "/api/akasha-models",
        json={
            "feature": "answer",
            "label": "answer-a",
            "base_url": "https://answer.example/v1",
            "model": "answer-model",
            "api_key": "secret",
        },
    ).json()
    body = client.get("/api/akasha-models?feature=answer").json()
    row = next(item for item in body["models"] if item["id"] == created["id"])
    assert row["feature"] == "answer"
    assert row["api_key_set"] is True
    assert "api_key" not in row

    client.put(
        "/api/akasha-models",
        json={
            "id": created["id"],
            "feature": "answer",
            "label": "answer-a",
            "base_url": "https://answer.example/v1",
            "model": "answer-model-2",
            "api_key": "",
        },
    )
    connection = connect(client.app.state.settings.db_path, read_only=True)
    try:
        stored = config_store.get_model_provider(connection, created["id"])
    finally:
        connection.close()
    assert stored["model"] == "answer-model-2"
    assert stored["api_key"] == "secret"


def test_embedding_model_dimension_roundtrip_and_validation(client):
    created = client.put(
        "/api/akasha-models",
        json={
            "feature": "embedding",
            "label": "embedding-a",
            "base_url": "https://embedding.example/v1",
            "model": "embedding-model",
            "parameters": {"dimension": 1024},
        },
    )
    assert created.status_code == 200
    row = next(
        item
        for item in client.get("/api/akasha-models?feature=embedding").json()["models"]
        if item["id"] == created.json()["id"]
    )
    assert row["parameters"]["dimension"] == 1024

    invalid = client.put(
        "/api/akasha-models",
        json={
            "feature": "embedding",
            "label": "bad-dimension",
            "base_url": "https://embedding.example/v1",
            "model": "embedding-model",
            "parameters": {"dimension": 0},
        },
    )
    assert invalid.status_code == 422
    assert "dimension" in invalid.json()["detail"]


def test_embedding_model_apply_sends_dimension(client, monkeypatch):
    client.put(
        "/api/connection",
        json={"base_url": "http://x", "email": "e@x", "password": "p"},
    )
    created = client.put(
        "/api/akasha-models",
        json={
            "feature": "embedding",
            "label": "embedding-a",
            "base_url": "https://embedding.example/v1",
            "model": "embedding-model",
            "parameters": {"dimension": 1536},
        },
    ).json()
    pushed = []

    from akasha_platform.api import config as config_api

    class Fake(_AkashaStub):
        def put_model_config(self, feature, payload):
            pushed.append((feature, payload))
            return {}

    monkeypatch.setattr(config_api, "AkashaClient", Fake)
    response = client.post(f"/api/akasha-models/{created['id']}/apply")

    assert response.status_code == 200
    assert pushed == [
        (
            "embedding",
            {
                "provider": "openai-compatible",
                "model": "embedding-model",
                "baseUrl": "https://embedding.example/v1",
                "apiKey": "",
                "parameters": {"dimension": 1536},
            },
        )
    ]


def test_config_import_round_trips_export(client):
    client.put("/api/connection", json={"password": "pw", "email": "e@x", "base_url": "http://y"})
    judge = client.put(
        "/api/providers/judge",
        json={"label": "d", "base_url": "https://x/v1", "model": "m", "api_key": "sk"},
    ).json()
    model = client.put(
        "/api/akasha-models",
        json={
            "feature": "answer",
            "label": "answer-a",
            "base_url": "https://answer.example/v1",
            "model": "a",
            "api_key": "gk",
            "parameters": {"temperature": 0},
        },
    ).json()
    exported = client.get("/api/config/export").json()
    assert exported["connection"]["password"] == "pw"
    assert set(exported) == {"connection", "models"}
    assert exported["models"] == [
        {
            "purpose": "answer",
            "label": "answer-a",
            "base_url": "https://answer.example/v1",
            "model": "a",
            "api_key": "gk",
            "parameters": {"temperature": 0},
        },
        {
            "purpose": "judge",
            "label": "d",
            "base_url": "https://x/v1",
            "model": "m",
            "api_key": "sk",
            "parameters": {},
        },
    ]

    # 清一遍再导回：导入后应与导出前一致。
    client.put("/api/connection", json={"password": "", "email": "", "base_url": ""})
    client.delete(f"/api/akasha-models/{model['id']}")
    client.delete(f"/api/providers/{judge['id']}")
    imported = client.post("/api/config/import", json=exported)
    assert imported.status_code == 200
    assert imported.json()["models"] == 2
    assert set(imported.json()) == {"connection", "models"}

    back = client.get("/api/config/export").json()
    assert back["connection"]["password"] == "pw"
    assert back["connection"]["base_url"] == "http://y"
    assert back["models"] == exported["models"]


def test_config_import_rejects_bad_role(client):
    payload = {"models": [{"purpose": "nonsense", "label": "d", "base_url": "u", "model": "m"}]}
    assert client.post("/api/config/import", json=payload).status_code == 422


def test_connection_test_flags_compiles_in_another_workspace(client, db_path, monkeypatch):
    """预检：不必等起了任务才发现这些编译在当前连接下用不了。"""
    connection = connect(db_path)
    try:
        compile_id = _compile_run(connection, run_id="r1")
        compile_store.update_compile_run(
            connection, compile_id, space_id="s1", workspace_id="w-original"
        )
        connection.commit()
    finally:
        connection.close()
    client.put("/api/connection", json={"base_url": "http://x", "email": "e@x", "password": "p"})

    from akasha_platform.api import config as config_api

    class Fake(_AkashaStub):
        def current_user(self):
            return {
                "user": {"id": "u", "email": "e@x", "role": "owner"},
                "workspace": {"id": "w-other", "name": "Other"},
            }

        def get_model_configs(self):
            return {"configs": []}

    monkeypatch.setattr(config_api, "AkashaClient", Fake)
    body = client.post("/api/connection/test").json()
    assert body["ok"] is True
    assert [entry["run_id"] for entry in body["blocked_compiles"]] == ["r1"]
    assert "w-original" in body["blocked_compiles"][0]["reason"]


def test_raw_samples_serves_qa_and_corpus_separately(client, dataset_dir, monkeypatch):
    """样本与语料是两个入口，读的是两个文件。"""
    from akasha_platform.api import datasets as datasets_api

    monkeypatch.setattr(datasets_api, "DEFAULT_DATASET_DIR", dataset_dir)

    qa = client.get("/api/datasets/hotpotqa/raw?kind=qa&limit=1").json()
    assert qa["source_file"] == "hotpotqa.json"
    assert qa["kind"] == "qa"
    assert len(qa["rows"]) == 1

    corpus = client.get("/api/datasets/hotpotqa/raw?kind=corpus&limit=1").json()
    assert corpus["source_file"] == "hotpotqa_corpus.json"
    assert corpus["kind"] == "corpus"
    assert len(corpus["rows"]) == 1
    assert corpus["rows"] != qa["rows"]

    # 默认是 qa，与加了 kind 之前的行为一致。
    assert client.get("/api/datasets/hotpotqa/raw").json()["kind"] == "qa"
    assert client.get("/api/datasets/hotpotqa/raw?kind=bogus").status_code == 422


def test_raw_samples_searches_qa_and_corpus(client, dataset_dir, monkeypatch):
    """原始结构因数据集而异，搜索应覆盖嵌套字段而不依赖适配器。"""
    from akasha_platform.api import datasets as datasets_api

    monkeypatch.setattr(datasets_api, "DEFAULT_DATASET_DIR", dataset_dir)

    qa = client.get(
        "/api/datasets/hotpotqa/raw",
        params={"kind": "qa", "q": "venice", "limit": 1},
    ).json()
    assert qa["total"] == 1
    assert qa["rows"][0]["_id"] == "q2"

    corpus = client.get(
        "/api/datasets/hotpotqa/raw",
        params={"kind": "corpus", "q": "film festival", "limit": 1},
    ).json()
    assert corpus["total"] == 1
    assert corpus["rows"][0]["title"] == "Venice"

    empty = client.get(
        "/api/datasets/hotpotqa/raw", params={"kind": "corpus", "q": "not-found"}
    ).json()
    assert empty["total"] == 0
    assert empty["rows"] == []


def test_normalized_sample_search_matches_gold_title_and_content(client, normalized):
    by_title = client.get("/api/datasets/hotpotqa/samples", params={"q": "Venice"}).json()
    by_content = client.get(
        "/api/datasets/hotpotqa/samples", params={"q": "film festival"}
    ).json()

    assert [row["sample_id"] for row in by_title["samples"]] == ["hotpotqa:q2"]
    assert [row["sample_id"] for row in by_content["samples"]] == ["hotpotqa:q2"]


def test_normalized_corpus_returns_full_text(client, normalized):
    long_text = "x" * 800
    normalized.execute(
        "UPDATE corpus_doc SET text = ? WHERE dataset = 'hotpotqa' AND doc_id = '0'",
        (long_text,),
    )
    normalized.commit()

    body = client.get(
        "/api/datasets/hotpotqa/corpus", params={"q": "Rita Moreno", "limit": 1}
    ).json()

    assert body["total"] == 1
    assert body["docs"][0]["text"] == long_text
    assert "truncated" not in body["docs"][0]


def test_normalized_dataset_browsing_filters_and_pages_in_sqlite(
    client, normalized, monkeypatch
):
    """列表路由不能再调用会把整组样本或正文载入内存的旧接口。"""
    from akasha_platform.api import datasets as datasets_api

    def reject_full_load(*args, **kwargs):
        raise AssertionError("full dataset load is forbidden for paged routes")

    monkeypatch.setattr(datasets_api.data_store, "samples_of", reject_full_load)
    monkeypatch.setattr(datasets_api.data_store, "corpus_of", reject_full_load)

    samples = client.get(
        "/api/datasets/hotpotqa/samples",
        params={"q": "venice", "limit": 1},
    ).json()
    assert samples["total"] == 1
    assert samples["samples"][0]["dataset_sample_id"] == "q2"

    corpus = client.get(
        "/api/datasets/hotpotqa/corpus",
        params={"limit": 1, "offset": 1},
    ).json()
    assert corpus["total"] == 4
    assert len(corpus["docs"]) == 1
    assert corpus["docs"][0]["doc_id"] == "1"


@pytest.mark.parametrize(
    "path",
    [
        "/api/datasets/hotpotqa/raw?limit=0",
        "/api/tasks/1?after_id=-1",
    ],
)
def test_paged_routes_reject_invalid_bounds(client, path):
    assert client.get(path).status_code == 422


def test_task_tree_route(client):
    response = client.get("/api/task-tree")
    assert response.status_code == 200
    assert set(response.json()) == {"total_tasks", "inactive_total", "compiles", "unlinked_tasks"}


def test_provider_rename_updates_the_same_row(client):
    """带 id 的改名改的是那一条 —— 不带 id 会按 label 认行，于是变成新增。"""
    created = client.put(
        "/api/providers/judge",
        json={"label": "default", "base_url": "https://x/v1", "model": "m", "api_key": "secret"},
    ).json()

    body = client.put(
        "/api/providers/judge",
        json={"id": created["id"], "label": "gpt4", "base_url": "https://y/v1", "model": "m2"},
    ).json()
    assert body["id"] == created["id"]

    providers = client.get("/api/providers?role=judge").json()
    assert [(p["id"], p["label"], p["base_url"], p["model"]) for p in providers] == [
        (created["id"], "gpt4", "https://y/v1", "m2")
    ]
    assert providers[0]["api_key_set"] is True


def test_provider_rename_onto_a_taken_label_is_rejected(client):
    for label in ("a", "b"):
        client.put(
            "/api/providers/judge",
            json={"label": label, "base_url": "https://x/v1", "model": "m"},
        )
    first = next(
        provider
        for provider in client.get("/api/providers?role=judge").json()
        if provider["label"] == "a"
    )

    conflict = client.put(
        "/api/providers/judge",
        json={"id": first["id"], "label": "b", "base_url": "https://x/v1", "model": "m"},
    )
    assert conflict.status_code == 409
    assert len(client.get("/api/providers?role=judge").json()) == 2

    missing = client.put(
        "/api/providers/judge",
        json={"id": 999, "label": "c", "base_url": "https://x/v1", "model": "m"},
    )
    assert missing.status_code == 404


def test_provider_carries_no_sampling_params(client):
    """温度与长度上限不是配置项 —— judge 分数要可比，它们由 judge/client.py 固定。"""
    client.put(
        "/api/providers/judge",
        json={"label": "d", "base_url": "https://x/v1", "model": "m", "temperature": 0.9},
    )
    stored = client.get("/api/providers?role=judge").json()[0]
    assert "temperature" not in stored
    assert "max_tokens" not in stored


def test_model_config_put_fills_the_only_legal_provider(client, monkeypatch):
    """provider 不进表单，但 Akasha 的 PUT 要它，所以后端补上。"""
    client.put("/api/connection", json={"base_url": "http://x", "email": "e@x", "password": "p"})
    sent: dict = {}

    from akasha_platform.api import config as config_api

    class Fake(_AkashaStub):
        def put_model_config(self, feature, payload):
            sent.update({"feature": feature, "payload": payload})
            return {"ok": True}

    monkeypatch.setattr(config_api, "AkashaClient", Fake)
    body = client.put("/api/model-configs/answer", json={"model": "m", "baseUrl": "u"}).json()
    assert body["requires_new_compile"] is False
    assert sent["payload"] == {"provider": "openai-compatible", "model": "m", "baseUrl": "u"}


def test_bad_role_is_rejected(client):
    assert (
        client.put("/api/providers/bogus", json={"base_url": "u", "model": "m"}).status_code == 422
    )
    assert client.get("/api/providers?role=bogus").status_code == 422


def test_missing_records_return_404(client):
    for path in ("/api/evals/9", "/api/attributions/9", "/api/compiles/9/docs"):
        assert client.get(path).status_code == 404, path


def test_compile_query_and_eval_records_are_searchable(client, normalized):
    compile_id = _compile_run(
        normalized, run_id="searchable", datasets=["hotpotqa"], qa_limit=2
    )
    compile_store.replace_compile_subset(
        normalized,
        compile_id,
        "hotpotqa",
        [],
        [
            {"doc_id": "0", "is_gold": True},
            {"doc_id": "2", "is_gold": False},
        ],
    )
    compile_store.record_page(
        normalized, compile_id, "hotpotqa", "0", page_id="page-rita", error=None
    )
    compile_store.record_page(
        normalized, compile_id, "hotpotqa", "2", page_id="page-venice", error=None
    )
    from akasha_benchmark.store import dumps

    compile_store.update_compile_run(
        normalized,
        compile_id,
        model_configs_json=dumps(
            {
                "configs": [
                    {"feature": "compiler", "model": "compiler-model"}
                ]
            }
        ),
    )
    query_id = query_store.create_query_run(
        normalized,
        name="search-query",
        compile_id=compile_id,
        concurrency=1,
        model_configs={
            "configs": [{"feature": "answer", "model": "answer-model"}]
        },
    )
    cases = (
        ("sample-rita", "Who won the award?", "Rita Moreno", "knowledge"),
        ("sample-venice", "Which city hosts the festival?", "Venice", "general"),
    )
    eval_id = eval_store.create_eval_run(
        normalized,
        name="search-eval",
        query_id=query_id,
        ks=[2],
        metrics=["em"],
        judge_provider_id=None,
    )
    for sample_id, question, answer, answer_mode in cases:
        query_store.record_response(
            normalized,
            query_id,
            sample_id=sample_id,
            dataset="hotpotqa",
            question=question,
            http_status=200,
            latency_ms=1,
            error=None,
            response={"answerMode": answer_mode, "answer": answer},
        )
        eval_store.record_sample_eval(
            normalized,
            eval_id,
            sample_id=sample_id,
            dataset="hotpotqa",
            answer_mode=answer_mode,
            http_status=200,
            answer=answer,
            detail={"question": question},
            metrics={"em": 1.0},
        )
    attribution_id = attribution_store.create_attribution_run(
        normalized,
        name="search-attribution",
        eval_id=eval_id,
        report_provider_id=None,
    )
    attribution_store.record_attribution(
        normalized,
        attribution_id,
        sample_id="sample-venice",
        root_cause="not_a_failure",
        evidence={},
    )
    normalized.commit()

    docs = client.get(
        f"/api/compiles/{compile_id}/docs", params={"q": "venice", "limit": 1}
    ).json()
    assert docs["total"] == 1
    assert docs["docs"][0]["doc_id"] == "2"

    responses = client.get(
        f"/api/queries/{query_id}/responses",
        params={"q": "city hosts", "limit": 1},
    ).json()
    assert responses["total"] == 1
    assert responses["responses"][0]["sample_id"] == "sample-venice"

    samples = client.get(
        f"/api/evals/{eval_id}/samples",
        params={"answer_mode": "general", "limit": 1},
    ).json()
    assert samples["total"] == 1
    assert samples["samples"][0]["sample_id"] == "sample-venice"
    assert samples["count_by_answer_mode"] == {"general": 1, "knowledge": 1}

    tree = client.get("/api/compiles").json()["compiles"]
    compile_run = next(run for run in tree if run["id"] == compile_id)
    query_run = next(run for run in compile_run["queries"] if run["id"] == query_id)
    eval_run = next(run for run in query_run["evals"] if run["id"] == eval_id)
    attribution_run = next(
        run for run in eval_run["attributions"] if run["id"] == attribution_id
    )
    assert (query_run["model_label"], query_run["sample_count"], query_run["success_count"]) == (
        "answer-model",
        2,
        2,
    )
    assert (eval_run["model_label"], eval_run["sample_count"], eval_run["success_count"]) == (
        "确定性指标",
        2,
        2,
    )
    assert (
        attribution_run["model_label"],
        attribution_run["sample_count"],
        attribution_run["success_count"],
    ) == ("规则归因", 2, 1)


def test_compile_tree_uses_constant_queries_and_no_postgres(client, normalized, monkeypatch):
    from akasha_platform.api import runs as runs_api

    for index in range(4):
        compile_id = _compile_run(normalized, run_id=f"tree-{index}")
        query_id = query_store.create_query_run(
            normalized,
            name=f"tree-query-{index}",
            compile_id=compile_id,
            concurrency=1,
            model_configs={},
        )
        eval_id = eval_store.create_eval_run(
            normalized,
            name=f"tree-eval-{index}",
            query_id=query_id,
            ks=[2],
            metrics=["em"],
            judge_provider_id=None,
            concurrency=3,
        )
        attribution_store.create_attribution_run(
            normalized,
            name=f"tree-attribution-{index}",
            eval_id=eval_id,
            report_provider_id=None,
        )
    normalized.commit()

    selects: list[str] = []

    @contextmanager
    def traced_db(_request):
        connection = connect(client.app.state.settings.db_path, read_only=True)
        connection.set_trace_callback(
            lambda sql: selects.append(sql)
            if sql.lstrip().upper().startswith("SELECT")
            else None
        )
        try:
            yield connection
        finally:
            connection.close()

    monkeypatch.setattr(runs_api, "db", traced_db)

    response = client.get("/api/compiles")

    assert response.status_code == 200
    assert len(response.json()["compiles"]) == 4
    evaluation = response.json()["compiles"][0]["queries"][0]["evals"][0]
    assert evaluation["concurrency"] == 3
    assert len(selects) <= 15


def test_compile_cleanup_cancels_remote_run_and_keeps_space(client, db_path, monkeypatch):
    calls: list[str] = []

    class FakeAkasha(_AkashaStub):
        def run_diagnostics(self, space_ids, *, limit=50):
            return {"items": [{"runId": "run-1", "status": "compiling"}]}
        def cancel_compile_run(self, run_id, reason):
            calls.append(run_id)
            return {"disposition": "cancelled", "runId": run_id, "status": "cancelled", "removedJobCount": 2}

    monkeypatch.setattr("akasha_platform.api.runs.AkashaClient", FakeAkasha)
    connection = connect(db_path)
    try:
        compile_id = _compile_run(connection)
        compile_store.update_compile_run(connection, compile_id, space_id="space-1")
        connection.commit()
    finally:
        connection.close()

    body = client.delete(f"/api/compiles/{compile_id}").json()
    assert body["deleted"] == 1
    assert body["space_id"] == "space-1"
    assert body["cancelled_runs"] == 1
    assert body["removed_bullmq_jobs"] == 2
    assert calls == ["run-1"]
    assert "空间没有删除" in body["note"]


def test_cleanup_refused_while_a_task_writes_the_record(client, db_path):
    connection = connect(db_path)
    try:
        compile_id = _compile_run(connection)
        task_id = task_store.create_task(connection, stage="compile", params={})
        task_store.transition(connection, task_id, task_store.RUNNING)
        task_store.set_task_target(connection, task_id, "compile", compile_id)
        connection.commit()
    finally:
        connection.close()
    assert client.delete(f"/api/compiles/{compile_id}").status_code == 409


def test_auth_token_is_required_when_set(db_path):
    app = create_app(Settings(db_path=db_path, auth_token="secret"))
    client = TestClient(app)
    assert client.get("/api/health").status_code == 401
    assert client.get("/api/health", headers={"X-Auth-Token": "wrong"}).status_code == 401
    assert client.get("/api/health", headers={"X-Auth-Token": "secret"}).status_code == 200


def test_auth_token_allows_cors_preflight_without_credentials(db_path):
    app = create_app(Settings(db_path=db_path, auth_token="secret"))
    client = TestClient(app)
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-auth-token",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_non_loopback_without_token_refuses_to_start(db_path):
    """这个服务持有 Akasha 管理员凭据并能起长任务，不能裸奔在 0.0.0.0 上。"""
    with pytest.raises(RuntimeError, match="拒绝绑定"):
        create_app(Settings(db_path=db_path, host="0.0.0.0"))


@pytest.mark.parametrize("outcome", ["succeeded", "failed", "paused", "invalid_result"])
def test_task_and_run_finish_together(settings, db, query_id, monkeypatch, outcome):
    from akasha_benchmark.task import Paused

    db.commit()

    def stage(ctx):
        ctx.bind("query", query_id)
        if outcome == "failed":
            raise RuntimeError("stage failed")
        if outcome == "paused":
            raise Paused("checkpoint")

    _register(monkeypatch, "unit", stage)
    runner = TaskRunner(settings)
    if outcome == "invalid_result":

        def reject_result(*args):
            raise ValueError("invalid result")

        monkeypatch.setattr(runner, "_verify", reject_result)
    task = runner.start("unit", {})
    expected = "failed" if outcome == "invalid_result" else outcome
    finished = _wait(settings, task["id"], {expected})
    run = query_store.get_query_run(db, query_id)
    assert run["status"] == finished["status"]
    assert run["finished_at"] is not None


def test_recovery_pauses_bound_run(settings, db, query_id):
    task_id = task_store.create_task(db, stage="query", params={})
    task_store.set_task_target(db, task_id, "query", query_id)
    task_store.transition(db, task_id, task_store.RUNNING)
    db.commit()
    assert TaskRunner(settings).recover() == 1
    assert task_store.get_task(db, task_id)["status"] == "paused"
    assert query_store.get_query_run(db, query_id)["status"] == "paused"


@pytest.mark.parametrize("bound", [False, True], ids=["queued-input", "bound-output"])
@pytest.mark.parametrize(
    "parent,stage",
    [
        ("compile", "query"),
        ("compile", "evaluate"),
        ("query", "attribute"),
        ("eval", "attribute"),
    ],
)
def test_cleanup_protects_active_descendants(
    client, db, compile_id, query_id, eval_id, parent, stage, bound
):
    from akasha_benchmark.store import attribution_store

    attribution_id = attribution_store.create_attribution_run(
        db,
        name="a",
        eval_id=eval_id,
        report_provider_id=None,
    )
    ids = {"compile": compile_id, "query": query_id, "eval": eval_id, "attribution": attribution_id}
    kind = {"query": "query", "evaluate": "eval", "attribute": "attribution"}[stage]
    input_kind, param = run_store.STAGE_INPUTS[stage]
    task_id = task_store.create_task(db, stage=stage, params={param: ids[input_kind]})
    if bound:
        task_store.set_task_target(db, task_id, kind, ids[kind])
        task_store.transition(db, task_id, task_store.RUNNING)
    db.commit()

    route = {"compile": "compiles", "query": "queries", "eval": "evals"}[parent]
    assert client.delete(f"/api/{route}/{ids[parent]}").status_code == 409
    assert run_store.get_run(db, parent, ids[parent]) is not None
    task_store.transition(db, task_id, task_store.PAUSED)
    db.commit()
    assert client.delete(f"/api/{route}/{ids[parent]}").status_code == 200
    assert attribution_store.get_attribution_run(db, attribution_id) is None


def test_concurrent_starts_admit_only_one_task(settings, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    _register(monkeypatch, "unit", lambda ctx: None)
    runner = TaskRunner(settings)
    monkeypatch.setattr(runner, "_spawn", lambda *args: None)
    barrier = threading.Barrier(2)

    def start():
        barrier.wait()
        try:
            return runner.start("unit", {})["id"]
        except TaskRejected:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: start(), range(2)))
    assert sum(task_id is not None for task_id in results) == 1


def test_compile_and_query_tasks_can_run_concurrently(settings, db, monkeypatch):
    monkeypatch.setattr(TaskRunner, "_spawn", lambda *args: None)
    runner = TaskRunner(settings)
    compile_task = runner.start("compile", {"datasets": ["hotpotqa"]})
    task_store.transition(db, int(compile_task["id"]), task_store.RUNNING)
    db.commit()
    query_task = runner.start("query", {"compile_id": 1})
    assert query_task["stage"] == "query"


@pytest.mark.parametrize(
    "stage,feature,allowed",
    [
        ("compile", "compiler", False),
        ("compile", "embedding", False),
        ("compile", "image", False),
        ("compile", "answer", True),
        ("query", "embedding", False),
        ("query", "answer", False),
        ("query", "compiler", True),
        ("query", "image", True),
    ],
)
def test_remote_model_config_lock_is_feature_scoped(
    client, db, monkeypatch, stage, feature, allowed
):
    from akasha_platform.api import config as config_api

    class Fake(_AkashaStub):
        def put_model_config(self, selected_feature, payload):
            return {"feature": selected_feature}

    monkeypatch.setattr(config_api, "AkashaClient", Fake)
    task_id = task_store.create_task(db, stage=stage, params={})
    task_store.transition(db, task_id, task_store.RUNNING)
    db.commit()
    response = client.put(
        f"/api/model-configs/{feature}",
        json={"model": "m", "baseUrl": "https://x/v1"},
    )
    assert response.status_code == (200 if allowed else 409)


@pytest.mark.parametrize("feature,expected", [("answer", 200), ("embedding", 409)])
def test_saved_model_apply_uses_the_same_feature_lock(
    client, db, monkeypatch, feature, expected
):
    from akasha_platform.api import config as config_api

    model_id = config_store.upsert_model_provider(
        db,
        purpose=feature,
        label=f"{feature}-lock",
        base_url="https://x/v1",
        model="m",
        api_key="",
    )
    task_id = task_store.create_task(db, stage="compile", params={})
    task_store.transition(db, task_id, task_store.RUNNING)
    db.commit()

    class Fake(_AkashaStub):
        def put_model_config(self, selected_feature, payload):
            return {"feature": selected_feature}

    monkeypatch.setattr(config_api, "AkashaClient", Fake)
    assert client.post(f"/api/akasha-models/{model_id}/apply").status_code == expected
