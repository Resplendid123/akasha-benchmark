"""平台层：任务生命周期（暂停/继续/清理）、审计日志留存、路由与鉴权。"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from akasha_benchmark.stages import STAGES, StageSpec
from akasha_benchmark.store import connect, run_store, task_store
from akasha_platform.main import create_app
from akasha_platform.settings import Settings
from akasha_platform.tasks import TaskRejected, TaskRunner


@pytest.fixture
def settings(db_path) -> Settings:
    return Settings(db_path=db_path)


@pytest.fixture
def client(settings) -> TestClient:
    return TestClient(create_app(settings))


# ------------------------------------------------------------ 任务生命周期


def _register(monkeypatch, name: str, run) -> None:
    monkeypatch.setitem(
        STAGES, name, StageSpec(name=name, label=name, run=run, params={"marker": str})
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
        task_store.start_task(connection, task_id)
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


def test_health_and_stages(client):
    assert client.get("/api/health").json()["ok"] is True
    stages = client.get("/api/stages").json()
    assert {entry["stage"] for entry in stages} == set(STAGES)


def test_datasets_route_reports_missing_files(client):
    body = client.get("/api/datasets").json()
    assert len(body["datasets"]) == 4
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
    client.put("/api/connection", json={"base_url": "http://x", "email": "a@b"})
    body = client.get("/api/connection").json()
    assert body["base_url"] == "http://x"
    assert body["password"] == ""

    client.put("/api/connection", json={"password": "p"})
    body = client.get("/api/connection").json()
    assert body["base_url"] == "http://x"
    assert body["password"] == "p"

    assert client.put("/api/connection", json={"concurrency": "many"}).status_code == 422


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

        stored = config_store.list_providers(connection, "judge")[0]
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
    system, user = sent[0]
    assert user.startswith("hi")
    assert "json" in user.lower()


def test_provider_probe_reports_missing_key(client):
    """没配密钥时也回 200 —— 那是配置问题，同样是要报告的结果。"""
    from akasha_benchmark.store import config_store

    connection = connect(client.app.state.settings.db_path)
    try:
        provider_id = config_store.upsert_provider(
            connection, role="judge", label="nokey", base_url="https://x/v1",
            model="m", api_key="",
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


def test_connection_test_flags_compiles_in_another_workspace(client, db_path, monkeypatch):
    """预检：不必等起了任务才发现这些编译在当前连接下用不了。"""
    connection = connect(db_path)
    try:
        compile_id = run_store.create_compile_run(
            connection, run_id="r1", datasets=[], seed=1, qa_limit=1, negatives_ratio=1.0
        )
        run_store.update_compile_run(
            connection, compile_id, space_id="s1", workspace_id="w-original"
        )
        connection.commit()
    finally:
        connection.close()
    client.put(
        "/api/connection", json={"base_url": "http://x", "email": "e@x", "password": "p"}
    )

    from akasha_platform.api import config as config_api

    class Fake:
        def __init__(self, cfg):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def login(self):
            pass

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


def test_provider_rename_updates_the_same_row(client):
    """带 id 的改名改的是那一条 —— 不带 id 会按 label 认行，于是变成新增。"""
    created = client.put(
        "/api/providers/judge",
        json={"label": "default", "base_url": "https://x/v1", "model": "m", "api_key": "secret"},
    ).json()

    body = client.put(
        "/api/providers/judge",
        json={"id": created["id"], "label": "gpt4", "base_url": "https://x/v1", "model": "m"},
    ).json()
    assert body["id"] == created["id"]

    providers = client.get("/api/providers?role=judge").json()
    assert [p["label"] for p in providers] == ["gpt4"]
    assert providers[0]["api_key_set"] is True


def test_provider_rename_onto_a_taken_label_is_rejected(client):
    for label in ("a", "b"):
        client.put(
            "/api/providers/judge",
            json={"label": label, "base_url": "https://x/v1", "model": "m"},
        )
    first = client.get("/api/providers?role=judge").json()[0]

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

    class Fake:
        def __init__(self, cfg):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def login(self):
            pass

        def put_model_config(self, feature, payload):
            sent.update({"feature": feature, "payload": payload})
            return {"ok": True}

    monkeypatch.setattr(config_api, "AkashaClient", Fake)
    body = client.put("/api/model-configs/answer", json={"model": "m", "baseUrl": "u"}).json()
    assert body["requires_new_compile"] is False
    assert sent["payload"] == {"provider": "openai-compatible", "model": "m", "baseUrl": "u"}


def test_bad_role_is_rejected(client):
    assert client.put("/api/providers/bogus", json={"base_url": "u", "model": "m"}).status_code == 422
    assert client.get("/api/providers?role=bogus").status_code == 422


def test_missing_records_return_404(client):
    for path in ("/api/evals/9", "/api/attributions/9", "/api/compiles/9/docs"):
        assert client.get(path).status_code == 404, path


def test_compile_cleanup_reports_untouched_space(client, db_path):
    connection = connect(db_path)
    try:
        compile_id = run_store.create_compile_run(
            connection, run_id="r", datasets=[], seed=1, qa_limit=1, negatives_ratio=1.0
        )
        run_store.update_compile_run(connection, compile_id, space_id="space-1")
        connection.commit()
    finally:
        connection.close()

    body = client.delete(f"/api/compiles/{compile_id}").json()
    assert body["deleted"] == 1
    assert body["space_id"] == "space-1"
    assert "没有删除" in body["note"]


def test_cleanup_refused_while_a_task_writes_the_record(client, db_path):
    connection = connect(db_path)
    try:
        compile_id = run_store.create_compile_run(
            connection, run_id="r", datasets=[], seed=1, qa_limit=1, negatives_ratio=1.0
        )
        task_id = task_store.create_task(connection, stage="compile", params={})
        task_store.start_task(connection, task_id)
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


def test_non_loopback_without_token_refuses_to_start(db_path):
    """这个服务持有 Akasha 管理员凭据并能起长任务，不能裸奔在 0.0.0.0 上。"""
    with pytest.raises(RuntimeError, match="拒绝绑定"):
        create_app(Settings(db_path=db_path, host="0.0.0.0"))
