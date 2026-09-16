"""平台层：任务生命周期（暂停/继续/清理）、审计日志留存、路由与鉴权。"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from akasha_benchmark.datasets import DATASET_NAMES
from akasha_benchmark.stages import STAGES, StageSpec
from akasha_benchmark.store import (
    compile_store,
    config_store,
    connect,
    query_store,
    run_store,
    task_store,
)
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


@pytest.mark.parametrize("action", ["pause", "cleanup"])
def test_compile_task_action_cancels_remote_bullmq_run(settings, db, monkeypatch, action):
    """暂停和清理通过 Akasha 控制面取消精确 Run，不暂停共享 BullMQ 队列。"""
    calls: list[tuple[str, str]] = []

    class FakeAkasha:
        def __init__(self, config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def login(self):
            pass

        def cancel_compile_run(self, run_id, reason):
            calls.append((run_id, reason))
            return {"disposition": "cancelled", "removedJobCount": 2}

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


def test_cleanup_finds_active_remote_run_for_legacy_compile_task(settings, db, monkeypatch):
    """旧任务没保存 runId 时，用绑定空间找活动 Run。"""
    calls: list[str] = []

    class FakeAkasha:
        def __init__(self, config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def login(self):
            pass

        def run_diagnostics(self, space_ids, *, limit=50):
            return {"items": [{"runId": "active-run", "status": "compiling"}]}

        def cancel_compile_run(self, run_id, reason):
            calls.append(run_id)
            return {"disposition": "cancelled", "removedJobCount": 1}

    monkeypatch.setattr("akasha_platform.tasks.AkashaClient", FakeAkasha)
    compile_id = compile_store.create_compile_run(
        db, run_id="legacy", datasets=[], seed=1, qa_limit=1, negatives_ratio=1.0
    )
    compile_store.update_compile_run(db, compile_id, space_id="space-1")
    task_id = task_store.create_task(db, stage="compile", params={})
    task_store.set_task_target(db, task_id, "compile", compile_id)
    task_store.transition(db, task_id, task_store.FAILED)
    db.commit()

    TaskRunner(settings).cleanup(task_id)
    assert calls == ["active-run"]


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


def test_compile_allows_three_active_tasks(settings, monkeypatch):
    runner = TaskRunner(settings)
    monkeypatch.setattr(runner, "_spawn", lambda *args: None)

    tasks = [runner.start("compile", {}) for _ in range(3)]
    assert len({task["id"] for task in tasks}) == 3
    with pytest.raises(TaskRejected, match="并发上限 3"):
        runner.start("compile", {})


def test_compile_resume_obeys_three_task_limit(settings, db, monkeypatch):
    paused_id = task_store.create_task(db, stage="compile", params={})
    task_store.transition(db, paused_id, task_store.PAUSED)
    for _ in range(3):
        task_id = task_store.create_task(db, stage="compile", params={})
        task_store.transition(db, task_id, task_store.RUNNING)
    db.commit()

    runner = TaskRunner(settings)
    monkeypatch.setattr(runner, "_spawn", lambda *args: None)
    with pytest.raises(TaskRejected, match="并发上限 3"):
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


def test_health_and_stages(client):
    assert client.get("/api/health").json()["ok"] is True
    stages = client.get("/api/stages").json()
    assert {entry["stage"] for entry in stages} == set(STAGES)


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
            connection,
            role="judge",
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


def test_provider_stores_and_returns_concurrency(client):
    client.put(
        "/api/providers/judge",
        json={"label": "d", "base_url": "https://x/v1", "model": "m", "concurrency": 4},
    )
    provider = client.get("/api/providers?role=judge").json()[0]
    assert provider["concurrency"] == 4

    from akasha_benchmark.judge.providers import resolve_provider

    connection = connect(client.app.state.settings.db_path)
    try:
        provider_id = config_store.upsert_provider(
            connection,
            role="judge",
            label="k",
            base_url="https://x/v1",
            model="m",
            api_key="key",
            concurrency=3,
        )
        connection.commit()
        resolved = resolve_provider(connection, provider_id, "judge")
    finally:
        connection.close()
    assert resolved.concurrency == 3


def test_akasha_config_group_roundtrip_hides_keys(client):
    client.put(
        "/api/akasha-configs",
        json={
            "label": "g1",
            "configs": {
                "compiler": {"model": "c", "baseUrl": "https://x/v1", "apiKey": "secret"},
            },
        },
    )
    body = client.get("/api/akasha-configs").json()
    group = body["groups"][0]
    assert group["label"] == "g1"
    assert group["configs"]["compiler"]["apiKeySet"] is True
    assert "apiKey" not in group["configs"]["compiler"]

    # 密钥留空保留原值。
    client.put(
        "/api/akasha-configs",
        json={"id": group["id"], "label": "g1", "configs": {"compiler": {"model": "c2"}}},
    )
    connection = connect(client.app.state.settings.db_path, read_only=True)
    try:
        stored = config_store.list_config_groups(connection)[0]
    finally:
        connection.close()
    from akasha_benchmark.store import loads

    configs = loads(stored["configs_json"], {})
    assert configs["compiler"]["apiKey"] == "secret"
    assert configs["compiler"]["model"] == "c2"


def test_akasha_config_delete(client):
    client.put("/api/akasha-configs", json={"label": "a", "configs": {}})
    client.put("/api/akasha-configs", json={"label": "b", "configs": {}})
    groups = {g["label"]: g["id"] for g in client.get("/api/akasha-configs").json()["groups"]}

    assert client.delete(f"/api/akasha-configs/{groups['a']}").status_code == 200
    labels = [g["label"] for g in client.get("/api/akasha-configs").json()["groups"]]
    assert labels == ["b"]


def test_akasha_config_apply_pushes_all_features(client, monkeypatch):
    client.put("/api/connection", json={"base_url": "http://x", "email": "e@x", "password": "p"})
    client.put(
        "/api/akasha-configs",
        json={
            "label": "g",
            "configs": {"answer": {"model": "a", "baseUrl": "https://x/v1", "apiKey": "k"}},
        },
    )
    group_id = client.get("/api/akasha-configs").json()["groups"][0]["id"]

    from akasha_platform.api import config as config_api

    pushed = []

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
            pushed.append((feature, payload))
            return {}

    monkeypatch.setattr(config_api, "AkashaClient", Fake)
    body = client.post(f"/api/akasha-configs/{group_id}/apply").json()
    assert set(body["applied"]) == set(config_api.FEATURES)
    assert all(payload["provider"] == "openai-compatible" for _, payload in pushed)

    # 应用后该组即为选中。
    selected = [g for g in client.get("/api/akasha-configs").json()["groups"] if g["selected"]]
    assert [g["id"] for g in selected] == [group_id]


def test_config_import_round_trips_export(client):
    client.put("/api/connection", json={"password": "pw", "email": "e@x", "base_url": "http://y"})
    client.put(
        "/api/providers/judge",
        json={"label": "d", "base_url": "https://x/v1", "model": "m", "api_key": "sk"},
    )
    client.put(
        "/api/akasha-configs",
        json={"label": "g", "configs": {"answer": {"model": "a", "apiKey": "gk"}}},
    )
    exported = client.get("/api/config/export").json()
    assert exported["connection"]["password"] == "pw"
    assert exported["providers"][0]["api_key"] == "sk"
    assert exported["akasha_configs"][0]["configs"]["answer"]["apiKey"] == "gk"

    # 清一遍再导回：导入后应与导出前一致。
    client.put("/api/connection", json={"password": "", "email": "", "base_url": ""})
    assert client.post("/api/config/import", json=exported).status_code == 200

    back = client.get("/api/config/export").json()
    assert back["connection"]["password"] == "pw"
    assert back["connection"]["base_url"] == "http://y"
    assert back["providers"][0]["api_key"] == "sk"
    assert back["akasha_configs"][0]["configs"]["answer"]["apiKey"] == "gk"


def test_config_import_rejects_bad_role(client):
    payload = {"providers": [{"role": "nonsense", "label": "d", "base_url": "u", "model": "m"}]}
    assert client.post("/api/config/import", json=payload).status_code == 422


def test_connection_test_flags_compiles_in_another_workspace(client, db_path, monkeypatch):
    """预检：不必等起了任务才发现这些编译在当前连接下用不了。"""
    connection = connect(db_path)
    try:
        compile_id = compile_store.create_compile_run(
            connection, run_id="r1", datasets=[], seed=1, qa_limit=1, negatives_ratio=1.0
        )
        compile_store.update_compile_run(
            connection, compile_id, space_id="s1", workspace_id="w-original"
        )
        connection.commit()
    finally:
        connection.close()
    client.put("/api/connection", json={"base_url": "http://x", "email": "e@x", "password": "p"})

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
    assert (
        client.put("/api/providers/bogus", json={"base_url": "u", "model": "m"}).status_code == 422
    )
    assert client.get("/api/providers?role=bogus").status_code == 422


def test_missing_records_return_404(client):
    for path in ("/api/evals/9", "/api/attributions/9", "/api/compiles/9/docs"):
        assert client.get(path).status_code == 404, path


def test_compile_cleanup_cancels_remote_run_and_keeps_space(client, db_path, monkeypatch):
    calls: list[str] = []

    class FakeAkasha:
        def __init__(self, config): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def login(self): pass
        def run_diagnostics(self, space_ids, *, limit=50):
            return {"items": [{"runId": "run-1", "status": "compiling"}]}
        def cancel_compile_run(self, run_id, reason):
            calls.append(run_id)
            return {"disposition": "cancelled", "removedJobCount": 2}

    monkeypatch.setattr("akasha_platform.api.runs.AkashaClient", FakeAkasha)
    connection = connect(db_path)
    try:
        compile_id = compile_store.create_compile_run(
            connection, run_id="r", datasets=[], seed=1, qa_limit=1, negatives_ratio=1.0
        )
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
        compile_id = compile_store.create_compile_run(
            connection, run_id="r", datasets=[], seed=1, qa_limit=1, negatives_ratio=1.0
        )
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
        metric="em",
        sample_limit=1,
        provider_id=None,
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
