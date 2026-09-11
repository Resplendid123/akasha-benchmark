"""连接配置、provider 与任务清理。

配置从 ``akasha.config.json`` 搬进库、去掉环境变量旁路、再收成单例的取舍是
明确的：配置要能在 UI 里填改，而只有一份就不会有「哪一份是真的」这个问题。
代价是 ``akasha_bench.db`` 成为凭据文件 —— 所以这一组里相当一部分测的是
「密钥不会从 API 漏出去」。

连接与层的绑定、workspace 闸门另见 ``test_connections.py``。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from akasha_benchmark.store import connect, repo
from akasha_benchmark.store.migrate import migrate
from fastapi.testclient import TestClient

from akasha_platform.main import create_app
from akasha_platform.settings import Settings


@pytest.fixture
def client(tmp_path: Path):
    db = tmp_path / "t.db"
    migrate(db, verbose=False)
    api = TestClient(create_app(Settings(db_path=db, web_dist=tmp_path / "none")))
    yield api, db


# --- 连接读写 ----------------------------------------------------------------


def test_connection_round_trips_through_the_api(client):
    api, _ = client
    response = api.put(
        "/api/connection",
        json={"base_url": "http://akasha:3000", "email": "eval@example.com", "password": "pw"},
    )
    assert response.status_code == 200
    # **取值不回传。**
    assert "pw" not in response.text
    assert response.json()["connection"]["password_set"] is True

    current = api.get("/api/connection").json()
    assert current["base_url"] == "http://akasha:3000"
    assert "password" not in current
    assert "database_url" not in current


def test_a_partial_update_does_not_wipe_the_password(client):
    """只改 base_url 不该把密码清掉。

    UI 拿不到明文，表单里那一格提交上来必然是空的 —— 当成清空的话，
    每次改地址都会顺手删掉密码，而那个失效要等到下次 ingest 登录失败才发现。
    """
    api, db = client
    api.put("/api/connection", json={"base_url": "http://a", "password": "pw"})
    api.put("/api/connection", json={"base_url": "http://b", "password": ""})

    connection = connect(db, read_only=True)
    row = repo.get_connection_row(connection)
    connection.close()
    assert row["base_url"] == "http://b"
    assert row["password"] == "pw"


def test_clearing_a_secret_has_to_be_explicit(client):
    api, db = client
    api.put("/api/connection", json={"password": "pw"})
    api.put("/api/connection", json={"clear": ["password"]})

    connection = connect(db, read_only=True)
    assert repo.get_connection_row(connection)["password"] == ""
    connection.close()


def test_unknown_connection_keys_are_dropped(client):
    api, db = client
    assert api.put("/api/connection", json={"base_url": "http://a", "evil": "x"}).status_code == 200
    connection = connect(db, read_only=True)
    assert "evil" not in repo.get_connection_row(connection).keys()
    connection.close()


def test_bad_types_are_rejected_with_422(client):
    api, _ = client
    assert api.put("/api/connection", json={"concurrency": "many"}).status_code == 422


def test_test_endpoint_refuses_without_credentials(client):
    """没填凭据就点「测试连接」，要给一句能指向缺什么的话。"""
    api, _ = client
    response = api.post("/api/connection/test")
    assert response.status_code == 422
    assert "password" in response.json()["detail"]


def test_model_configs_refuse_without_credentials(client):
    """Akasha 那边的模型配置也在配置层里，同样要先能连上。"""
    api, _ = client
    assert api.get("/api/model-configs").status_code == 422


# --- provider ----------------------------------------------------------------


def test_provider_api_key_is_never_returned(client):
    api, _ = client
    api.put(
        "/api/providers/judge",
        json={"label": "local", "base_url": "http://llm/v1", "model": "m", "api_key": "sk-secret"},
    )
    listed = api.get("/api/providers/judge")
    assert listed.status_code == 200
    assert "sk-secret" not in listed.text
    assert listed.json()[0]["api_key_set"] is True


def test_provider_update_keeps_the_existing_key(client):
    """改模型名不该把密钥清掉（同连接那条的理由）。"""
    api, db = client
    api.put(
        "/api/providers/judge",
        json={"label": "l", "base_url": "http://llm/v1", "model": "m1", "api_key": "sk-1"},
    )
    api.put(
        "/api/providers/judge",
        json={"label": "l", "base_url": "http://llm/v1", "model": "m2", "api_key": ""},
    )
    connection = connect(db, read_only=True)
    record = repo.get_model_provider(connection, "judge", "l")
    connection.close()
    assert record["model"] == "m2"
    assert record["api_key"] == "sk-1"


def test_provider_role_is_validated(client):
    api, _ = client
    response = api.put("/api/providers/nonsense", json={"base_url": "http://x", "model": "m"})
    assert response.status_code == 422


def test_resolve_provider_needs_a_stored_key(client):
    """密钥只有一条来路：库里那份。

    曾经支持「存一个环境变量名、运行时从那里读」，删了 —— 同一份密钥有两个
    来源时，「填了但没生效」查不出来。
    """
    from akasha_benchmark.judge.client import JudgeConfigError
    from akasha_benchmark.judge.run import resolve_provider

    api, db = client
    api.put(
        "/api/providers/analysis",
        json={"label": "l", "base_url": "http://llm/v1", "model": "m", "api_key": "sk-x"},
    )
    connection = connect(db)
    provider = resolve_provider(connection, "l", role="analysis")
    assert provider.resolve_key() == "sk-x"

    # 密钥被清空后要给一句明确的错，而不是拿空 key 去打端点。
    repo.upsert_model_provider(
        connection, role="analysis", label="l", base_url="http://llm/v1", model="m", api_key=""
    )
    connection.commit()
    with pytest.raises(JudgeConfigError, match="has no api key"):
        resolve_provider(connection, "l", role="analysis")
    connection.close()


def test_resolve_provider_reports_a_missing_configuration(client):
    from akasha_benchmark.judge.client import JudgeConfigError
    from akasha_benchmark.judge.run import resolve_provider

    _, db = client
    connection = connect(db)
    with pytest.raises(JudgeConfigError, match="no analysis model provider"):
        resolve_provider(connection, None, role="analysis")
    connection.close()


# --- 任务清理 ----------------------------------------------------------------


def test_running_tasks_cannot_be_cleaned_up(client):
    """清理的语义是「这条记录不用看了」，不是「停掉它」。

    删一个在跑的任务会留下一个没人认领的子进程，而它还在往库里写。
    """
    api, db = client
    connection = connect(db)
    task_id = repo.create_task(connection, stage="ingest", argv=["x"])
    repo.start_task(connection, task_id, pid=1234, log_path="x.log")
    connection.commit()
    connection.close()

    response = api.delete(f"/api/tasks/{task_id}")
    assert response.status_code == 409
    assert "cancel it before" in response.json()["detail"]


def test_finished_tasks_can_be_cleaned_up(client):
    api, db = client
    connection = connect(db)
    task_id = repo.create_task(connection, stage="normalize", argv=["x"])
    repo.finish_task(connection, task_id, status="succeeded", exit_code=0, error=None)
    connection.commit()
    connection.close()

    assert api.delete(f"/api/tasks/{task_id}").json()["deleted"] == 1
    assert api.get(f"/api/tasks/{task_id}").status_code == 404


def test_bulk_cleanup_leaves_running_tasks_alone(client):
    api, db = client
    connection = connect(db)
    done = repo.create_task(connection, stage="normalize", argv=["x"])
    repo.finish_task(connection, done, status="failed", exit_code=1, error="e")
    running = repo.create_task(connection, stage="ingest", argv=["y"])
    repo.start_task(connection, running, pid=999, log_path=None)
    connection.commit()
    connection.close()

    assert api.post("/api/tasks/cleanup/finished").json()["deleted"] == 1
    assert api.get(f"/api/tasks/{running}").status_code == 200


def test_task_list_exposes_the_run_config_arguments(client):
    """argv 里只剩一个 id，所以参数要单独给 —— 否则任务列表看不出跑的是什么。"""
    api, db = client
    connection = connect(db)
    run_config_id = repo.create_run_config(connection, "subset", {"label": "runX", "seed": 7})
    task_id = repo.create_task(
        connection, stage="subset", argv=["py", "-m", "m", "--run-config", str(run_config_id)]
    )
    repo.finish_task(connection, task_id, status="succeeded", exit_code=0, error=None)
    connection.commit()
    connection.close()

    listed = api.get("/api/tasks").json()
    assert listed[0]["args"] == {"label": "runX", "seed": 7}
    assert api.get(f"/api/tasks/{task_id}").json()["args"]["label"] == "runX"
