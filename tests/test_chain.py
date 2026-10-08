from __future__ import annotations

import time
from contextlib import closing

from test_akasha import FakeClient

from akasha_benchmark.stages import (
    STAGES,
    chain,
    compile,
    query,
)
from akasha_benchmark.datasets import get_adapter
from akasha_benchmark.metrics import registry
from akasha_benchmark.store import (
    compile_store,
    config_store,
    connect,
    task_store,
)


def _imported_pages(connection) -> list[str]:
    runs = compile_store.list_compile_runs(connection)
    if not runs:
        return []
    docs = compile_store.compile_docs(connection, int(runs[0]["id"]))
    return [doc["page_id"] for doc in docs if doc["page_id"] and doc["is_gold"]]


def test_build_lays_out_four_steps(normalized, monkeypatch):
    monkeypatch.setattr(chain, "DATASETS", ("hotpotqa",))
    sample_id = "hotpotqa:q1"
    steps = chain.build({"dataset": "hotpotqa", "sample_id": sample_id}, normalized)

    assert [s["stage"] for s in steps] == ["compile", "query", "evaluate", "attribute"]
    assert "link" not in steps[0]
    assert [s["link"] for s in steps[1:]] == ["compile_id", "query_id", "eval_id"]
    assert all(s["stage"] in STAGES for s in steps)
    assert steps[0]["params"]["sample_ids"] == [sample_id]
    assert steps[0]["params"]["negatives_ratio"] == 1.5
    assert steps[2]["params"]["ks"] == [2]
    expected = {
        definition.name
        for definition in registry.available(get_adapter("hotpotqa").provides)
        if definition.kind == registry.KIND_DETERMINISTIC
    }
    assert set(steps[2]["params"]["metrics"]) == expected


def test_build_adds_all_judge_metrics_when_requested(normalized, monkeypatch):
    monkeypatch.setattr(chain, "DATASETS", ("hotpotqa",))

    steps = chain.build(
        {"dataset": "hotpotqa", "sample_id": "hotpotqa:q1", "with_judge": True},
        normalized,
    )

    expected = {
        definition.name
        for definition in registry.available(get_adapter("hotpotqa").provides)
    }
    assert set(steps[2]["params"]["metrics"]) == expected


def test_build_rejects_unsupported_dataset(normalized):
    try:
        chain.build({"dataset": "narrativeqa"}, normalized)
    except ValueError as exc:
        assert "仅支持" in str(exc)
    else:
        raise AssertionError("narrativeqa 没有 gold 标注，链路测试应当拒绝")


def test_build_requires_normalized_dataset(db):
    try:
        chain.build({"dataset": "hotpotqa"}, db)
    except ValueError as exc:
        assert "归一化" in str(exc)
    else:
        raise AssertionError("未归一化时应当拒绝")


def test_build_requires_sample_from_selected_dataset(normalized, monkeypatch):
    monkeypatch.setattr(chain, "DATASETS", ("hotpotqa",))
    try:
        chain.build({"dataset": "hotpotqa", "sample_id": "missing"}, normalized)
    except ValueError as exc:
        assert "样本" in str(exc)
    else:
        raise AssertionError("不存在的样本应当拒绝")


def test_runner_advances_the_chain(db_path, normalized, monkeypatch):
    from akasha_platform import tasks as platform_tasks
    from akasha_platform.settings import Settings
    from akasha_platform.tasks import TaskRunner

    with closing(connect(db_path)) as connection:
        config_store.update_connection(connection, base_url="http://x", email="e@x", password="p")
        connection.commit()

    monkeypatch.setattr(compile, "AkashaClient", lambda config: FakeClient(config))
    monkeypatch.setattr(platform_tasks, "AkashaClient", lambda config: FakeClient(config))

    def _fake_query_client(config):
        with closing(connect(db_path)) as probe:
            return FakeClient(config, retrieved=_imported_pages(probe))

    monkeypatch.setattr(query, "AkashaClient", _fake_query_client)
    monkeypatch.setattr(chain, "DATASETS", ("hotpotqa",))

    runner = TaskRunner(Settings(db_path=db_path))
    head = runner.start_chain(
        {"dataset": "hotpotqa", "sample_id": "hotpotqa:q1", "use_model": False}
    )

    _settle(db_path)

    with closing(connect(db_path)) as probe:
        tasks = task_store.list_tasks(probe, limit=50)
        failed = [(t["stage"], t["error"]) for t in tasks if t["status"] == task_store.FAILED]
        assert not failed, f"链上有任务失败：{failed}"
        assert [t["stage"] for t in tasks][::-1] == [
            "compile",
            "query",
            "evaluate",
            "attribute",
        ]
        assert all(t["status"] == task_store.SUCCEEDED for t in tasks)
        assert {t["chain_id"] for t in tasks} == {int(head["id"])}
        assert task_store.task_chain(probe, max(int(t["id"]) for t in tasks))[1] == []


def test_query_follow_up_runs_evaluate_and_attribute(db_path, normalized, monkeypatch):
    """勾了自动跟进的查询，完成后自己把评测和归因串起来。"""
    from akasha_platform.settings import Settings
    from akasha_platform.tasks import TaskRunner

    with closing(connect(db_path)) as connection:
        config_store.update_connection(connection, base_url="http://x", email="e@x", password="p")
        connection.commit()

    monkeypatch.setattr(compile, "AkashaClient", lambda config: FakeClient(config))

    def _fake_query_client(config):
        with closing(connect(db_path)) as probe:
            return FakeClient(config, retrieved=_imported_pages(probe))

    monkeypatch.setattr(query, "AkashaClient", _fake_query_client)

    runner = TaskRunner(Settings(db_path=db_path))
    runner.start("compile", {"datasets": ["hotpotqa"], "qa_limit": 2})
    _settle(db_path)

    with closing(connect(db_path)) as probe:
        compile_id = int(compile_store.list_compile_runs(probe)[0]["id"])
    head = runner.start("query", {"compile_id": compile_id, "follow_up": True})
    _settle(db_path)

    with closing(connect(db_path)) as probe:
        tasks = task_store.list_tasks(probe, limit=50)
        failed = [(t["stage"], t["error"]) for t in tasks if t["status"] == task_store.FAILED]
        assert not failed, f"链上有任务失败：{failed}"
        assert [t["stage"] for t in tasks][::-1] == [
            "compile",
            "query",
            "evaluate",
            "attribute",
        ]
        followed = [t for t in tasks if t["stage"] in ("query", "evaluate", "attribute")]
        assert {t["chain_id"] for t in followed} == {int(head["id"])}


def _settle(db_path, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with closing(connect(db_path)) as probe:
            if not task_store.active_tasks(probe):
                return
        time.sleep(0.2)
    raise AssertionError("任务没跑完")
