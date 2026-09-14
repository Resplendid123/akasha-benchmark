"""链路测试：编译到归因四条任务串起来跑一遍（用假 Akasha 客户端）。

这一条守的是「各阶段的接口对得上」：编译产出的 page_id 能被评测反查、
查询固化的样本与评测的样本集一致、归因能从评测结果排出最差 N 条。

链路测试不是一个阶段，所以这里测的是 :func:`chain.build` 摊平出的四步,
以及运行器把它们一条接一条推进的过程。
"""

from __future__ import annotations

import threading
import time

from test_akasha import FakeClient

from akasha_benchmark.stages import (
    STAGES,
    chain,
    compile,
    query,
)
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
from akasha_benchmark.task import TaskContext, execute


def _imported_pages(connection) -> list[str]:
    """本次编译已导入的 page_id。gold 优先，让检索指标算得出非零值。"""
    runs = compile_store.list_compile_runs(connection)
    if not runs:
        return []
    docs = compile_store.compile_docs(connection, int(runs[0]["id"]))
    return [doc["page_id"] for doc in docs if doc["page_id"] and doc["is_gold"]]


def test_chain_is_not_a_stage():
    """链路测试不能出现在阶段表里 —— 否则任务列表又多一类。"""
    assert set(STAGES) == {
        "download",
        "normalize",
        "compile",
        "query",
        "evaluate",
        "attribute",
    }


def test_build_lays_out_four_steps(normalized, monkeypatch):
    monkeypatch.setattr(chain, "DATASETS", ("hotpotqa",))
    steps = chain.build({"dataset": "hotpotqa", "samples": 2}, normalized)

    assert [s["stage"] for s in steps] == ["compile", "query", "evaluate", "attribute"]
    # 链首不需要关联参数，后三步各自等上一步的产物 id。
    assert "link" not in steps[0]
    assert [s["link"] for s in steps[1:]] == ["compile_id", "query_id", "eval_id"]
    # 每一步的阶段名都得是真实阶段，否则运行器起不来。
    assert all(s["stage"] in STAGES for s in steps)


def test_build_seed_defaults_to_today(normalized, monkeypatch):
    monkeypatch.setattr(chain, "DATASETS", ("hotpotqa",))
    steps = chain.build({"dataset": "hotpotqa", "samples": 1}, normalized)
    assert steps[0]["params"]["seed"] == compile.default_seed()


def test_full_chain(normalized, monkeypatch):
    """四步依次跑完，三层样本集一致。"""
    config_store.update_connection(normalized, base_url="http://x", email="e@x", password="p")
    normalized.commit()

    monkeypatch.setattr(compile, "AkashaClient", lambda config: FakeClient(config))
    # 查询时回本次编译真实的 page_id，这样评测才反查得回语料文档。
    monkeypatch.setattr(
        query,
        "AkashaClient",
        lambda config: FakeClient(config, retrieved=_imported_pages(normalized)),
    )
    monkeypatch.setattr(chain, "DATASETS", ("hotpotqa",))

    steps = chain.build({"dataset": "hotpotqa", "samples": 2, "use_model": False}, normalized)

    # 按运行器的方式推进：上一步的产物 id 填进下一步的关联参数。
    target: int | None = None
    for index, step in enumerate(steps, start=1):
        params = {k: v for k, v in step["params"].items() if v is not None}
        if link := step.get("link"):
            params[link] = target
        task_id = task_store.create_task(normalized, stage=step["stage"], params=params)
        normalized.commit()
        ctx = TaskContext(
            task_id=task_id,
            stage=step["stage"],
            params=params,
            connection=normalized,
            pause_event=threading.Event(),
        )
        execute(STAGES[step["stage"]].run, ctx)

        task = task_store.get_task(normalized, task_id) or {}
        target = task.get("target_id")
        assert target is not None, f"第 {index} 步 {step['stage']} 没有绑定产物"
        # 契约校验就在这一步之后跑，和运行器里的顺序一致。
        if check := chain.VERIFY.get(step["stage"]):
            check(normalized, int(target))

    compile_run = compile_store.list_compile_runs(normalized)[0]
    compile_id = int(compile_run["id"])
    assert compile_run["status"] == run_store.STATUS_SUCCEEDED

    query_run = query_store.list_query_runs(normalized, compile_id)[0]
    query_id = int(query_run["id"])
    assert query_run["status"] == run_store.STATUS_SUCCEEDED

    eval_run = eval_store.list_eval_runs(normalized, query_id)[0]
    eval_id = int(eval_run["id"])
    assert eval_run["status"] == run_store.STATUS_SUCCEEDED

    # 三层的样本集必须一致，否则指标的分母就不是同一批东西。
    expected = {s["sample_id"] for s in compile_store.compile_samples(normalized, compile_id)}
    assert {r["sample_id"] for r in query_store.responses_of(normalized, query_id)} == expected
    assert {r["sample_id"] for r in eval_store.sample_evals(normalized, eval_id)} == expected

    attribution_run = attribution_store.list_attribution_runs(normalized, eval_id)[0]
    assert attribution_run["status"] == run_store.STATUS_SUCCEEDED
    results = attribution_store.attribution_results(normalized, int(attribution_run["id"]))
    assert results
    # 没配归因模型时只出规则结论 —— 那仍然是一条有效的归因。
    assert all(r["rule_based"] == 1 for r in results)


def test_chain_records_share_a_chain_id(db):
    """同一条链的任务共用链号，链首自己就是链号。"""
    head = task_store.create_task(db, stage="compile", params={})
    task_store.set_task_chain(db, head, chain=[{"stage": "query"}], chain_id=head)
    db.commit()

    chain_id, remaining = task_store.task_chain(db, head)
    assert chain_id == head
    assert remaining == [{"stage": "query"}]

    # 链尾：剩余为空，运行器据此停止推进。
    tail = task_store.create_task(db, stage="attribute", params={})
    task_store.set_task_chain(db, tail, chain=[], chain_id=head)
    db.commit()
    assert task_store.task_chain(db, tail) == (head, [])


def test_chain_bookkeeping_stays_out_of_responses(db):
    """chain_json 是运行器的内部账本，不能出现在接口响应里。"""
    task_id = task_store.create_task(db, stage="compile", params={"a": 1})
    task_store.set_task_chain(db, task_id, chain=[{"stage": "query"}], chain_id=task_id)
    db.commit()

    task = task_store.get_task(db, task_id) or {}
    assert "chain_json" not in task
    assert "params_json" not in task
    assert task["chain_id"] == task_id


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


def test_build_rejects_out_of_range_samples(normalized, monkeypatch):
    monkeypatch.setattr(chain, "DATASETS", ("hotpotqa",))
    try:
        chain.build({"dataset": "hotpotqa", "samples": 99}, normalized)
    except ValueError as exc:
        assert "样本数" in str(exc)
    else:
        raise AssertionError("样本数超出上限应当拒绝")


def test_runner_advances_the_chain(db_path, normalized, monkeypatch):
    """运行器自动推进完整链路，最终完成四条任务。"""
    from akasha_platform.settings import Settings
    from akasha_platform.tasks import TaskRunner

    connection = connect(db_path)
    try:
        config_store.update_connection(connection, base_url="http://x", email="e@x", password="p")
        connection.commit()
    finally:
        connection.close()

    monkeypatch.setattr(compile, "AkashaClient", lambda config: FakeClient(config))

    def _fake_query_client(config):
        probe = connect(db_path)
        try:
            return FakeClient(config, retrieved=_imported_pages(probe))
        finally:
            probe.close()

    monkeypatch.setattr(query, "AkashaClient", _fake_query_client)
    monkeypatch.setattr(chain, "DATASETS", ("hotpotqa",))

    runner = TaskRunner(Settings(db_path=db_path))
    head = runner.start_chain({"dataset": "hotpotqa", "samples": 2, "use_model": False})

    # 四条任务依次跑完，最长的一步是编译。
    deadline = time.time() + 120
    while time.time() < deadline:
        probe = connect(db_path)
        try:
            tasks = task_store.list_tasks(probe, limit=50)
            done = [t for t in tasks if t["status"] == task_store.SUCCEEDED]
            active = [t for t in tasks if t["status"] in task_store.ACTIVE]
            if len(done) == 4 or (not active and len(tasks) >= 1):
                break
        finally:
            probe.close()
        time.sleep(0.2)

    probe = connect(db_path)
    try:
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
        # 四条共用链首的链号，且链尾不再往下接。
        assert {t["chain_id"] for t in tasks} == {int(head["id"])}
        assert task_store.task_chain(probe, max(int(t["id"]) for t in tasks))[1] == []
    finally:
        probe.close()
