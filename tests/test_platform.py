"""平台后端：绑定安全、answerMode 默认切分、任务白名单、原文/编译 diff。

这几条各自对着一个具体后果：绑 0.0.0.0 不设认证等于把 Akasha 管理员凭据交出去;
样本列表不按 answerMode 切分会把 1 条检索问题读成 4 条；任务 argv 不走白名单
等于装了个远程执行入口。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from akasha_benchmark.store import repo
from akasha_benchmark.store.migrate import migrate
from akasha_platform import diff
from akasha_platform.main import create_app
from akasha_platform.settings import Settings
from akasha_platform.tasks import STAGE_ARGS, STAGE_MODULES, TaskRejected, _argv, _clean_args

DATASET = "hotpotqa"


# --- 绑定与认证（红线）------------------------------------------------------


def test_refuses_to_bind_a_public_address_without_a_token():
    """§12.10 的红线：这个服务持有 Akasha 管理员凭据、只读数据库连接、
    以及启动长任务的能力。暴露到 0.0.0.0 而不设认证等于把三样一起交出去。"""
    with pytest.raises(RuntimeError, match="refusing to bind"):
        Settings(host="0.0.0.0", auth_token="").validate_binding()


def test_public_binding_is_allowed_once_a_token_is_set():
    Settings(host="0.0.0.0", auth_token="t" * 32).validate_binding()


def test_loopback_needs_no_token():
    Settings(host="127.0.0.1", auth_token="").validate_binding()


def test_settings_never_expose_the_token():
    view = Settings(auth_token="secret-token").redacted()
    assert view["auth_required"] is True
    assert "secret-token" not in str(view)


def test_token_is_enforced_on_every_request(tmp_path: Path):
    db = tmp_path / "t.db"
    migrate(db, verbose=False)
    app = create_app(Settings(db_path=db, auth_token="right-token", web_dist=tmp_path / "none"))
    client = TestClient(app)

    assert client.get("/api/health").status_code == 401
    assert client.get("/api/health", headers={"X-Auth-Token": "wrong"}).status_code == 401
    assert client.get("/api/health", headers={"X-Auth-Token": "right-token"}).status_code == 200


# --- 任务白名单 -------------------------------------------------------------


def test_argv_is_built_from_a_whitelist(tmp_path: Path):
    """argv 只从白名单取模块名，参数走库不走命令行。

    argv 的长度因此固定，不随参数个数增长 —— 原先 14 个参数逐项映射成命令行
    标志，每加一个旋钮要同时改映射表和阶段的 argparse，两处漂了就会出现
    「界面上改了但跑的还是默认值」。
    """
    settings = Settings(db_path=tmp_path / "t.db")
    argv = _argv("ingest", 7, settings)
    assert argv[1:3] == ["-m", "akasha_benchmark.ingest"]
    assert argv[-2:] == ["--run-config", "7"]
    # 实验参数不在 argv 里。
    assert "--label" not in argv


def test_unknown_stage_is_rejected(tmp_path: Path):
    with pytest.raises(TaskRejected, match="unknown stage"):
        _argv("rm -rf /", 1, Settings(db_path=tmp_path / "t.db"))
    with pytest.raises(TaskRejected, match="unknown stage"):
        _clean_args("rm -rf /", {})


def test_unmapped_arguments_never_reach_the_stage(tmp_path: Path):
    """请求体里的未知键不会进 run_config，阶段代码也就读不到它们。

    ``run_config.args_json`` 是通过 HTTP 写进来的，而阶段进程把它当参数读。
    不过滤等于让请求体决定阶段代码看到什么 —— 那是个远程执行面。
    """
    cleaned = _clean_args("ingest", {"label": "x", "evil": "--dangerous", "extra_flag": True})
    assert cleaned == {"label": "x"}


def test_stage_args_are_type_checked():
    """类型不对直接拒掉，不让它进库再到阶段里炸。"""
    assert _clean_args("subset", {"seed": "42"}) == {"seed": 42}
    with pytest.raises(TaskRejected, match="expected int"):
        _clean_args("subset", {"seed": "not-a-number"})


def test_run_config_round_trips_into_stage_arguments(tmp_path: Path):
    """库里的参数能盖到 argparse 的结果上，而未声明的属性设不进去。"""
    import argparse

    from akasha_benchmark import run_args
    from akasha_benchmark.store import connect

    db = tmp_path / "t.db"
    migrate(db, verbose=False)
    connection = connect(db)
    run_config_id = repo.create_run_config(
        connection, "subset", {"label": "runX", "datasets": ["hotpotqa"], "seed": 7, "evil": "x"}
    )
    connection.commit()
    connection.close()

    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default=None)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--db", type=Path, default=None)
    run_args.add_argument(parser)
    args = run_args.apply(
        parser.parse_args(["--db", str(db), "--run-config", str(run_config_id)]), stage="subset"
    )

    assert args.label == "runX"
    # UI 那边是 datasets（复数），阶段的 argparse 是 dataset —— 名字对不上
    # 就会静默跑全量，所以这一项要专门映射。
    assert args.dataset == ["hotpotqa"]
    assert args.seed == 7
    # 未声明的属性设不进去。
    assert not hasattr(args, "evil")


def test_every_whitelisted_stage_maps_to_a_real_module():
    import importlib

    for stage, module in STAGE_MODULES.items():
        assert importlib.util.find_spec(module) is not None, stage


def test_every_stage_with_args_has_a_module_and_vice_versa():
    """两张白名单必须对齐。

    只在一张里出现的阶段是个静默的坑：有模块没参数表 -> 起任务时报「未知阶段」;
    有参数表没模块 -> 拼 argv 时才报。
    """
    assert set(STAGE_ARGS) == set(STAGE_MODULES)


def test_stage_arg_names_exist_on_the_stage_parsers():
    """参数表里的键必须是阶段 argparse 真的认的属性。

    这一条防的是最隐蔽的那种漂移：UI 传 ``qa_limit``、阶段声明的是
    ``qa_limit``，改名之后 run_args.apply 会静默跳过它 —— 界面上改了，
    跑的还是默认值，而没有任何地方报错。
    """
    import importlib

    # datasets 走 dataset 的特例映射，provider_label 也是显式 dest。
    aliases = {"datasets": "dataset", "k": "k", "metrics": "metrics"}
    for stage, spec in STAGE_ARGS.items():
        module = importlib.import_module(STAGE_MODULES[stage])
        parser_args = _declared_dests(module)
        for key in spec:
            target = aliases.get(key, key)
            assert target in parser_args, f"{stage}: {key!r} is not declared by {module.__name__}"


def _declared_dests(module) -> set[str]:
    """跑一遍阶段的 main parser，收集它声明了哪些 dest。

    直接构造 parser 而不是解析 ``--help`` 文本 —— 后者会随格式变化而碎。
    """
    import argparse
    from unittest.mock import patch

    captured: dict[str, argparse.ArgumentParser] = {}
    original_init = argparse.ArgumentParser.__init__

    def spy(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured.setdefault("parser", self)

    with patch.object(argparse.ArgumentParser, "__init__", spy):
        with patch.object(argparse.ArgumentParser, "parse_args", side_effect=SystemExit):
            try:
                module.main([])
            except SystemExit:
                pass
    parser = captured.get("parser")
    return {action.dest for action in parser._actions} if parser else set()


# --- 原文 vs 编译 diff ------------------------------------------------------


def test_diff_reports_words_the_compiler_dropped():
    """§12.9 的根因形态：编译把查询需要的短语删了。

    原文有 Grammy / Emmy，编译产物没有，而问题问的正是这两个词 ——
    于是词法召回在这条样本上必然断，且这不是调参能救的。
    """
    source = (
        "Guests in the album include the Grammy and Emmy award winning Cyndi Lauper, "
        "along with other artists."
    )
    compiled = (
        "The album features vocal contributions from guest artists including Cyndi Lauper "
        "and several others."
    )
    result = diff.diff_vocabulary(source, compiled)
    assert "grammy" in result["dropped"]
    assert "emmy" in result["dropped"]
    # 保留的部分也要看得见，否则读者无法判断这是「丢了修饰语」还是「整段换了」。
    assert result["kept"] > 0

    # 问题里的实词有三个落在 dropped 里 —— "award" 同样被改写掉了
    # （"award winning" -> "guest artists"）。三个都要报出来。
    lost = diff.question_terms_lost("who won Grammy and Emmy award", result)
    assert lost == ["award", "emmy", "grammy"]


def test_diff_expansion_ratio_shows_compilation_expands():
    """编译**不是压缩而是扩写**（实测中位 2.19 倍）。

    这个比值让读者自己看到「丢词是改写策略，不是空间不足」。
    """
    result = diff.diff_vocabulary("short source", "a much longer compiled rendition " * 5)
    assert result["expansion_ratio"] > 1


def test_question_terms_lost_is_empty_when_nothing_relevant_dropped():
    result = diff.diff_vocabulary("Cyndi Lauper won a Grammy", "Cyndi Lauper won a Grammy award")
    assert diff.question_terms_lost("who won Grammy", result) == []


def test_stopwords_do_not_count_as_dropped():
    """虚词的增删不说明任何问题，不该混进 dropped 里。"""
    result = diff.diff_vocabulary("the cat sat on the mat", "cat sat mat")
    assert result["dropped"] == []


# --- answerMode 默认切分（§12.10 的红线）------------------------------------


@pytest.fixture
def seeded(tmp_path: Path):
    """一个装好 4 条样本的评测层：3 条 general（检索得分按定义为 0）+ 1 条真漏 gold。

    这正是 run001 上那四条 ``recall@5 < 1.0`` 的形态。
    """
    db = tmp_path / "t.db"
    migrate(db, verbose=False)
    from akasha_benchmark.store import connect

    connection = connect(db)
    repo.upsert_dataset(
        connection,
        name=DATASET,
        adapter="A",
        adapter_version="1",
        provides=["gold_docs", "reference_answers"],
        identity_rules={},
        qa_path="q",
        qa_sha256="0" * 64,
        qa_rows=4,
        corpus_path="c",
        corpus_sha256="1" * 64,
        corpus_rows=2,
        dedup_stats={},
        gold_count_distribution={},
        unique_question_texts=4,
    )
    layer_id = repo.create_index_layer(
        connection,
        label="L",
        subset_hash="sh",
        seed=1,
        qa_limit=4,
        negatives_ratio=1.0,
        narrativeqa_docs=2,
    )
    query_layer_id = repo.create_query_layer(
        connection,
        index_layer_id=layer_id,
        label="Q",
        config_hash="qh",
        score_threshold=None,
        concurrency=1,
        request_interval_seconds=0.5,
        model_configs=None,
        model_configs_match_index=True,
        allow_config_drift=False,
    )
    eval_id = repo.create_eval_layer(
        connection,
        query_layer_id=query_layer_id,
        label="E",
        config_hash="eh",
        ks=[5],
        metrics=["recall"],
    )
    rows = [
        ("g1", "general", 0.0),
        ("g2", "general", 0.0),
        ("g3", "general", 0.0),
        ("k1", "knowledge", 0.5),
        ("k2", "knowledge", 1.0),
    ]
    for sample_id, mode, recall in rows:
        repo.record_sample_eval(
            connection,
            eval_id,
            sample_id=sample_id,
            dataset=DATASET,
            answer_mode=mode,
            ok=True,
            http_status=200,
            gold_count=2,
            retrieved_count=0 if mode == "general" else 5,
            citation_count=0,
            snippet_count=0,
            latency_ms=100,
            answer="a",
            detail={"metadata": {"type": "bridge"}},
        )
        repo.record_sample_metrics(
            connection, eval_id, sample_id, DATASET, {"recall@5": recall}
        )
    connection.commit()
    app = create_app(Settings(db_path=db, web_dist=tmp_path / "none"))
    yield TestClient(app), eval_id, connection
    connection.close()


def test_sample_list_groups_by_answer_mode_without_being_asked(seeded):
    """按 answerMode 分组是**默认行为**，不是可选筛选器。

    ``no_match`` / ``general`` 无条件返回空 retrievedSources，检索得分按定义为 0。
    混在一起读会把生成端拒答误当成检索失败。
    """
    client, eval_id, _ = seeded
    body = client.get(f"/api/layers/eval/{eval_id}/samples").json()
    assert set(body["by_answer_mode"]) == {"general", "knowledge"}
    assert body["counts"] == {"general": 3, "knowledge": 2}
    assert "检索失败" in body["note"]


def test_worst_samples_expose_the_answer_mode_split(seeded):
    """失败案例入口必须先给出按 answerMode 的计数。

    否则「最差的 4 条」里 3 条是生成端回落，读者会把 1 条检索问题读成 4 条。
    """
    client, eval_id, _ = seeded
    body = client.get(f"/api/layers/eval/{eval_id}/worst?metric=recall@5&limit=4").json()
    assert body["count_by_answer_mode"]["general"] == 3
    assert body["count_by_answer_mode"]["knowledge"] == 1
    # 只有一条是真的漏 gold。
    knowledge = [s for s in body["samples"] if s["answer_mode"] == "knowledge"]
    assert len(knowledge) == 1
    assert knowledge[0]["value"] == pytest.approx(0.5)


def test_worst_respects_metric_direction(seeded):
    """``truncation_loss`` 是越低越好，排序方向必须跟着指标声明走。"""
    client, eval_id, connection = seeded
    repo.record_sample_metrics(connection, eval_id, "k1", DATASET, {"truncation_loss": 9.0})
    repo.record_sample_metrics(connection, eval_id, "k2", DATASET, {"truncation_loss": 1.0})
    connection.commit()

    body = client.get(f"/api/layers/eval/{eval_id}/worst?metric=truncation_loss&limit=2").json()
    assert body["higher_is_better"] is False
    # 越低越好 -> 最差的是最大的那个。
    assert body["samples"][0]["value"] == pytest.approx(9.0)


def test_metric_definitions_expose_requires_and_direction():
    """前端靠 requires 决定某列该不该显示，靠 higher_is_better 决定排序方向。"""
    app = create_app(Settings(db_path=Path("nonexistent.db"), web_dist=Path("nope")))
    client = TestClient(app)
    definitions = {d["name"]: d for d in client.get("/api/metrics/definitions").json()}

    assert definitions["recall"]["requires"] == ["gold_docs"]
    # faithfulness 不需要任何标注，所以四组都成立 —— 这是 §12.4 的具体收获。
    assert definitions["faithfulness"]["requires"] == []
    assert definitions["truncation_loss"]["higher_is_better"] is False
    assert definitions["recall"]["per_k"] is True


def test_lineage_returns_503_without_a_readonly_database(tmp_path: Path):
    """没配 database_url 时给一句明确的 503，其余视图照常工作。

    不 patch 任何东西：空库里 app_config 就是空的，所以 database_url 为空 ——
    这正是「刚建好库还没配」的真实状态。
    """
    db = tmp_path / "t.db"
    migrate(db, verbose=False)
    client = TestClient(create_app(Settings(db_path=db, web_dist=tmp_path / "none")))

    response = client.get("/api/lineage/some-page-id")
    assert response.status_code == 503
    assert "database_url" in response.json()["detail"]
