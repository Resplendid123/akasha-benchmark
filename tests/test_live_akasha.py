"""单样本在线冒烟：拿真实的 Akasha 把一条样本从入库跑到查询，逐条核对
[akasha_api.md](../docs/akasha_api.md) 里写下的每个字段约定。

和 test_client_and_stages.py 的分工：那边跑在 ``httpx.MockTransport`` 上，锁的是
**我们**的行为（multipart 字段名、闸门、续跑）；替身是照我们的理解写的，所以理解
错了它也照样绿。这里锁的是**服务端**的行为 —— 文档里那些「响应里没有
retrievalDiagnostics」「citationEvidence 与 citations 等长」「budget 上限恒为
12000」如果哪天不成立了，评测算出来的数就是错的，而且不会有任何报错。

默认整个模块 skip。要跑：

    AKASHA_LIVE=1 uv run pytest tests/test_live_akasha.py -v

需要一个在线的 Akasha、一份能登录的 OWNER 凭据（同 akasha.config.json），
以及 ``data/subsets/<run-id>/<dataset>/`` 已有产物。缺任何一项都 skip 而不是 fail。

这一趟会烧真实的 LLM 调用（编译 + 一次 query），所以：

* 只导 1 条样本的 gold 文档加少量干扰文档，不是整个子集
* 全过程写进 ``data/smoke/<ts>-roundtrip.json``，事后翻字段不用重跑
* ``AKASHA_LIVE_REPLAY=<那个文件>`` 可以纯离线重放断言
* 建的是独立的 ``smoke`` 前缀 Space，**不碰** ``bench`` 前缀那几个 ——
  往真实入库的 Space 里塞页会污染它的 page_map 与质量计数
* 跑完删掉该 Space，``AKASHA_LIVE_KEEP=1`` 可保留以便手工看

环境变量：

===========================  ==================================================
``AKASHA_LIVE``              置 1 才跑；其余全部可选
``AKASHA_LIVE_REPLAY``       离线重放已有的 roundtrip.json，不连服务器
``AKASHA_LIVE_RUN_ID``       取哪个 run 的子集，默认 run001
``AKASHA_LIVE_DATASET``      取哪个数据集，默认 hotpotqa
``AKASHA_LIVE_SAMPLE_ID``    指定 sample_id，默认第一条 gold 文件齐全的
``AKASHA_LIVE_DISTRACTORS``  额外导入的非 gold 文档数，默认 3
``AKASHA_LIVE_TIMEOUT``      等编译的秒数，默认 900
``AKASHA_LIVE_KEEP``         置 1 则保留 Space 不删
===========================  ==================================================
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from akasha_benchmark.akasha_client import (
    ACTIVE_RUN_STATUSES,
    TERMINAL_RUN_STATUSES,
    AkashaClient,
    AkashaError,
)
from akasha_benchmark.config import AkashaConfig, load_config
from akasha_benchmark.datasets import CanonicalSample, subset_dir
from akasha_benchmark.ingest import MODEL_FEATURES
from akasha_benchmark.io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    read_jsonl,
    sha256_text,
    utc_now,
)
from akasha_benchmark.metrics import qa, retrieval

# --- 服务端已核对过的常量（行号指向 ../Akasha） -------------------------------

# ai-knowledge-chat.service.ts，三个取值穷举。
ANSWER_MODES = frozenset({"knowledge", "no_match", "general"})

# knowledge-context-pack.service.ts:137 与 snippets[].retrievalReasons 的并集。
RETRIEVAL_REASONS = frozenset(
    {"semantic", "lexical", "exact-title", "graph-neighbor", "sidecar-prefiltered"}
)

# 没有调用点给 buildContextPack 传 budget，所以这两个恒为默认值
# （knowledge-context-pack.service.ts:242-262）。
BUDGET_MAX_CONTEXT_LENGTH = 12000
BUDGET_RESPONSE_RESERVE = 0

# llm-wiki.controller.ts:174 解构时排除，只写进 knowledge_query_audit.metadata。
STRIPPED_FROM_RESPONSE = ("retrievalDiagnostics", "retrievalScope")

# page.repo.ts:28-47 的 baseFields 不含正文三兄弟。
IMPORT_RESPONSE_ABSENT = ("content", "ydoc", "textContent")

# ai-knowledge-chat.service.ts:518 的 stripCitationMarkers 会全部删掉。
CITATION_MARKER = re.compile(r"\[\[cite:")

# 无条件清空的四个数组（ai-knowledge-chat.service.ts:641,667）。
EMPTIED_WHEN_NOT_KNOWLEDGE = ("retrievedSources", "citations", "citationEvidence", "snippets")

SMOKE_SLUG_PREFIX = "smoke"


# --- 输入准备 -----------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _eligible_samples(dataset: str, run_id: str) -> tuple[list[CanonicalSample], Path]:
    """子集里 gold 文档全部落在磁盘上的样本，以及 corpus 目录。

    gold 不全的排除掉：那种样本「召回不到」时分不清是检索问题还是根本没导进去。
    子集不存在就 skip —— 在线冒烟不负责生产离线产物，那是 ``make subset`` 的事。
    """
    src = subset_dir(run_id, dataset, Path("data"))
    samples_path = src / "samples.jsonl"
    if not samples_path.is_file():
        pytest.skip(f"no subset at {samples_path}; run: make subset RUN_ID={run_id}")

    corpus = src / "corpus"
    eligible = [
        s
        for s in (CanonicalSample.model_validate(row) for row in read_jsonl(samples_path))
        if s.gold_doc_ids and all((corpus / f"{d}.md").is_file() for d in s.gold_doc_ids)
    ]
    if not eligible:
        pytest.skip(f"no sample in {samples_path} has all of its gold documents on disk")
    return eligible, corpus


def _pick_sample(dataset: str, run_id: str) -> tuple[CanonicalSample, Path]:
    """单样本往返用的那一条。``AKASHA_LIVE_SAMPLE_ID`` 可以指定。"""
    eligible, corpus = _eligible_samples(dataset, run_id)
    wanted = os.environ.get("AKASHA_LIVE_SAMPLE_ID")
    if not wanted:
        return eligible[0], corpus
    for sample in eligible:
        if sample.sample_id == wanted:
            return sample, corpus
    pytest.skip(f"sample {wanted!r} is not in the subset, or its gold md is missing")


def _pick_samples(dataset: str, run_id: str, count: int) -> tuple[list[CanonicalSample], Path]:
    """批量阶段用的前 N 条。取前 N 条而不是随机抽，让整趟可复现。"""
    eligible, corpus = _eligible_samples(dataset, run_id)
    return eligible[:count], corpus


def _distractor_ids(corpus: Path, gold: tuple[str, ...], count: int) -> list[str]:
    """挑若干非 gold 文档一起导入。

    只导 gold 的话 Recall@k 恒为 1，这条断言就没有区分力了 —— 库里除了答案
    没有别的东西，随便召回什么都是对的。加几篇干扰项让排序至少要做点事。
    """
    others = sorted(p.stem for p in corpus.glob("*.md") if p.stem not in set(gold))
    return others[:count]


# --- 批量三段的输入：切一份迷你子集 -------------------------------------------


def _carve_subset(
    dest_data: Path,
    dataset: str,
    run_id: str,
    samples: list[CanonicalSample],
    corpus: Path,
    distractors: int,
) -> list[str]:
    """把选中的样本连同其 gold 文档切成一份独立子集，返回 doc_id 列表。

    批量三段走的是真实的 ``ingest.run()`` / ``run_queries.run()`` /
    ``evaluate.run()``，它们都从 ``<data>/subsets/<run>/<dataset>/`` 读输入，
    所以给它们一份真实产物的**子集**，而不是另造一套假数据 —— 这样
    corpus 的 md 字节、``manifest.json`` 的 sha256、``samples.jsonl`` 的字段
    全都和正式跑的时候一致。

    ``manifest.json`` 只列这几篇。``import_corpus`` 是按 manifest 的
    ``corpus_md_sha256`` 决定导什么，多列一篇就会去导一个不存在的文件。
    """
    gold = {d for s in samples for d in s.gold_doc_ids}
    extra = [p.stem for p in sorted(corpus.glob("*.md")) if p.stem not in gold][:distractors]
    doc_ids = sorted(gold) + extra

    dest = dest_data / "subsets" / run_id / dataset
    (dest / "corpus").mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for doc_id in doc_ids:
        markdown = (corpus / f"{doc_id}.md").read_text(encoding="utf-8")
        # newline="\n" 是必须的：Windows 上默认会写成 CRLF，sha256 立刻对不上，
        # 而 ingest 校验哈希不符时会直接抛错。
        (dest / "corpus" / f"{doc_id}.md").write_text(markdown, encoding="utf-8", newline="\n")
        hashes[doc_id] = sha256_text(markdown)

    atomic_write_jsonl(dest / "samples.jsonl", [s.model_dump(mode="json") for s in samples])
    atomic_write_json(dest / "manifest.json", {"corpus_md_sha256": hashes})
    return doc_ids


# --- 一次真实往返 -------------------------------------------------------------


def _wait_for_compile(client: AkashaClient, space_id: str, timeout: int) -> dict[str, Any]:
    """轮询到编译进入终态。超时不 fail，把观察到的状态带回去让断言去判。"""
    deadline = time.monotonic() + timeout
    interval = 5.0
    while True:
        summary = client.run_diagnostics_summary([space_id])
        counts: dict[str, int] = summary.get("statusCounts") or {}
        active = sum(c for name, c in counts.items() if name in ACTIVE_RUN_STATUSES)
        if active == 0:
            return {"status_counts": counts, "timed_out": False, "waited": True}
        if time.monotonic() > deadline:
            return {"status_counts": counts, "timed_out": True, "waited": True}
        time.sleep(interval)


def _import_one(client: AkashaClient, md_path: Path, space_id: str) -> dict[str, Any]:
    """导一篇，记下我们发出去的和服务端返回的，供后面断言身份对应关系。"""
    markdown = md_path.read_text(encoding="utf-8")
    page = client.import_page(md_path, space_id)
    return {
        "doc_id": md_path.stem,
        "filename": md_path.name,
        # 首个 heading 是我们期望服务端拿去当 title 的那一行。
        "first_heading": markdown.splitlines()[0].removeprefix("# ").strip(),
        "md_sha256": sha256_text(markdown),
        "page": page,
    }


def _round_trip() -> dict[str, Any]:
    """真的连一次 Akasha，把一条样本走完入库到查询，返回全部观察结果。

    任何一步抛异常都直接冒出去 —— 冒烟测试的意义就是让这些失败可见，
    在这里 catch 成 skip 等于把「服务端坏了」伪装成「没跑」。
    """
    run_id = os.environ.get("AKASHA_LIVE_RUN_ID", "run001")
    dataset = os.environ.get("AKASHA_LIVE_DATASET", "hotpotqa")
    timeout = _env_int("AKASHA_LIVE_TIMEOUT", 900)
    sample, corpus = _pick_sample(dataset, run_id)
    distractors = _distractor_ids(
        corpus, sample.gold_doc_ids, _env_int("AKASHA_LIVE_DISTRACTORS", 3)
    )
    doc_ids = list(sample.gold_doc_ids) + distractors

    config = load_config(None)
    try:
        config.require_credentials()
    except ValueError as exc:
        pytest.skip(f"{exc}")

    observed: dict[str, Any] = {
        "kind": "akasha-live-smoke",
        "generated_at": utc_now(),
        "connection": config.redacted(),
        "run_id": run_id,
        "dataset": dataset,
        "sample": sample.model_dump(mode="json"),
        "gold_doc_ids": list(sample.gold_doc_ids),
        "distractor_doc_ids": distractors,
    }
    try:
        return _execute(config, observed, sample, corpus, doc_ids, dataset, run_id, timeout)
    finally:
        _persist(observed)


def _execute(
    config: AkashaConfig,
    observed: dict[str, Any],
    sample: CanonicalSample,
    corpus: Path,
    doc_ids: list[str],
    dataset: str,
    run_id: str,
    timeout: int,
) -> dict[str, Any]:
    space_id: str | None = None
    with AkashaClient(config) as client:
        # 登录单独走一次裸请求，好把状态码和响应体都记下来 ——
        # 「成功时响应体是空的」本身就是要验的一条。
        try:
            login = client.request(
                "POST",
                "auth/login",
                json_body={"email": config.email, "password": config.password},
            )
        except OSError as exc:
            pytest.skip(f"cannot reach {config.base_url}: {type(exc).__name__}: {exc}")
        observed["login"] = {
            "status": login.status,
            "body": login.body,
            "cookie_names": sorted(client._client.cookies.keys()),
        }

        me = client.current_user()
        observed["me"] = me
        role = (me.get("user") or {}).get("role")
        if role != "owner":
            pytest.skip(
                f"live user role is {role!r}, not 'owner'; a non-owner loses chunks silently "
                "at the authorization gate, so this run would prove nothing"
            )

        observed["model_configs"] = client.get_model_configs()

        # 独立 Space，slug 带随机后缀：并发跑或上一次没删干净都不会撞。
        slug = f"{SMOKE_SLUG_PREFIX}{uuid.uuid4().hex[:10]}"
        space = client.create_space(
            name=f"smoke {dataset} {run_id}"[:100],
            slug=slug,
            description="Akasha-Benchmark live smoke test. Generated, safe to delete.",
        )
        space_id = space["id"]
        observed["space"] = space
        print(f"\nsmoke space {space_id} (slug {slug})")

        try:
            observed["imports"] = [
                _import_one(client, corpus / f"{doc_id}.md", space_id) for doc_id in doc_ids
            ]
            observed["compile"] = client.compile_spaces([space_id])
            observed["runs"] = _wait_for_compile(client, space_id, timeout)
            observed["quality"] = client.quality_diagnostics([space_id])

            response = client.query(sample.question, [space_id])
            observed["query"] = {
                "request": {
                    "query": sample.question,
                    "spaceIds": [space_id],
                    "type": "user",
                },
                "status": response.status,
                "latency_ms": response.latency_ms,
                "body": response.body,
            }
            print(
                f"query -> HTTP {response.status} in {response.latency_ms}ms, "
                f"answerMode={(response.body or {}).get('answerMode')!r}"
            )
        finally:
            observed["space_deleted"] = _cleanup(client, space_id)

    return observed


def _persist(observed: dict[str, Any]) -> None:
    """把这一趟的观察结果落盘。

    在 ``finally`` 里调用，**失败时也要落**：这一趟烧了真实的 LLM 调用，
    而失败时的响应恰恰是最需要慢慢看的那一份。只在真的建过 Space 之后才写，
    否则「连不上服务器」这种 skip 会留下一堆没内容的文件。
    """
    if "space" not in observed:
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = Path("data") / "smoke" / f"{stamp}-roundtrip.json"
    atomic_write_json(path, observed)
    observed["artifact_path"] = str(path)
    print(f"round trip written to {path}")


def _run_pipeline(
    data_dir: Path, dataset: str, source_run: str, run_id: str, samples: int
) -> dict[str, Any]:
    """跑真实的 ingest → query → report 三段，入库和查询各跑两遍验续跑。

    调的是各阶段的 ``run()``，不是重写一遍它们的逻辑 —— 冒烟要验的就是那几个
    真实入口在真实服务上的行为，自己再实现一份等于什么都没验。

    ``source_run`` 是读的那个 run（正式子集，通常 run001），``run_id`` 是这一趟
    自己的输出 run。两者必须分开：拿输出 run 去读源子集会找不到文件。
    """
    from akasha_benchmark import evaluate as ev_mod
    from akasha_benchmark import ingest as ingest_mod
    from akasha_benchmark import run_queries as rq_mod
    from akasha_benchmark.metrics.retrieval import DEFAULT_KS

    picked, corpus = _pick_samples(dataset, source_run, samples)
    doc_ids = _carve_subset(
        data_dir, dataset, run_id, picked, corpus, _env_int("AKASHA_LIVE_DISTRACTORS", 3)
    )
    out: dict[str, Any] = {
        "run_id": run_id,
        "source_run": source_run,
        "dataset": dataset,
        "sample_ids": [s.sample_id for s in picked],
        "doc_ids": doc_ids,
    }
    print(f"\npipeline: {len(picked)} sample(s), {len(doc_ids)} doc(s), run_id={run_id}")

    # --- 入库 ---
    out["ingest_exit"] = ingest_mod.run(run_id, [dataset], None, data_dir)
    ingest_manifest_path = ingest_mod.ingest_dir(run_id, data_dir) / "manifest.json"
    out["ingest_manifest"] = load_json(ingest_manifest_path)
    out["page_map"] = list(read_jsonl(ingest_mod.ingest_dir(run_id, data_dir) / "page_map.jsonl"))
    out["space_ids"] = [s["id"] for s in out["ingest_manifest"]["spaces"].values()]

    # --- 查询，两遍 ---
    out["query_exit"] = rq_mod.run(run_id, [dataset], None, data_dir, None, None, False)
    responses_path = rq_mod.responses_dir(run_id, data_dir) / f"{dataset}.jsonl"
    out["responses"] = list(read_jsonl(responses_path))
    out["query_manifest"] = load_json(rq_mod.responses_dir(run_id, data_dir) / "manifest.json")

    print("query stage again (resume: nothing should be re-requested)")
    out["query_resume_exit"] = rq_mod.run(run_id, [dataset], None, data_dir, None, None, False)
    out["query_manifest_resume"] = load_json(
        rq_mod.responses_dir(run_id, data_dir) / "manifest.json"
    )
    out["responses_after_resume"] = list(read_jsonl(responses_path))

    # --- 报告 ---
    out["report_exit"] = ev_mod.run(run_id, [dataset], data_dir, DEFAULT_KS)
    reports = ev_mod.reports_dir(run_id, data_dir)
    out["metrics"] = load_json(reports / "metrics.json")
    out["per_sample"] = list(read_jsonl(reports / "per_sample.jsonl"))
    out["report_md"] = (reports / "report.md").read_text(encoding="utf-8")

    # --- 入库续跑放最后 ---
    # 它会覆盖 manifest（--skip-compile 让 quality_passed 变 False），所以必须在
    # 查询和报告都读完之后再跑。带 skip_compile 是为了省一次没有意义的重编译 ——
    # 导入侧的续跑与编译无关。
    print("ingest stage again (resume: nothing should be re-imported)")
    out["ingest_resume_exit"] = ingest_mod.run(
        run_id, [dataset], None, data_dir, skip_compile=True
    )
    out["ingest_manifest_resume"] = load_json(ingest_manifest_path)
    out["page_map_after_resume"] = list(
        read_jsonl(ingest_mod.ingest_dir(run_id, data_dir) / "page_map.jsonl")
    )
    return out


def _cleanup(client: AkashaClient, space_id: str) -> bool:
    """删掉冒烟用的 Space。删不掉不算测试失败，但要显式说出来。"""
    if os.environ.get("AKASHA_LIVE_KEEP") == "1":
        print(f"AKASHA_LIVE_KEEP=1, keeping space {space_id}")
        return False
    try:
        client.delete_space(space_id)
        return True
    except (AkashaError, OSError) as exc:
        print(f"WARNING could not delete smoke space {space_id}: {exc}")
        return False


# --- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module")
def trip() -> dict[str, Any]:
    """整个模块共用一次往返。``AKASHA_LIVE_REPLAY`` 指定文件时改为离线重放。"""
    replay = os.environ.get("AKASHA_LIVE_REPLAY")
    if replay:
        path = Path(replay)
        if not path.is_file():
            pytest.skip(f"AKASHA_LIVE_REPLAY={replay} is not a file")
        return json.loads(path.read_text(encoding="utf-8"))
    if os.environ.get("AKASHA_LIVE") != "1":
        pytest.skip("live test; set AKASHA_LIVE=1 to run it against a real Akasha")
    return _round_trip()


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """跑一次真实的 ingest → query → report，覆盖批量、续跑与报告产出。

    与 :func:`trip` 是两件事，所以各建各的 Space：那边验单条响应的字段约定
    （刻意绕开阶段代码，直接打 HTTP），这边验三个阶段入口本身。两者可以用
    ``-k`` 单独跑。

    产物写进 pytest 的临时目录，不落到仓库的 ``data/`` —— 这一趟的 run_id 是随机的，
    留在 ``data/`` 里只会堆垃圾。
    """
    if os.environ.get("AKASHA_LIVE_REPLAY"):
        pytest.skip("pipeline stages need a live server; there is nothing to replay")
    if os.environ.get("AKASHA_LIVE") != "1":
        pytest.skip("live test; set AKASHA_LIVE=1 to run it against a real Akasha")

    dataset = os.environ.get("AKASHA_LIVE_DATASET", "hotpotqa")
    source_run = os.environ.get("AKASHA_LIVE_RUN_ID", "run001")
    samples = _env_int("AKASHA_LIVE_PIPELINE_SAMPLES", 3)
    timeout = _env_int("AKASHA_LIVE_TIMEOUT", 900)

    config = load_config(None)
    try:
        config.require_credentials()
    except ValueError as exc:
        pytest.skip(f"{exc}")

    # run_id 带随机后缀：Space 的 slug 由 (prefix, dataset, run_id) 拼成，
    # 换一个 run_id 就换一个 Space，重复跑或并发跑都不会撞。
    run_id = f"smoke{uuid.uuid4().hex[:8]}"
    data_dir = tmp_path_factory.mktemp("pipeline-data")

    # 用环境变量覆盖，而不是 patch load_config：这样阶段代码走的是它平时那条
    # 配置加载路径，顺带把「环境变量优先」这条也验了。凭据仍从配置文件来，
    # 不必把密码写到临时文件里。
    patch = pytest.MonkeyPatch()
    patch.setenv("AKASHA_SPACE_SLUG_PREFIX", SMOKE_SLUG_PREFIX)
    patch.setenv("AKASHA_POLL_INTERVAL_SECONDS", "5")
    patch.setenv("AKASHA_POLL_TIMEOUT_SECONDS", str(timeout))

    result: dict[str, Any] = {}
    try:
        result = _run_pipeline(data_dir, dataset, source_run, run_id, samples)
        yield result
    finally:
        patch.undo()
        _persist_pipeline(result)
        _delete_spaces(config, result.get("space_ids") or [])


def _persist_pipeline(result: dict[str, Any]) -> None:
    """三段的产物落盘，失败时也落 —— 这一趟同样烧了真实 LLM 调用。"""
    if not result:
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = Path("data") / "smoke" / f"{stamp}-pipeline.json"
    # report.md 单独存成 .md，塞进 JSON 里读起来太难受。
    report_md = result.pop("report_md", None)
    atomic_write_json(path, result)
    if report_md is not None:
        result["report_md"] = report_md
        (path.parent / f"{stamp}-report.md").write_text(report_md, encoding="utf-8", newline="\n")
    print(f"pipeline artifacts written to {path}")


def _delete_spaces(config: AkashaConfig, space_ids: list[str]) -> None:
    """删掉批量阶段建的 Space。删不掉不算测试失败，但要说出来。"""
    if not space_ids or os.environ.get("AKASHA_LIVE_KEEP") == "1":
        if space_ids:
            print(f"AKASHA_LIVE_KEEP=1, keeping spaces {space_ids}")
        return
    with AkashaClient(config) as client:
        client.login()
        for space_id in space_ids:
            try:
                client.delete_space(space_id)
            except (AkashaError, OSError) as exc:
                print(f"WARNING could not delete pipeline space {space_id}: {exc}")


@pytest.fixture(scope="module")
def body(trip: dict[str, Any]) -> dict[str, Any]:
    """查询响应体。非 2xx 时 fail 而不是让后面每条断言各自炸一遍。"""
    query = trip["query"]
    assert query["status"] == 200, (
        f"POST /api/llm-wiki/query returned HTTP {query['status']}: "
        f"{str(query['body'])[:400]}"
    )
    assert isinstance(query["body"], dict), f"expected a JSON object, got {type(query['body'])}"
    return query["body"]


@pytest.fixture(scope="module")
def page_to_doc(trip: dict[str, Any]) -> dict[str, str]:
    """page_id -> doc_id，即 ingest 阶段 page_map 的那张反查表。"""
    return {row["page"]["id"]: row["doc_id"] for row in trip["imports"]}


# --- 认证与身份 ---------------------------------------------------------------


def test_login_sets_auth_token_cookie_with_an_empty_body(trip: dict[str, Any]):
    """登录成功只 set httpOnly 的 authToken cookie，响应体是空的。

    响应体开始带东西（比如 MFA 分支）时这里会先炸，而不是让后续每个请求 401。
    """
    login = trip["login"]
    assert 200 <= login["status"] < 300
    assert "authToken" in login["cookie_names"], (
        f"no authToken cookie; got {login['cookie_names']}. MFA on this account would do this."
    )
    assert login["body"] in (None, "", {}), f"login body is no longer empty: {login['body']!r}"


def test_current_user_reports_role_and_resolved_workspace(trip: dict[str, Any]):
    """``users/me`` 带回 user.role 和中间件解析出的 workspace，两者 ingest 都要用。"""
    me = trip["me"]
    assert (me.get("user") or {}).get("role") == "owner"
    # 自建部署靠 workspaceRepo.findFirst() 解析，不需要伪造 Host 头 —— 拿到 id 即证明。
    assert (me.get("workspace") or {}).get("id"), f"no workspace resolved: {me}"


def test_model_configs_expose_the_four_tracked_features(trip: dict[str, Any]):
    """四项模型配置都能读到，否则 ingest 的快照比对形同虚设。

    形状是 ``{"configs": [{"feature": ..., "model": ..., "apiKeySet": ...}, ...]}``。
    快照比对是整体相等判断，所以这里只要求四项都在、且都配了 key ——
    真正的漂移检测由 run_queries 做。
    """
    configs = trip["model_configs"]
    assert configs, "model-configs returned nothing; the drift guard would compare None to None"
    entries = configs.get("configs") if isinstance(configs, dict) else None
    assert isinstance(entries, list), f"expected {{'configs': [...]}}, got {configs!r}"

    by_feature = {e.get("feature"): e for e in entries}
    missing = [f for f in MODEL_FEATURES if f not in by_feature]
    assert not missing, f"model-configs lacks {missing}; present: {sorted(by_feature)}"
    # 没配 key 的 feature 一到真正调用就失败，且失败得很晚（编译或生成中途）。
    unset = [f for f, e in by_feature.items() if not e.get("apiKeySet")]
    assert not unset, f"these features have no API key set: {unset}"
    print("models: " + ", ".join(f"{f}={by_feature[f].get('model')}" for f in MODEL_FEATURES))


# --- 导入 ---------------------------------------------------------------------


def test_import_returns_a_page_id_for_every_document(trip: dict[str, Any]):
    """每篇都拿到 page_id，且各不相同。缺 id 等于永久丢失映射。"""
    pages = [row["page"] for row in trip["imports"]]
    assert all(p.get("id") for p in pages), f"some imports returned no id: {pages}"
    ids = [p["id"] for p in pages]
    assert len(set(ids)) == len(ids), f"page ids collided across documents: {ids}"


def test_import_takes_the_title_from_the_first_heading_not_the_filename(trip: dict[str, Any]):
    """title 来自首个 Markdown heading，所以文件名可以专职当 doc_id。

    这条要是反了（回落成文件名），page_map 的 title 会变成裸 doc_id，
    而 corpus 里 647 个重复 title 的 musique 就完全没法人工核对了。
    """
    for row in trip["imports"]:
        assert row["page"].get("title") == row["first_heading"], (
            f"{row['filename']}: title is {row['page'].get('title')!r}, "
            f"expected the first heading {row['first_heading']!r}"
        )
        assert row["page"]["title"] != row["doc_id"], (
            f"{row['filename']}: title fell back to the filename; "
            "heading extraction (import.service.ts:106) no longer works"
        )


def test_import_response_carries_no_document_body(trip: dict[str, Any]):
    """``baseFields`` 不含正文三兄弟，落盘时也就不该期待它们。"""
    page = trip["imports"][0]["page"]
    present = [field for field in IMPORT_RESPONSE_ABSENT if field in page]
    assert not present, f"import response now returns {present}; page.repo.ts baseFields changed"
    assert page.get("spaceId"), f"no spaceId on the imported page: {page}"


# --- 编译与质量闸门 -----------------------------------------------------------


def test_compile_spaces_accepts_a_run_immediately(trip: dict[str, Any]):
    """compile-spaces 立刻建 Run，绕过 1 小时静默期。"""
    compile_result = trip["compile"]
    accepted = compile_result.get("acceptedRunCount")
    coalesced = compile_result.get("coalescedRunCount")
    assert accepted is not None, f"no acceptedRunCount in {compile_result}"
    # 合并到一个既有 Run 也算成功触发，所以两者之和为 0 才是问题。
    assert (accepted or 0) + (coalesced or 0) > 0, (
        f"compile-spaces created and coalesced nothing: {compile_result}. "
        "Nothing would ever get indexed."
    )


def test_compile_runs_reach_a_terminal_state(trip: dict[str, Any]):
    """轮询到终态，且状态名都在已知集合里。"""
    runs = trip["runs"]
    counts: dict[str, int] = runs["status_counts"]
    assert not runs["timed_out"], f"compile still active after the timeout: {counts}"
    unknown = set(counts) - TERMINAL_RUN_STATUSES - ACTIVE_RUN_STATUSES
    assert not unknown, f"unknown compile run statuses {unknown}; update the status frozensets"
    # partial 不算失败，但一定要看见 —— 它会让指标偏低而不报错。
    if counts.get("partial"):
        print(f"NOTE {counts['partial']} run(s) finished 'partial'; some pages did not compile")
    assert counts.get("succeeded") or counts.get("partial"), (
        f"no run succeeded: {counts}. Metrics off this index would be meaningless."
    )


def test_quality_gate_fields_are_camel_case_and_all_zero(trip: dict[str, Any]):
    """四项计数用的是 camelCase，且都为 0。

    键名这一条必须单独验：PLAN.md 早先写的是 snake_case，那样每项取到 None、
    ``all(v == 0)`` 直接假通过，然后拿半成品索引跑出一堆没意义的指标。
    """
    summary = trip["quality"].get("summary") or {}
    gates = (
        "missingChunkPageCount",
        "missingEmbeddingPageCount",
        "missingSourcePageCount",
        "stalePageCount",
    )
    missing_keys = [g for g in gates if g not in summary]
    assert not missing_keys, (
        f"quality summary lacks {missing_keys}; keys are {sorted(summary)}. "
        "The ingest gate reads these names and would pass vacuously."
    )
    nonzero = {g: summary[g] for g in gates if summary[g] != 0}
    assert not nonzero, f"quality gate failed: {nonzero}"
    assert summary.get("compiledPageCount") == len(trip["imports"]), (
        f"compiled {summary.get('compiledPageCount')} pages but imported {len(trip['imports'])}"
    )


# --- 查询响应形状 -------------------------------------------------------------


def test_query_response_omits_retrieval_diagnostics(body: dict[str, Any]):
    """响应里没有 retrievalDiagnostics / retrievalScope。

    这是 audit_join 存在的全部理由。哪天它们出现在响应里，三段归因就不必再连
    数据库了 —— 那是好消息，但得先让这条断言告诉我们。
    """
    present = [key for key in STRIPPED_FROM_RESPONSE if key in body]
    assert not present, (
        f"{present} now come back in the query response (llm-wiki.controller.ts:174 "
        "no longer strips them). audit_join can read them from HTTP instead of the database."
    )


def test_query_response_holds_the_documented_camel_case_keys(body: dict[str, Any]):
    """评测直接读的那些键都在，且是 camelCase。"""
    required = (
        "answer",
        "answerMode",
        "citations",
        "citationEvidence",
        "retrievedSources",
        "snippets",
        "warnings",
        "budget",
    )
    missing = [key for key in required if key not in body]
    assert not missing, f"query response lacks {missing}; keys are {sorted(body)}"
    # snake_case 变体不该同时存在，否则说明服务端改了序列化策略。
    assert "answer_mode" not in body and "retrieved_sources" not in body


def test_answer_mode_is_one_of_the_three_known_values(body: dict[str, Any]):
    """answerMode 只有三个取值，评测按它切 knowledge 分片。"""
    mode = body["answerMode"]
    assert mode in ANSWER_MODES, f"unknown answerMode {mode!r}; evaluate.py slices on this field"
    print(f"answerMode={mode}")


def test_citation_evidence_maps_one_to_one_onto_citations(body: dict[str, Any]):
    """``citationEvidence`` 与 ``citations`` **等长、同页集合**，不是更小的子集。

    attribution 的 evidence_verifiable_rate 用的是「有 excerpts 的页 / 被引的页」，
    如果这里变成真子集，那个分母就错了。
    """
    citations, evidence = body["citations"], body["citationEvidence"]
    assert len(evidence) == len(citations), (
        f"citationEvidence has {len(evidence)} entries but citations has {len(citations)}; "
        "ai-knowledge-chat.service.ts:1036 no longer maps them one-to-one"
    )
    assert {c["sourcePageId"] for c in citations} == {e["sourcePageId"] for e in evidence}


def test_citations_are_a_subset_of_retrieved_sources(body: dict[str, Any]):
    """``retrievedSources`` ⊇ ``citations``，按 sourcePageId 比。

    检索指标一律用 retrievedSources；这条保证「用 citations 算 Recall 会低估」
    这个判断成立，而不是两个数组各说各话。
    """
    retrieved = {s["sourcePageId"] for s in body["retrievedSources"]}
    cited = {c["sourcePageId"] for c in body["citations"]}
    assert cited <= retrieved, (
        f"cited pages {sorted(cited - retrieved)} are absent from retrievedSources; "
        "the two arrays are no longer nested"
    )


def test_only_citations_carry_images(body: dict[str, Any]):
    """controller 只给 citations 做图片富化，retrievedSources 没有 images。

    所以两个数组不能整对象比 —— 必须按 sourcePageId 比。
    """
    for source in body["retrievedSources"]:
        assert "images" not in source, (
            "retrievedSources now carries images too; comparing the two arrays by object "
            "equality would start working, but by-sourcePageId stays correct either way"
        )
    for citation in body["citations"]:
        assert "images" in citation, f"citation lost its images field: {citation}"


def test_budget_caps_are_the_hardcoded_defaults(body: dict[str, Any]):
    """三个上限恒为默认值，只有用量字段随查询变。

    报告里唯一有信息量的是 ``omittedItemCount`` —— 上下文截断把召回挤掉了多少。
    上限要是变了，attribution 的 truncation_loss 口径就得跟着改。
    """
    budget = body["budget"]
    assert budget.get("maxContextLength") == BUDGET_MAX_CONTEXT_LENGTH, (
        f"maxContextLength is {budget.get('maxContextLength')}, expected "
        f"{BUDGET_MAX_CONTEXT_LENGTH}; some call site now passes a budget"
    )
    assert budget.get("responseReserve") == BUDGET_RESPONSE_RESERVE
    # perItemMaxLength 回落成等于 maxContextLength。
    assert budget.get("perItemMaxLength") == budget.get("maxContextLength")
    for field in ("usedContextLength", "includedItemCount", "omittedItemCount"):
        assert isinstance(budget.get(field), int), f"budget.{field} is not an int: {budget}"
    if budget["omittedItemCount"]:
        print(f"NOTE budget omitted {budget['omittedItemCount']} item(s) to fit the context")


def test_snippet_ids_are_bare_uuids_and_reasons_are_known(body: dict[str, Any]):
    """snippet.id 是裸 UUID（不带 chunk/capsule 前缀），reason 全在已知集合里。

    多跳指标按 ``snippets[].retrievalReasons`` 归因，出现没见过的取值就意味着
    graph_neighbor_share 之类的数悄悄漏掉了一路信号。
    """
    if not body["snippets"]:
        pytest.skip(f"answerMode={body['answerMode']} returned no snippets")
    seen: set[str] = set()
    for snippet in body["snippets"]:
        uuid.UUID(snippet["id"])  # 带前缀就会在这里抛 ValueError
        reasons = set(snippet.get("retrievalReasons") or [])
        unknown = reasons - RETRIEVAL_REASONS
        assert not unknown, f"unknown retrievalReasons {unknown} on snippet {snippet['id']}"
        seen |= reasons
        # kind 没出现在 snippet 里，所以分不出原文块还是编译产物。
        assert "kind" not in snippet, "snippets now expose kind; multihop.py could split by it"
    print(f"snippet retrievalReasons seen: {sorted(seen)}")


def test_top_level_retrieval_reasons_are_the_union_of_snippet_reasons(body: dict[str, Any]):
    """顶层 ``retrievalReasons`` 是入选条目信号的去重合集，不是别的东西。"""
    if not body["snippets"]:
        pytest.skip(f"answerMode={body['answerMode']} returned no snippets")
    top = set(body.get("retrievalReasons") or [])
    from_snippets = {r for s in body["snippets"] for r in (s.get("retrievalReasons") or [])}
    assert top == from_snippets, (
        f"top-level retrievalReasons {sorted(top)} != union over snippets "
        f"{sorted(from_snippets)}; multihop attribution reads the per-snippet field"
    )


def test_answer_has_no_inline_citation_markers(body: dict[str, Any]):
    """``[[cite:...]]`` 标记在返回前已被删净，所以答案 F1 不会被它污染。"""
    answer = body["answer"] or ""
    assert not CITATION_MARKER.search(answer), (
        "the answer still contains [[cite:...]] markers; stripCitationMarkers "
        "(ai-knowledge-chat.service.ts:518) no longer runs, and answer F1 is now polluted"
    )


def test_non_knowledge_modes_return_all_four_arrays_empty(body: dict[str, Any]):
    """``no_match`` / ``general`` 无条件清空四个数组，与检索实际结果无关。

    这正是每个检索指标都要出 ``retrieval_knowledge_only`` 第二份的原因。
    knowledge 模式下反过来要求 retrievedSources 非空 —— 空了说明召回真的什么都没找到。
    """
    mode = body["answerMode"]
    if mode == "knowledge":
        assert body["retrievedSources"], (
            "answerMode=knowledge but retrievedSources is empty; the index has content "
            "(the quality gate passed) so this is a real retrieval failure"
        )
        return
    for field in EMPTIED_WHEN_NOT_KNOWLEDGE:
        assert body[field] == [], f"answerMode={mode} but {field} is non-empty: {body[field]}"
    print(f"NOTE answerMode={mode}: generation declined to use the retrieved knowledge")


# --- 端到端：身份链条 ---------------------------------------------------------


def test_retrieved_page_ids_resolve_back_to_doc_ids(
    body: dict[str, Any], page_to_doc: dict[str, str]
):
    """响应里的 sourcePageId 全部能经 page_map 反查回 doc_id。

    这条链断了，全部检索/归因指标都会退化成 ``__unmapped__`` 占位 —— 而那不会
    报错，只会让 Recall 变成 0，看起来像检索差。这个 Space 里除了我们导的没有
    别的页，所以此处的 unmapped 一定是身份链问题，不是「库里有子集外的页」。
    """
    if not body["retrievedSources"]:
        pytest.skip(f"answerMode={body['answerMode']} returned no retrievedSources")
    unmapped = retrieval.unmapped_page_ids(body["retrievedSources"], page_to_doc)
    assert not unmapped, (
        f"{len(unmapped)} retrieved page id(s) are not in the page map: {unmapped}. "
        "Every metric would silently score these as non-gold."
    )


def test_single_sample_retrieval_and_answer_are_reported(
    trip: dict[str, Any], body: dict[str, Any], page_to_doc: dict[str, str]
):
    """把这一条样本的指标算出来打印，证明离线评测能吃下真实响应。

    **不对指标数值下断言** —— 单样本的 Recall 是 0 还是 1 说明不了检索质量，
    真正的读数要看完整实验。这里只验「算得出来」以及排名位没有错乱。
    """
    gold = tuple(trip["gold_doc_ids"])
    ranked = retrieval.ranked_doc_ids(body["retrievedSources"], page_to_doc)
    # 去重后的排名不该比原数组还长，也不该出现空串。
    assert len(ranked) <= len(body["retrievedSources"])
    assert all(ranked), f"blank doc_id in the ranking: {ranked}"

    scores = retrieval.evaluate_sample(ranked, gold, ks=(2, 5, 10))
    answer_scores = qa.score_answer(body["answer"] or "", trip["sample"]["answers"])

    print(
        f"\n  sample      {trip['sample']['sample_id']}"
        f"\n  gold        {list(gold)}"
        f"\n  ranked      {ranked}"
        f"\n  recall@10   {scores['recall@10']:.3f}   full_coverage@10 "
        f"{scores['full_coverage@10']:.3f}   mrr {scores['mrr']:.3f}"
        f"\n  answer F1   {answer_scores['f1']:.3f}"
        f"\n  answer      {(body['answer'] or '')[:200]}"
    )
    # gold 一篇都没召回时明确提示，但不 fail：单样本不足以判定检索能力。
    if not set(gold) & set(ranked):
        print("  NOTE no gold document was retrieved for this sample")


# --- 批量：入库 ---------------------------------------------------------------


def test_pipeline_ingest_maps_every_document_and_passes_the_gate(pipeline: dict[str, Any]):
    """入库阶段的验收标准：page_map 身份集合等于子集，且质量闸门通过。"""
    assert pipeline["ingest_exit"] == 0, "ingest stage returned a non-zero exit code"
    manifest = pipeline["ingest_manifest"]

    assert manifest["quality_passed"] is True, f"quality gates: {manifest['quality']['gates']}"
    assert manifest["import_failure_count"] == 0

    mapped = {row["doc_id"] for row in pipeline["page_map"]}
    assert mapped == set(pipeline["doc_ids"]), (
        f"page_map covers {sorted(mapped)} but the subset has {sorted(pipeline['doc_ids'])}"
    )
    assert manifest["page_map_rows"] == len(pipeline["doc_ids"])
    # page_id 必须两两不同，反查表靠它把 sourcePageId 映回 doc_id。
    page_ids = [row["page_id"] for row in pipeline["page_map"]]
    assert len(set(page_ids)) == len(page_ids)
    # 密钥绝不能进 manifest。
    assert manifest["connection"]["password"] == "***"


def test_pipeline_ingest_used_a_smoke_prefixed_space(pipeline: dict[str, Any]):
    """批量阶段也必须建 smoke 前缀的 Space，不能碰 bench 那几个。

    slug 前缀是靠环境变量覆盖的，这条同时验了「环境变量优先于配置文件」。
    """
    for name, space in pipeline["ingest_manifest"]["spaces"].items():
        assert space["slug"].startswith(SMOKE_SLUG_PREFIX), (
            f"{name} landed in space {space['slug']!r}, which is not smoke-prefixed; "
            "a real ingest space would have its page_map and quality counts polluted"
        )


def test_pipeline_ingest_resume_reimports_nothing(pipeline: dict[str, Any]):
    """续跑：第二遍入库一篇都不该重导，page_map 也不该多出行。

    这条坏了不会报错 —— 只会让 page_map 出现重复行，然后反查表里同一个 doc_id
    对上多个 page_id，检索指标跟着虚高或虚低。
    """
    assert pipeline["ingest_resume_exit"] == 0
    resumed = pipeline["ingest_manifest_resume"]
    imports = resumed["imports"][0]
    assert imports["imported"] == 0, f"resume re-imported {imports['imported']} document(s)"
    assert imports["skipped_already_present"] == len(pipeline["doc_ids"])
    assert imports["failures"] == []
    # 行数与身份集合都不能变。
    before, after = pipeline["page_map"], pipeline["page_map_after_resume"]
    assert len(after) == len(before), f"page_map grew from {len(before)} to {len(after)} rows"
    assert {(r["dataset"], r["doc_id"]) for r in after} == {
        (r["dataset"], r["doc_id"]) for r in before
    }


# --- 批量：查询 ---------------------------------------------------------------


def test_pipeline_query_writes_exactly_one_row_per_sample(pipeline: dict[str, Any]):
    """查询阶段的验收标准：全量 sample_id 恰好一行，且都成功。"""
    assert pipeline["query_exit"] == 0
    rows = pipeline["responses"]
    ids = [r["sample_id"] for r in rows]
    assert ids == pipeline["sample_ids"], f"expected {pipeline['sample_ids']}, got {ids}"
    assert len(set(ids)) == len(ids), "duplicate sample_id in the response file"

    failed = [(r["sample_id"], r["http_status"], r["error"]) for r in rows if r["http_status"] != 200]
    assert not failed, f"these queries failed: {failed}"
    # 完整响应体必须落盘 —— 重跑要烧 LLM 调用，字段不全就等于要重跑。
    for row in rows:
        assert isinstance(row["response"], dict) and row["response"].get("answerMode")

    manifest = pipeline["query_manifest"]
    assert manifest["total_failures"] == 0
    assert manifest["model_configs_match_ingest"] is True, (
        "model configs drifted between ingest and query within a single smoke run"
    )
    modes = [r["response"]["answerMode"] for r in rows]
    print(f"answer modes: {modes}")


def test_pipeline_query_resume_reissues_nothing(pipeline: dict[str, Any]):
    """续跑：第二遍查询一条都不该重发，行数也不该变。

    这条坏了要么重复烧 LLM 调用，要么产生重复行 —— 后者会让 evaluate 直接以
    「duplicate sample_id」终止。
    """
    assert pipeline["query_resume_exit"] == 0
    resumed = pipeline["query_manifest_resume"]["datasets"][0]
    assert resumed["requested"] == 0, f"resume re-issued {resumed['requested']} query/queries"
    assert resumed["skipped_already_done"] == len(pipeline["sample_ids"])

    before, after = pipeline["responses"], pipeline["responses_after_resume"]
    assert len(after) == len(before), f"response file grew from {len(before)} to {len(after)} rows"
    assert [r["sample_id"] for r in after] == [r["sample_id"] for r in before]


# --- 批量：报告 ---------------------------------------------------------------


def test_pipeline_report_covers_every_sample(pipeline: dict[str, Any]):
    """报告阶段：三份产物齐全，覆盖率与失败数明确，没有漏样本。"""
    assert pipeline["report_exit"] == 0
    summary = pipeline["metrics"]["datasets"][0]
    n = len(pipeline["sample_ids"])

    assert summary["responses_evaluated"] == n
    assert summary["samples_in_subset"] == n
    assert summary["missing_responses"] == [], (
        f"these samples never got a response: {summary['missing_responses']}"
    )
    assert summary["http_failures"] == 0
    assert len(pipeline["per_sample"]) == n
    assert {r["sample_id"] for r in pipeline["per_sample"]} == set(pipeline["sample_ids"])


def test_pipeline_report_resolves_every_retrieved_page(pipeline: dict[str, Any]):
    """``unmapped_page_ids`` 必须为空：Space 里只有我们导的那几篇。

    非空就说明 page_map 反查断了 —— 那不会报错，只会让这些页永远算不上 gold，
    Recall 因此虚低，看起来像检索差。
    """
    summary = pipeline["metrics"]["datasets"][0]
    assert summary["unmapped_page_ids"] == [], (
        f"{len(summary['unmapped_page_ids'])} retrieved page id(s) are absent from page_map: "
        f"{summary['unmapped_page_ids'][:5]}"
    )


def test_pipeline_report_metrics_are_well_formed(pipeline: dict[str, Any]):
    """指标本身要成形：取值在定义域内、两份检索切片都在。

    **不断言指标高低** —— 三条样本说明不了检索质量。这里验的是「算得出来且没算错口径」。
    """
    summary = pipeline["metrics"]["datasets"][0]
    for key in ("retrieval", "retrieval_knowledge_only", "attribution", "multihop", "stratified"):
        assert key in summary, f"summary lacks {key}; keys are {sorted(summary)}"

    for slice_name in ("retrieval", "retrieval_knowledge_only"):
        for metric, value in summary[slice_name].items():
            assert 0.0 <= value <= 1.0, f"{slice_name}.{metric} = {value} is out of [0, 1]"

    # EM 报出来但预期为 0（散文答案对不上短跨度参考，见 metrics/qa.py）。
    # 这里不断言它必须为 0 —— 换了 answer prompt 后它可能变正，那是形态变化不是失败。
    assert 0.0 <= summary["qa"]["em"] <= 1.0
    assert 0.0 <= summary["qa"]["f1"] <= 1.0

    # knowledge 切片是全样本的子集，所以计数不可能更多。
    assert summary["knowledge_answer_count"] <= summary["responses_evaluated"]
    modes = summary["answer_mode_distribution"]
    assert abs(sum(modes.values()) - 1.0) < 1e-9, f"mode shares do not sum to 1: {modes}"

    print(
        f"  R@10 {summary['retrieval'].get('recall@10', 0.0):.3f}"
        f" (knowledge-only {summary['retrieval_knowledge_only'].get('recall@10', 0.0):.3f})"
        f"   EM {summary['qa']['em']:.3f}   F1 {summary['qa']['f1']:.3f}"
    )


def test_pipeline_report_md_is_readable_and_states_its_caveats(pipeline: dict[str, Any]):
    """人读报告必须带上架构边界那段说明。

    不写清楚「召回跑在编译产物上而非原文」，读者会拿这些数直接跟公开 baseline 比，
    而那个比较是无效的。这段是结论的一部分，不是客套。
    """
    report = pipeline["report_md"]
    assert report.startswith("# Akasha-Benchmark 评测报告"), report[:80]
    for fragment in (
        "这些数字该怎么读",
        "knowledge_chunks",
        "无效的",
        "模型配置",
        # EM 报出来了，所以「为什么预期是 0」必须同时在报告里：
        # 熟悉 hotpotqa 的读者看到 0.0000 会以为系统坏了。
        "Exact Match 预期就是 0.0000",
        "答案**形状**的探针",
        pipeline["dataset"],
    ):
        assert fragment in report, f"report.md lacks {fragment!r}"
    # 检索表两份都要出现。
    assert "| 指标 | 全样本 | 仅 knowledge |" in report
    # 分层表要有 EM 列。
    assert "| EM | F1 |" in report


def test_the_smoke_space_was_cleaned_up(trip: dict[str, Any]):
    """冒烟建的 Space 跑完要删掉，否则真实入库时会看到一堆残留。"""
    if os.environ.get("AKASHA_LIVE_KEEP") == "1":
        pytest.skip("AKASHA_LIVE_KEEP=1 asked to keep the space")
    if os.environ.get("AKASHA_LIVE_REPLAY"):
        pytest.skip("replaying a recorded round trip; nothing was created this time")
    assert trip["space_deleted"], (
        f"smoke space {trip['space']['id']} could not be deleted; remove it by hand so it "
        "does not get picked up as a stale space later"
    )
