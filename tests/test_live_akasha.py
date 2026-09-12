"""真实 Akasha 接口契约；默认跳过。

运行：AKASHA_LIVE=1 uv run pytest tests/test_live_akasha.py -v
需要评测库中的 OWNER 连接凭据和已抽取的子集。测试调用真实模型，创建独立
smoke Space，结束后删除；过程写入 data/smoke/。

可选环境变量（均以 AKASHA_LIVE_ 开头）：
REPLAY：离线重放 roundtrip.json；KEEP=1：保留临时 Space。
LABEL：索引层标签；DATASET：数据集；SAMPLE_ID：指定样本。
DISTRACTORS：干扰文档数；TIMEOUT：编译等待秒数。
端到端流程与续跑验证已移至平台的「小样本验证」。
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
from akasha_benchmark.datasets import CanonicalSample
from akasha_benchmark.ingest import MODEL_FEATURES
from akasha_benchmark.store import DEFAULT_DB_PATH, connect, repo
from akasha_benchmark.io_utils import atomic_write_json, sha256_text, utc_now
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


def _materialize_subset(dataset: str, label: str) -> tuple[list[CanonicalSample], Path]:
    """从**库里**取子集，并把 md 正文摊到一个临时目录。"""
    connection = connect(DEFAULT_DB_PATH, read_only=True)
    try:
        layer = repo.index_layer_by_label(connection, label)
        if layer is None:
            pytest.skip(
                f"库里没有标签为 {label!r} 的索引层；先跑：make subset LABEL={label}"
            )
        layer_id = int(layer["id"])
        rows = repo.subset_samples(connection, layer_id, dataset)
        docs = repo.subset_docs(connection, layer_id, dataset)
    finally:
        connection.close()

    if not rows:
        pytest.skip(f"索引层 {label!r} 里没有 {dataset} 的子集样本")

    corpus = Path("data") / "smoke" / f"subset-{label}-{dataset}"
    corpus.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        (corpus / f"{doc['doc_id']}.md").write_text(
            doc["md_text"], encoding="utf-8", newline="\n"
        )

    on_disk = {doc["doc_id"] for doc in docs}
    eligible = [
        CanonicalSample.model_validate(
            {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "dataset_sample_id": row["dataset_sample_id"],
                "question": row["question"],
                "answers": list(row["answers"]),
                "gold_doc_ids": list(row["gold_doc_ids"]),
                "metadata": row["metadata"],
            }
        )
        for row in rows
        # gold 不全的排除掉：那种样本「召回不到」时分不清是检索问题还是
        # 根本没导进去。
        if row["gold_doc_ids"] and all(d in on_disk for d in row["gold_doc_ids"])
    ]
    return eligible, corpus


def _eligible_samples(dataset: str, label: str) -> tuple[list[CanonicalSample], Path]:
    """gold 文档齐全的样本，以及摊好正文的 corpus 目录。"""
    eligible, corpus = _materialize_subset(dataset, label)
    if not eligible:
        pytest.skip(
            f"索引层 {label!r} 的 {dataset} 子集里，没有一条样本的 gold 是齐的"
        )
    return eligible, corpus


def _pick_sample(dataset: str, label: str) -> tuple[CanonicalSample, Path]:
    """单样本往返用的那一条。``AKASHA_LIVE_SAMPLE_ID`` 可以指定。"""
    eligible, corpus = _eligible_samples(dataset, label)
    wanted = os.environ.get("AKASHA_LIVE_SAMPLE_ID")
    if not wanted:
        return eligible[0], corpus
    for sample in eligible:
        if sample.sample_id == wanted:
            return sample, corpus
    pytest.skip(f"sample {wanted!r} is not in the subset, or its gold md is missing")


def _distractor_ids(corpus: Path, gold: tuple[str, ...], count: int) -> list[str]:
    """挑若干非 gold 文档一起导入。"""
    others = sorted(p.stem for p in corpus.glob("*.md") if p.stem not in set(gold))
    return others[:count]


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
    """真的连一次 Akasha，把一条样本走完入库到查询，返回全部观察结果。"""
    label = os.environ.get("AKASHA_LIVE_LABEL") or "run001"
    dataset = os.environ.get("AKASHA_LIVE_DATASET") or "hotpotqa"
    timeout = _env_int("AKASHA_LIVE_TIMEOUT", 900)
    sample, corpus = _pick_sample(dataset, label)
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
        "label": label,
        "dataset": dataset,
        "sample": sample.model_dump(mode="json"),
        "gold_doc_ids": list(sample.gold_doc_ids),
        "distractor_doc_ids": distractors,
    }
    try:
        return _execute(config, observed, sample, corpus, doc_ids, dataset, label, timeout)
    finally:
        _persist(observed)


def _execute(
    config: AkashaConfig,
    observed: dict[str, Any],
    sample: CanonicalSample,
    corpus: Path,
    doc_ids: list[str],
    dataset: str,
    label: str,
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
            name=f"smoke {dataset} {label}"[:100],
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
    """把这一趟的观察结果落盘。"""
    if "space" not in observed:
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = Path("data") / "smoke" / f"{stamp}-roundtrip.json"
    atomic_write_json(path, observed)
    observed["artifact_path"] = str(path)
    print(f"round trip written to {path}")


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
    """登录成功只 set httpOnly 的 authToken cookie，响应体是空的。"""
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
    """四项模型配置都能读到，否则 ingest 的快照比对形同虚设。"""
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
    """title 来自首个 Markdown heading，所以文件名可以专职当 doc_id。"""
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
    """四项计数用的是 camelCase，且都为 0。"""
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
    """响应里没有 retrievalDiagnostics / retrievalScope。"""
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
    """``citationEvidence`` 与 ``citations`` **等长、同页集合**，不是更小的子集。"""
    citations, evidence = body["citations"], body["citationEvidence"]
    assert len(evidence) == len(citations), (
        f"citationEvidence has {len(evidence)} entries but citations has {len(citations)}; "
        "ai-knowledge-chat.service.ts:1036 no longer maps them one-to-one"
    )
    assert {c["sourcePageId"] for c in citations} == {e["sourcePageId"] for e in evidence}


def test_citations_are_a_subset_of_retrieved_sources(body: dict[str, Any]):
    """``retrievedSources`` ⊇ ``citations``，按 sourcePageId 比。"""
    retrieved = {s["sourcePageId"] for s in body["retrievedSources"]}
    cited = {c["sourcePageId"] for c in body["citations"]}
    assert cited <= retrieved, (
        f"cited pages {sorted(cited - retrieved)} are absent from retrievedSources; "
        "the two arrays are no longer nested"
    )


def test_only_citations_carry_images(body: dict[str, Any]):
    """controller 只给 citations 做图片富化，retrievedSources 没有 images。"""
    for source in body["retrievedSources"]:
        assert "images" not in source, (
            "retrievedSources now carries images too; comparing the two arrays by object "
            "equality would start working, but by-sourcePageId stays correct either way"
        )
    for citation in body["citations"]:
        assert "images" in citation, f"citation lost its images field: {citation}"


def test_budget_caps_are_the_hardcoded_defaults(body: dict[str, Any]):
    """三个上限恒为默认值，只有用量字段随查询变。"""
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
    """snippet.id 是裸 UUID（不带 chunk/capsule 前缀），reason 全在已知集合里。"""
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
    """``no_match`` / ``general`` 无条件清空四个数组，与检索实际结果无关。"""
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
    """响应里的 sourcePageId 全部能经 page_map 反查回 doc_id。"""
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
    """把这一条样本的指标算出来打印，证明离线评测能吃下真实响应。"""
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


# --- 批量：查询 ---------------------------------------------------------------


# --- 批量：报告 ---------------------------------------------------------------


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
