"""入库与查询跑在 mock 的 Akasha 上，覆盖请求形状与断点续跑。

真正跑一遍仍然需要在线的 Akasha；这里锁住的是那些容易悄悄搞错的地方 ——
multipart 的字段名、OWNER 闸门、质量闸门、续跑时的跳过逻辑，
以及失败必须落盘而不是被丢掉。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from akasha_benchmark.akasha_client import AkashaClient, AkashaError, unwrap_envelope
from akasha_benchmark.config import AkashaConfig
from akasha_benchmark.io_utils import atomic_write_json, atomic_write_jsonl, read_jsonl
from akasha_benchmark import ingest as ingest_mod
from akasha_benchmark import run_queries as rq_mod

DATASET = "hotpotqa"
RUN_ID = "r1"


# 与 _prepare_query_stage 写进假 manifest 的值共用一处：漂移闸门是整体相等判断，
# 两边一旦不同步，每个查询测试都会以「配置漂移」失败。
MOCK_MODEL_CONFIGS = {"configs": [{"feature": "embedding", "model": "m1", "apiKeySet": True}]}


def enveloped(payload: Any, status: int = 200) -> httpx.Response:
    """按真实服务端的形状包一层 ``{data, success, status}``。

    ``main.ts:160`` 给所有路由挂了 ``TransformHttpResponseInterceptor``，只有
    ``@SkipTransform()`` 的 handler 例外（mcp / health / robots.txt，本评测都不用）。
    替身**必须**照这个形状返回：早先它返回裸响应，于是这里 73 个测试全绿，
    而真实服务上每个阶段都在静默读空 —— 最坏的一处是质量闸门四项计数全取到
    ``None``，``all(value == 0)`` 假通过。替身照我们的理解写，理解错了它也照样绿，
    所以这一层是拿在线冒烟换回来的，别再把它改回裸响应。
    """
    return httpx.Response(status, json={"data": payload, "success": True, "status": status})


class FakeAkasha:
    """入库与查询用到的那几个端点的最小替身。

    ``role`` 和 ``quality_clean`` 用来构造两种失败场景，
    ``fail_import_for`` / ``fail_query_for`` 用来指定哪些条目要失败。

    响应一律经 :func:`enveloped` 套信封，与真实服务端一致。
    """

    def __init__(self, *, role: str = "owner", quality_clean: bool = True) -> None:
        self.role = role
        self.quality_clean = quality_clean
        self.requests: list[tuple[str, str]] = []
        self.imported: list[dict[str, Any]] = []
        self.queries: list[dict[str, Any]] = []
        self.spaces: dict[str, dict[str, Any]] = {}
        self.fail_import_for: set[str] = set()
        self.fail_query_for: set[str] = set()
        # 这些 query 直接抛传输层异常，模拟断连/超时而不是 HTTP 错误码。
        self.raise_transport_for: set[str] = set()
        self._next_page = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append((request.method, path))

        if path == "/api/auth/login":
            # 真实的 login handler 没有返回值，所以信封里没有 data 键，只有
            # {success, status}，剥出来是 None。
            return httpx.Response(
                200,
                json={"success": True, "status": 200},
                headers={"set-cookie": "authToken=jwt; Path=/"},
            )
        if path == "/api/users/me":
            return enveloped(
                {
                    "user": {"id": "u1", "role": self.role},
                    "workspace": {"id": "w1", "name": "bench"},
                }
            )
        if path == "/api/llm-wiki/admin/model-configs":
            # 真实形状是 {"configs": [{"feature": ..., "model": ...}, ...]}。
            return enveloped(MOCK_MODEL_CONFIGS)
        if path == "/api/spaces":
            return enveloped({"items": list(self.spaces.values()), "meta": {}})
        if path == "/api/spaces/create":
            body = json.loads(request.content)
            space = {"id": f"space-{body['slug']}", "slug": body["slug"], "name": body["name"]}
            self.spaces[body["slug"]] = space
            return enveloped(space)
        if path == "/api/pages/import":
            raw = request.content.decode("utf-8", "replace")
            # 文件名承担 doc_id，heading 承担 title，两者互不干扰。
            name = raw.split('filename="', 1)[1].split('"', 1)[0]
            doc_id = name.removesuffix(".md")
            if doc_id in self.fail_import_for:
                return httpx.Response(500, json={"message": "boom"})  # 错误响应不套信封
            self._next_page += 1
            space_id = raw.split('name="spaceId"', 1)[1].split("\r\n\r\n", 1)[1].split("\r\n", 1)[0]
            # page_id 按序号发，不按 doc_id：跨数据集撞 doc_id 时两篇必须拿到
            # 不同的 page_id，否则反查表会把两组混成一条。
            page = {"id": f"page-{self._next_page}", "title": f"T{doc_id}"}
            self.imported.append(
                {"doc_id": doc_id, "raw_has_space_id": "spaceId" in raw, "space_id": space_id}
            )
            return enveloped(page)
        if path == "/api/llm-wiki/admin/compile-spaces":
            return enveloped(
                {"requestedSpaceCount": 1, "acceptedRunCount": 1, "coalescedRunCount": 0}
            )
        if path == "/api/llm-wiki/admin/diagnostics/summary":
            return enveloped({"statusCounts": {"succeeded": 1}})
        if path == "/api/llm-wiki/admin/diagnostics/quality":
            counts = 0 if self.quality_clean else 3
            return enveloped(
                {
                    "summary": {
                        "pageCount": 2,
                        "compiledPageCount": 2,
                        "stalePageCount": 0,
                        "missingSourcePageCount": 0,
                        "missingChunkPageCount": counts,
                        "missingEmbeddingPageCount": 0,
                    },
                    "spaces": [],
                    "topIssues": [],
                }
            )
        if path == "/api/llm-wiki/query":
            body = json.loads(request.content)
            self.queries.append(body)
            if body["query"] in self.raise_transport_for:
                raise httpx.ConnectError("connection reset", request=request)
            if body["query"] in self.fail_query_for:
                return httpx.Response(503, json={"message": "unavailable"})
            return enveloped(
                {
                    "answer": "an answer",
                    "answerMode": "knowledge",
                    "citations": [],
                    "citationEvidence": [],
                    "retrievedSources": [{"sourcePageId": "page-0"}],
                    "snippets": [],
                    "warnings": [],
                    "budget": {},
                    "completenessNotice": None,
                }
            )
        return httpx.Response(404, json={"message": f"unhandled {path}"})


@pytest.fixture
def config() -> AkashaConfig:
    return AkashaConfig(
        base_url="http://akasha.test",
        email="e@x.com",
        password="pw",
        request_interval_seconds=0.0,
        poll_interval_seconds=0.0,
    )


def client_for(fake: FakeAkasha, config: AkashaConfig) -> AkashaClient:
    """把客户端的 transport 换成 mock。"""
    client = AkashaClient(config)
    client._client = httpx.Client(transport=httpx.MockTransport(fake.handler))
    return client


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # 正常信封。
        ({"data": {"id": "p1"}, "success": True, "status": 200}, {"id": "p1"}),
        # login：handler 无返回值，信封里没有 data 键。
        ({"success": True, "status": 200}, None),
        # data 是数组、是 None，都照剥。
        ({"data": [1, 2], "success": True, "status": 200}, [1, 2]),
        ({"data": None, "success": True, "status": 200}, None),
    ],
)
def test_unwrap_envelope_strips_the_global_wrapper(body: Any, expected: Any):
    """全局拦截器的 ``{data, success, status}`` 要剥掉，各阶段才能按文档读字段。"""
    assert unwrap_envelope(body) == expected


@pytest.mark.parametrize(
    "body",
    [
        # 载荷里正常带一个叫 data 的字段 —— 不是信封，不能动。
        {"data": {"x": 1}, "meta": {"page": 1}},
        # 缺 success / status，或类型不对，都不是信封。
        {"data": 1, "success": True},
        {"data": 1, "status": 200},
        {"data": 1, "success": "yes", "status": 200},
        {"data": 1, "success": True, "status": "200"},
        # 非 dict 一律原样返回。
        [1, 2, 3],
        "plain text",
        None,
    ],
)
def test_unwrap_envelope_leaves_everything_else_alone(body: Any):
    """判据收紧到信封自身形状：只看有没有 ``data`` 会把正常载荷剥坏。"""
    assert unwrap_envelope(body) == body


def test_client_unwraps_envelopes_end_to_end(config: AkashaConfig):
    """经 :class:`AkashaClient` 出来的响应已经剥好，调用方不必自己判断。

    这条是拿在线冒烟换回来的：客户端早先不剥信封，于是 ``users/me`` 取不到
    role（OWNER 闸门永远拒绝执行）、导入取不到 id（每篇记成失败）、质量诊断
    取不到 summary（四项闸门全 ``None``，``all(value == 0)`` **假通过**）。
    这些都不报错，只是静默读空。
    """
    fake = FakeAkasha()
    with client_for(fake, config) as client:
        client.login()
        me = client.current_user()
        # 剥之前这两个都是 None。
        assert (me["user"] or {})["role"] == "owner"
        assert (me["workspace"] or {})["id"] == "w1"
        quality = client.quality_diagnostics(["space-1"])
        assert quality["summary"]["missingChunkPageCount"] == 0
        spaces = client.list_spaces()
        assert "items" in spaces and "meta" in spaces


def test_login_sets_cookie_and_import_sends_multipart(tmp_path: Path, config: AkashaConfig):
    """导入必须以 multipart 发出，且带上 spaceId 字段。"""
    fake = FakeAkasha()
    md = tmp_path / "42.md"
    md.write_text("# Title\n\nBody\n", encoding="utf-8")

    with client_for(fake, config) as client:
        client.login()
        page = client.import_page(md, "space-1")

    assert page["id"] == "page-1"
    assert fake.imported[0] == {
        "doc_id": "42",
        "raw_has_space_id": True,
        "space_id": "space-1",
    }


def test_login_without_cookie_is_an_error(config: AkashaConfig):
    """返回 200 但没有 authToken cookie（例如开了 MFA），要当失败处理。"""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"userHasMfa": True})

    client = AkashaClient(config)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    with client, pytest.raises(AkashaError, match="no authToken cookie"):
        client.login()


def test_query_records_non_2xx_without_raising(config: AkashaConfig):
    """查询的非 2xx 不抛异常，交给调用方落盘。"""
    fake = FakeAkasha()
    fake.fail_query_for.add("bad question")
    with client_for(fake, config) as client:
        client.login()
        response = client.query("bad question", ["space-1"])
    assert response.status == 503
    assert response.body["message"] == "unavailable"


def _write_subset(
    data: Path,
    doc_ids: list[str],
    questions: list[tuple[str, str]],
    dataset: str = DATASET,
) -> None:
    """伪造一份子集产出，含 md 文件与其 sha256。"""
    subset = data / "subsets" / RUN_ID / dataset
    corpus = subset / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    hashes = {}
    from akasha_benchmark.io_utils import sha256_text

    for doc_id in doc_ids:
        # 正文带上数据集名：跨组撞 doc_id 时两篇内容不同，sha256 也不同，
        # 这样「导错了组」不会碰巧通过哈希校验。
        markdown = f"# T{doc_id}\n\nbody {doc_id} of {dataset}\n"
        (corpus / f"{doc_id}.md").write_text(markdown, encoding="utf-8", newline="\n")
        hashes[doc_id] = sha256_text(markdown)

    atomic_write_jsonl(
        subset / "samples.jsonl",
        [
            {
                "dataset": dataset,
                "sample_id": f"{dataset}:{sid}",
                "dataset_sample_id": sid,
                "question": q,
                "answers": ["a"],
                "gold_doc_ids": [doc_ids[0]],
                "metadata": {"type": "bridge"},
            }
            for sid, q in questions
        ],
    )
    atomic_write_json(subset / "manifest.json", {"corpus_md_sha256": hashes})


@pytest.fixture
def staged(tmp_path: Path) -> Path:
    """准备好入库需要的输入。"""
    data = tmp_path / "data"
    _write_subset(data, ["0", "1", "2"], [("s1", "first question"), ("s2", "second question")])
    return data


def _patch_client(monkeypatch: pytest.MonkeyPatch, module, fake: FakeAkasha, config: AkashaConfig):
    """把目标模块里的配置加载与客户端都替换掉。"""
    monkeypatch.setattr(module, "load_config", lambda _p: config)

    class Patched(AkashaClient):
        def __init__(self, cfg: AkashaConfig) -> None:
            super().__init__(cfg)
            self._client = httpx.Client(transport=httpx.MockTransport(fake.handler))

    monkeypatch.setattr(module, "AkashaClient", Patched)


def test_ingest_maps_every_doc_and_passes_quality(
    staged: Path, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """正常路径：每篇都进 page_map，质量闸门通过。"""
    fake = FakeAkasha()
    _patch_client(monkeypatch, ingest_mod, fake, config)

    assert ingest_mod.run(RUN_ID, [DATASET], None, staged) == 0

    page_map = list(read_jsonl(ingest_mod.ingest_dir(RUN_ID, staged) / "page_map.jsonl"))
    assert {row["doc_id"] for row in page_map} == {"0", "1", "2"}
    manifest = json.loads(
        (ingest_mod.ingest_dir(RUN_ID, staged) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["quality_passed"] is True
    assert manifest["page_map_rows"] == 3
    # 密钥绝不能出现在 manifest 里。
    assert manifest["connection"]["password"] == "***"


def test_ingest_refuses_a_non_owner_user(
    staged: Path, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """非 OWNER 必须在导入前就被拦住，一篇都不能导。"""
    fake = FakeAkasha(role="member")
    _patch_client(monkeypatch, ingest_mod, fake, config)
    # 走 main()，因为把异常转成退出码是在那一层做的。
    exit_code = ingest_mod.main(
        ["--run-id", RUN_ID, "--dataset", DATASET, "--data-dir", str(staged)]
    )
    assert exit_code == 1
    assert not fake.imported


def test_ingest_fails_the_quality_gate(
    staged: Path, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """质量报告非 0 时不能放行到查询。"""
    fake = FakeAkasha(quality_clean=False)
    _patch_client(monkeypatch, ingest_mod, fake, config)
    assert ingest_mod.run(RUN_ID, [DATASET], None, staged) == 1


def test_ingest_resumes_and_skips_imported_docs(
    staged: Path, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """断点续跑：第二次运行不重复导入。"""
    fake = FakeAkasha()
    _patch_client(monkeypatch, ingest_mod, fake, config)
    ingest_mod.run(RUN_ID, [DATASET], None, staged)
    assert len(fake.imported) == 3

    second = FakeAkasha()
    _patch_client(monkeypatch, ingest_mod, second, config)
    ingest_mod.run(RUN_ID, [DATASET], None, staged)
    # 已经全部映射过，所以第二次一条都不该重新导入。
    assert second.imported == []


OTHER_DATASET = "2wikimultihopqa"


def test_ingest_keeps_colliding_doc_ids_of_two_datasets_apart(
    tmp_path: Path, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """两组撞 doc_id 时，page_map 必须按 (dataset, doc_id) 分别记全。

    doc_id 是各数据集内部的裸 ID，跨组会撞（锁定的 run001 子集里
    hotpotqa×2wiki 撞 15 个）。只按 doc_id 建续跑字典的话，后一组会覆盖
    前一组的行，前一组这些 doc 就被当成已导入而跳过。
    """
    data = tmp_path / "data"
    # "1" 和 "2" 两组都有，"9" 只有 hotpotqa，"7" 只有 2wiki。
    _write_subset(data, ["1", "2", "9"], [("s1", "q1")], dataset=DATASET)
    _write_subset(data, ["1", "2", "7"], [("s2", "q2")], dataset=OTHER_DATASET)

    fake = FakeAkasha()
    _patch_client(monkeypatch, ingest_mod, fake, config)
    assert ingest_mod.run(RUN_ID, [DATASET, OTHER_DATASET], None, data) == 0

    rows = list(read_jsonl(ingest_mod.ingest_dir(RUN_ID, data) / "page_map.jsonl"))
    by_dataset: dict[str, set[str]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], set()).add(row["doc_id"])

    # 每组的映射集合等于它自己的子集，撞的那两个不能被吞掉。
    assert by_dataset == {DATASET: {"1", "2", "9"}, OTHER_DATASET: {"1", "2", "7"}}
    assert len(rows) == 6
    # page_id 不能共用：反查表靠它把 sourcePageId 映回 doc_id。
    assert len({row["page_id"] for row in rows}) == 6
    # 撞的 doc_id 分别进了各自数据集的 space。
    spaces = {(row["dataset"], row["doc_id"]): row["space_id"] for row in rows}
    assert spaces[(DATASET, "1")] != spaces[(OTHER_DATASET, "1")]

    manifest = json.loads(
        (ingest_mod.ingest_dir(RUN_ID, data) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["page_map_rows"] == 6

    # 续跑：两组都已导全，一条都不该重发。
    second = FakeAkasha()
    _patch_client(monkeypatch, ingest_mod, second, config)
    assert ingest_mod.run(RUN_ID, [DATASET, OTHER_DATASET], None, data) == 0
    assert second.imported == []


def test_ingest_reports_import_failures(
    staged: Path, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """导入失败要计数并写进 manifest，不能静默跳过。"""
    fake = FakeAkasha()
    fake.fail_import_for.add("1")
    _patch_client(monkeypatch, ingest_mod, fake, config)
    assert ingest_mod.run(RUN_ID, [DATASET], None, staged) == 1

    manifest = json.loads(
        (ingest_mod.ingest_dir(RUN_ID, staged) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["import_failure_count"] == 1


def test_ingest_detects_markdown_edited_after_sampling(
    staged: Path, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """md 与抽子集时记录的 sha256 不符时终止，否则入库内容无从追溯。"""
    (staged / "subsets" / RUN_ID / DATASET / "corpus" / "1.md").write_text(
        "# T1\n\ntampered\n", encoding="utf-8"
    )
    fake = FakeAkasha()
    _patch_client(monkeypatch, ingest_mod, fake, config)
    exit_code = ingest_mod.main(
        ["--run-id", RUN_ID, "--dataset", DATASET, "--data-dir", str(staged)]
    )
    assert exit_code == 1
    # 被改过的那篇绝不能进库。
    assert "1" not in {row["doc_id"] for row in fake.imported}


def _prepare_query_stage(data: Path) -> None:
    """伪造入库的 manifest，让查询能读到 space id 与模型配置。"""
    atomic_write_json(
        ingest_mod.ingest_dir(RUN_ID, data) / "manifest.json",
        {
            "spaces": {DATASET: {"id": "space-1"}},
            "model_configs": MOCK_MODEL_CONFIGS,
        },
    )


def test_run_queries_writes_one_row_per_sample(
    staged: Path, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """验收标准：每个 sample_id 恰好一行，请求体形状固定。"""
    _prepare_query_stage(staged)
    fake = FakeAkasha()
    _patch_client(monkeypatch, rq_mod, fake, config)

    assert rq_mod.run(RUN_ID, [DATASET], None, staged, None, None, False) == 0
    rows = list(read_jsonl(rq_mod.responses_dir(RUN_ID, staged) / f"{DATASET}.jsonl"))
    assert [r["sample_id"] for r in rows] == [f"{DATASET}:s1", f"{DATASET}:s2"]
    assert all(r["http_status"] == 200 for r in rows)
    # 串行、单个 space、type=user、不带 chatContext（本评测是单轮）。
    assert fake.queries[0] == {
        "query": "first question",
        "spaceIds": ["space-1"],
        "type": "user",
    }


def test_run_queries_records_failures_as_rows(
    staged: Path, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """失败也占一行，并计入 manifest 的失败数。"""
    _prepare_query_stage(staged)
    fake = FakeAkasha()
    fake.fail_query_for.add("second question")
    _patch_client(monkeypatch, rq_mod, fake, config)

    rq_mod.run(RUN_ID, [DATASET], None, staged, None, None, False)
    rows = list(read_jsonl(rq_mod.responses_dir(RUN_ID, staged) / f"{DATASET}.jsonl"))
    assert len(rows) == 2
    failed = [r for r in rows if r["http_status"] != 200]
    assert len(failed) == 1
    manifest = json.loads(
        (rq_mod.responses_dir(RUN_ID, staged) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["total_failures"] == 1


def test_run_queries_records_transport_errors_instead_of_crashing(
    staged: Path, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """断连/超时要落盘成失败行，并且不能中断后面的样本。

    ``httpx.RequestError`` 不是 ``OSError`` 的子类，漏掉它的话跑到一半
    网络抖一下整个阶段就带 traceback 崩掉，那一行也不会落盘。
    """
    _prepare_query_stage(staged)
    fake = FakeAkasha()
    fake.raise_transport_for.add("first question")
    _patch_client(monkeypatch, rq_mod, fake, config)

    assert rq_mod.run(RUN_ID, [DATASET], None, staged, None, None, False) == 0

    rows = list(read_jsonl(rq_mod.responses_dir(RUN_ID, staged) / f"{DATASET}.jsonl"))
    # 崩掉的那条照样占一行，后一条继续跑完。
    assert [r["sample_id"] for r in rows] == [f"{DATASET}:s1", f"{DATASET}:s2"]
    failed, ok = rows[0], rows[1]
    assert failed["http_status"] == 0
    assert failed["response"] is None
    assert failed["error"].startswith("ConnectError:")
    assert ok["http_status"] == 200

    manifest = json.loads(
        (rq_mod.responses_dir(RUN_ID, staged) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["total_failures"] == 1
    assert manifest["datasets"][0]["requested"] == 2


def test_run_queries_resumes_from_existing_rows(
    staged: Path, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """断点续跑：已完成的 sample_id 不再请求。"""
    _prepare_query_stage(staged)
    fake = FakeAkasha()
    _patch_client(monkeypatch, rq_mod, fake, config)
    rq_mod.run(RUN_ID, [DATASET], None, staged, None, None, False)
    assert len(fake.queries) == 2

    second = FakeAkasha()
    _patch_client(monkeypatch, rq_mod, second, config)
    rq_mod.run(RUN_ID, [DATASET], None, staged, None, None, False)
    assert second.queries == []


def test_run_queries_stops_on_model_config_drift(
    staged: Path, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """模型配置与入库时不一致时终止，除非显式允许。"""
    atomic_write_json(
        ingest_mod.ingest_dir(RUN_ID, staged) / "manifest.json",
        {"spaces": {DATASET: {"id": "space-1"}}, "model_configs": {"embedding": {"model": "OLD"}}},
    )
    fake = FakeAkasha()
    _patch_client(monkeypatch, rq_mod, fake, config)

    assert rq_mod.run(RUN_ID, [DATASET], None, staged, None, None, False) == 1
    assert fake.queries == []
    # 显式加 --allow-config-drift 才继续。
    assert rq_mod.run(RUN_ID, [DATASET], None, staged, None, None, True) == 0
    assert len(fake.queries) == 2
