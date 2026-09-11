"""入库与查询跑在 mock 的 Akasha 上，覆盖请求形状与断点续跑。

真正跑一遍仍然需要在线的 Akasha；这里锁住的是那些容易悄悄搞错的地方 ——
multipart 的字段名、OWNER 闸门、质量闸门、续跑时的跳过逻辑，
以及失败必须落盘而不是被丢掉。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from akasha_benchmark import akasha_client as client_mod
from akasha_benchmark.akasha_client import AkashaClient, AkashaError, unwrap_envelope
from akasha_benchmark.config import AkashaConfig
from akasha_benchmark.store import connect, repo
from akasha_benchmark.store.migrate import migrate
from akasha_benchmark import ingest as ingest_mod
from akasha_benchmark import run_queries as rq_mod

DATASET = "hotpotqa"
LABEL = "r1"
QUERY_LABEL = "r1-query"


# 与 _prepare_query_stage 写进假 manifest 的值共用一处：漂移闸门是整体相等判断，
# 两边一旦不同步，每个查询测试都会以「配置漂移」失败。
# 四项都给：漂移判定要能区分 embedding（拒绝执行）与 answer（仅不可比），
# 只声明 embedding 的话后者根本测不出来。
MOCK_MODEL_CONFIGS = {
    "configs": [
        {"feature": "embedding", "model": "m1", "baseUrl": "http://x/v1", "apiKeySet": True},
        {"feature": "compiler", "model": "c1", "baseUrl": "http://x/v1", "apiKeySet": True},
        {"feature": "answer", "model": "a1", "baseUrl": "http://x/v1", "apiKeySet": True},
        {"feature": "image", "model": "i1", "baseUrl": "http://x/v1", "apiKeySet": True},
    ]
}


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
        # 服务端解析出的 workspace。改它模拟「换了账号/换了部署」——
        # 客户端选不了这个值，所以闸门的判据只能是它。
        self.workspace_id = "w1"
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
                    "workspace": {"id": self.workspace_id, "name": "bench"},
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


def test_query_records_non_2xx_without_raising(config: AkashaConfig, no_sleep: None):
    """查询的非 2xx 不抛异常，交给调用方落盘。

    替身返回的 503 属于可重试状态，所以这里走完整的重试再返回 —— 断言的是
    「重试用尽后仍是 503 且不抛」。用 ``no_sleep`` 跳过退避，否则要真等 5 次。
    """
    fake = FakeAkasha()
    fake.fail_query_for.add("bad question")
    with client_for(fake, config) as client:
        client.login()
        response = client.query("bad question", ["space-1"])
    assert response.status == 503
    assert response.body["message"] == "unavailable"


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """退避的 sleep 换成空操作，测试不必真等。"""
    monkeypatch.setattr(client_mod.time, "sleep", lambda _seconds: None)


def _counting_handler(
    statuses: list[int], payload: Any = None
) -> tuple[Any, list[httpx.Request]]:
    """按 ``statuses`` 依次返回状态码，用尽后一律 200。同时记下每次请求。"""
    seen: list[httpx.Request] = []
    remaining = list(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if remaining:
            status = remaining.pop(0)
            return httpx.Response(status, text="")
        return enveloped(payload if payload is not None else {"ok": True})

    return handler, seen


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_transient_status_is_retried_then_succeeds(
    config: AkashaConfig, no_sleep: None, status: int
):
    """瞬时 5xx/429 要重试而不是让整个阶段退出。

    这条是拿一次真实事故换回来的：编译 400 页要轮询上千次，dev server
    （``nest start --watch``）偶发重启会返回 502，客户端不重试就让 ingest
    进程直接退出 —— 而服务端的编译还在 BullMQ 里继续跑，于是没人接管进度、
    manifest 也写不出来。
    """
    handler, seen = _counting_handler([status])
    client = AkashaClient(config)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    with client:
        body = client.get("llm-wiki/admin/model-configs")
    assert body == {"ok": True}
    assert len(seen) == 2  # 一次失败 + 一次成功


def test_transport_error_is_retried_then_succeeds(config: AkashaConfig, no_sleep: None):
    """连接层异常（服务重启时的 connect reset）同样要重试。"""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return enveloped({"ok": True})

    client = AkashaClient(config)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    with client:
        assert client.get("llm-wiki/admin/model-configs") == {"ok": True}
    assert attempts["n"] == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_client_errors_are_not_retried(config: AkashaConfig, no_sleep: None, status: int):
    """4xx 是请求本身的问题，重试只会放大 —— 必须一次就抛。"""
    handler, seen = _counting_handler([status])
    client = AkashaClient(config)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    with client, pytest.raises(AkashaError) as excinfo:
        client.get("llm-wiki/admin/model-configs")
    assert excinfo.value.status == status
    assert len(seen) == 1


def test_retries_are_bounded_and_then_raise(config: AkashaConfig, no_sleep: None):
    """一直 502 时不能无限重试，用尽次数后照常抛，由调用方决定怎么处理。"""
    handler, seen = _counting_handler([502] * 50)
    client = AkashaClient(config)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    with client, pytest.raises(AkashaError) as excinfo:
        client.get("llm-wiki/admin/model-configs")
    assert excinfo.value.status == 502
    assert len(seen) == client_mod.MAX_RETRIES + 1


def test_multipart_retry_resends_the_whole_file(
    tmp_path: Path, config: AkashaConfig, no_sleep: None
):
    """重试 multipart 前必须把文件句柄拨回开头。

    不 rewind 的话第一次尝试已经把句柄读到末尾，重发的 body 是空的 ——
    服务端会收下一个空文件并返回 200，于是「导入成功」但内容为空，
    这种缺陷不报错，只会让后续召回莫名其妙地找不到东西。

    导入本身是 ``retry=False``（见下一条），所以这里直接调 ``request``
    把重试打开，锁住 rewind 这个通用行为。
    """
    md = tmp_path / "42.md"
    md.write_text("# Title\n\nBody line\n", encoding="utf-8")
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        if len(bodies) == 1:
            return httpx.Response(502, text="")
        return enveloped({"id": "page-1", "title": "Title"})

    client = AkashaClient(config)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    with client, md.open("rb") as handle:
        response = client.request(
            "POST",
            "pages/import",
            files={"file": (md.name, handle, "text/markdown")},
            data={"spaceId": "space-1"},
            retry=True,
        )

    assert response.body["id"] == "page-1"
    assert len(bodies) == 2
    # 两次都要带上完整正文，第二次不能是空 body。
    for body in bodies:
        assert b"Body line" in body
    assert b'name="spaceId"' in bodies[1]


def test_import_page_does_not_retry_ambiguous_failures(
    tmp_path: Path, config: AkashaConfig, no_sleep: None
):
    """导入遇到 5xx 不重试 —— 服务端可能已建好 page，重试会建出第二个。

    重复 page 不在 ``page_map`` 里，续跑发现不了，只会悄悄抬高语料规模。
    缺篇相反是可发现、可续跑补齐的，所以这里宁可失败也不重试。
    """
    md = tmp_path / "42.md"
    md.write_text("# Title\n\nBody\n", encoding="utf-8")
    handler, seen = _counting_handler([502] * 10)

    client = AkashaClient(config)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    with client, pytest.raises(AkashaError) as excinfo:
        client.import_page(md, "space-1")

    assert excinfo.value.status == 502
    assert len(seen) == 1


def test_create_space_does_not_retry(config: AkashaConfig, no_sleep: None):
    """建 Space 同样是写入，重试可能建出第二个，所以一次就抛。"""
    handler, seen = _counting_handler([502] * 10)
    client = AkashaClient(config)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    with client, pytest.raises(AkashaError):
        client.create_space(name="n", slug="s1")
    assert len(seen) == 1


def test_polling_endpoints_do_retry(config: AkashaConfig, no_sleep: None):
    """只读的诊断端点必须重试 —— 编译 400 页要轮询上千次，一次 502 不能掀桌。"""
    payload = {"statusCounts": {"compiling": 1}}
    handler, seen = _counting_handler([502, 503], payload)
    client = AkashaClient(config)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    with client:
        assert client.run_diagnostics_summary(["space-1"]) == payload
    assert len(seen) == 3


def test_retry_delay_is_bounded_and_jittered(config: AkashaConfig):
    """退避有上限且带抖动：上限防止越等越久，抖动避免并发请求齐步重试。"""
    client = AkashaClient(config)
    with client:
        delays = [client._retry_delay(attempt) for attempt in range(10)]
    assert all(0 < d <= client_mod.RETRY_MAX_DELAY * 1.25 for d in delays)
    # 前几次应当随尝试次数增长（取抖动下界比较，避免偶发翻转）。
    assert client._retry_delay(0) < client_mod.RETRY_MAX_DELAY
    varied = {round(client._retry_delay(3), 6) for _ in range(20)}
    assert len(varied) > 1


def _seed_subset(
    connection,
    doc_ids: list[str],
    questions: list[tuple[str, str]],
    dataset: str = DATASET,
    layer_id: int | None = None,
) -> int:
    """伪造一份子集**写进库**，含 md 正文与其 sha256。返回索引层 id。"""
    from akasha_benchmark.io_utils import sha256_text

    repo.upsert_dataset(
        connection,
        name=dataset,
        adapter="A",
        adapter_version="1",
        provides=["gold_docs", "reference_answers"],
        identity_rules={},
        qa_path="q",
        qa_sha256="0" * 64,
        qa_rows=len(questions),
        corpus_path="c",
        corpus_sha256="1" * 64,
        corpus_rows=len(doc_ids),
        dedup_stats={},
        gold_count_distribution={},
        unique_question_texts=len({q for _, q in questions}),
    )
    repo.replace_samples(
        connection,
        dataset,
        [
            {
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
    if layer_id is None:
        layer_id = repo.create_index_layer(
            connection,
            label=LABEL,
            subset_hash="sh",
            seed=1,
            qa_limit=len(questions),
            negatives_ratio=1.0,
            narrativeqa_docs=2,
        )
    repo.upsert_index_layer_dataset(
        connection,
        layer_id,
        dataset,
        strategy="uniform_qa_then_gold_corpus",
        qa_count=len(questions),
        corpus_count=len(doc_ids),
        gold_doc_count=1,
        negative_doc_count=len(doc_ids) - 1,
        strata={},
        normalized_qa_sha256="0" * 64,
        normalized_corpus_sha256="1" * 64,
    )
    docs = []
    for doc_id in doc_ids:
        # 正文带上数据集名：跨组撞 doc_id 时两篇内容不同，sha256 也不同，
        # 这样「导错了组」不会碰巧通过哈希校验。
        markdown = f"# T{doc_id}\n\nbody {doc_id} of {dataset}\n"
        docs.append(
            {
                "doc_id": doc_id,
                "md_text": markdown,
                "md_sha256": sha256_text(markdown),
                "is_gold": doc_id == doc_ids[0],
            }
        )
    repo.replace_subset(
        connection, layer_id, dataset, [f"{dataset}:{sid}" for sid, _ in questions], docs
    )
    connection.commit()
    return layer_id


class Staged:
    """库连接 + 路径 + 索引层 id。"""

    def __init__(self, connection, db_path: Path, layer_id: int) -> None:
        self.connection = connection
        self.db_path = db_path
        self.layer_id = layer_id


@pytest.fixture
def staged(tmp_path: Path):
    """准备好入库需要的输入。"""
    path = tmp_path / "t.db"
    migrate(path, verbose=False)
    connection = connect(path)
    layer_id = _seed_subset(
        connection, ["0", "1", "2"], [("s1", "first question"), ("s2", "second question")]
    )
    yield Staged(connection, path, layer_id)
    connection.close()


def _patch_client(monkeypatch: pytest.MonkeyPatch, module, fake: FakeAkasha, config: AkashaConfig):
    """把目标模块里的配置加载与客户端都替换掉。"""
    # 配置从库里读。测试直接把它替掉，不去写 connection 表 —— 那样测的就是
    # 「配置能不能存」而不是「阶段行为对不对」。配置读写另有 test_config_store.py。
    monkeypatch.setattr(module, "load_config", lambda _connection: config)

    class Patched(AkashaClient):
        def __init__(self, cfg: AkashaConfig) -> None:
            super().__init__(cfg)
            self._client = httpx.Client(transport=httpx.MockTransport(fake.handler))

    monkeypatch.setattr(module, "AkashaClient", Patched)


def test_ingest_maps_every_doc_and_passes_quality(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """正常路径：每篇都进 page_map，质量闸门通过。"""
    fake = FakeAkasha()
    _patch_client(monkeypatch, ingest_mod, fake, config)

    assert ingest_mod.run(LABEL, [DATASET], staged.db_path) == 0

    connection = staged.connection
    assert set(repo.page_to_doc(connection, staged.layer_id, DATASET).values()) == {"0", "1", "2"}
    assert repo.page_map_counts(connection, staged.layer_id) == {DATASET: 3}
    gate = repo.latest_quality_gate(connection, staged.layer_id)
    assert gate["passed"] == 1
    layer = repo.get_index_layer(connection, staged.layer_id)
    assert layer["quality_passed"] == 1
    # 索引层的身份到入库才凑齐：config_hash 之前是 NULL。
    assert layer["config_hash"]
    # 密钥绝不能落库。
    assert repo.loads(layer["connection_json"])["password"] == "***"
    # 这一层现在可以开始跑查询了。
    assert repo.index_layer_readiness(connection, staged.layer_id)["ready"] is True


def test_ingest_refuses_a_non_owner_user(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """非 OWNER 必须在导入前就被拦住，一篇都不能导。"""
    fake = FakeAkasha(role="member")
    _patch_client(monkeypatch, ingest_mod, fake, config)
    # 走 main()，因为把异常转成退出码是在那一层做的。
    exit_code = ingest_mod.main(
        ["--label", LABEL, "--dataset", DATASET, "--db", str(staged.db_path)]
    )
    assert exit_code == 1
    assert not fake.imported


def test_ingest_fails_the_quality_gate(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """质量报告非 0 时不能放行到查询。"""
    fake = FakeAkasha(quality_clean=False)
    _patch_client(monkeypatch, ingest_mod, fake, config)
    assert ingest_mod.run(LABEL, [DATASET], staged.db_path) == 1
    # 而且这一层必须被判为「不能跑查询」。
    readiness = repo.index_layer_readiness(staged.connection, staged.layer_id)
    assert readiness["ready"] is False
    assert any("quality gate failed" in r for r in readiness["reasons"])


def test_skip_compile_is_not_an_accepted_ingest(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """``--skip-compile`` 是调试入口，**必须非零退出**。

    §10.1 记的缺口：它原来可以返回成功，于是「跳过编译」看起来像「验收通过」，
    而那之后跑出来的指标全部偏低且不报错。
    """
    fake = FakeAkasha()
    _patch_client(monkeypatch, ingest_mod, fake, config)
    assert ingest_mod.run(LABEL, [DATASET], staged.db_path, skip_compile=True) == 1
    # 质量闸门没跑过，所以 quality_passed 必须是 NULL 而不是 0 或 1 ——
    # 「没跑过」与「跑了且失败」是两回事。
    assert repo.get_index_layer(staged.connection, staged.layer_id)["quality_passed"] is None
    assert repo.latest_quality_gate(staged.connection, staged.layer_id) is None
    readiness = repo.index_layer_readiness(staged.connection, staged.layer_id)
    assert readiness["ready"] is False
    assert any("never run" in r for r in readiness["reasons"])


def test_ingest_resumes_and_skips_imported_docs(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """断点续跑：第二次运行不重复导入。"""
    fake = FakeAkasha()
    _patch_client(monkeypatch, ingest_mod, fake, config)
    ingest_mod.run(LABEL, [DATASET], staged.db_path)
    assert len(fake.imported) == 3

    # 新客户端，但**服务端那个 Space 还在** —— 续跑时 ingest 会校验库里记的
    # space_id 仍然解析得到（slug 相同不等于同一个 space），所以替身要照实
    # 保留它。不保留就等于模拟了「换了账号」，那是另一个用例。
    second = FakeAkasha()
    second.spaces = fake.spaces
    _patch_client(monkeypatch, ingest_mod, second, config)
    ingest_mod.run(LABEL, [DATASET], staged.db_path)
    # 已经全部映射过，所以第二次一条都不该重新导入。
    assert second.imported == []
    assert repo.page_map_counts(staged.connection, staged.layer_id) == {DATASET: 3}


def test_ingest_refuses_when_the_server_resolves_another_workspace(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """入库前拿**登录后解析出的** workspace 比对，不符拒绝执行。

    不拦的话：``list_spaces`` 按 workspace 过滤，找不到同 slug 的 space,
    于是建一个新的并覆盖库里的 space_id —— 而 page_map 里的 page_id 还指向旧
    workspace 的页。之后查询照常跑完，每条都召回不到，看起来像
    「这批语料检索效果差」。

    判据刻意不来自配置：workspace 由服务端决定（自建部署走
    ``workspaceRepo.findFirst()``），客户端选不了，让用户填只会带来填错时的误报。
    """
    fake = FakeAkasha()
    _patch_client(monkeypatch, ingest_mod, fake, config)
    assert ingest_mod.run(LABEL, [DATASET], staged.db_path) == 0
    assert repo.get_index_layer(staged.connection, staged.layer_id)["workspace_id"] == "w1"

    # 同一个连接，但服务端现在解析到另一个 workspace（换了账号/换了部署）。
    moved = FakeAkasha()
    moved.spaces = fake.spaces
    moved.workspace_id = "w2"
    _patch_client(monkeypatch, ingest_mod, moved, config)
    assert ingest_mod.run(LABEL, [DATASET], staged.db_path) == 1

    # **一条都不许重导**：拦截必须发生在任何写入之前。
    assert moved.imported == []
    # 库里的 space 绑定与 page_map 没被动过。
    assert repo.spaces_of(staged.connection, staged.layer_id) == {DATASET: "space-benchhotpotqar1"}
    assert repo.page_map_counts(staged.connection, staged.layer_id) == {DATASET: 3}


def test_run_queries_refuses_when_the_server_resolves_another_workspace(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """查询同样要判：这一层的 space_id 只在它入库时那个 workspace 里解析得到。

    换个地方跑，每条 query 都会打到一个空 space —— 而那不报错，只会给出一份
    「recall 全 0」的报告。
    """
    fake = FakeAkasha()
    _patch_client(monkeypatch, ingest_mod, fake, config)
    assert ingest_mod.run(LABEL, [DATASET], staged.db_path) == 0

    moved = FakeAkasha()
    moved.spaces = fake.spaces
    moved.workspace_id = "w2"
    _patch_client(monkeypatch, rq_mod, moved, config)
    assert rq_mod.run(LABEL, [DATASET], staged.db_path, None, None, False) == 1
    # 一条 query 都不许发出去。
    assert moved.queries == []


OTHER_DATASET = "2wikimultihopqa"


def test_ingest_keeps_colliding_doc_ids_of_two_datasets_apart(
    tmp_path: Path, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """两组撞 doc_id 时，page_map 必须按 (层, 数据集, doc_id) 分别记全。

    doc_id 是各数据集内部的裸 ID，跨组会撞（锁定的 run001 子集里
    hotpotqa×2wiki 撞 15 个）。少了 dataset 这一维，后一组会覆盖前一组的行，
    前一组这些 doc 就被当成已导入而跳过。
    """
    path = tmp_path / "t.db"
    migrate(path, verbose=False)
    connection = connect(path)
    try:
        # "1" 和 "2" 两组都有，"9" 只有 hotpotqa，"7" 只有 2wiki。
        layer_id = _seed_subset(connection, ["1", "2", "9"], [("s1", "q1")], dataset=DATASET)
        _seed_subset(
            connection, ["1", "2", "7"], [("s2", "q2")], dataset=OTHER_DATASET, layer_id=layer_id
        )

        fake = FakeAkasha()
        _patch_client(monkeypatch, ingest_mod, fake, config)
        assert ingest_mod.run(LABEL, [DATASET, OTHER_DATASET], path) == 0

        rows = [
            dict(r)
            for r in connection.execute(
                "SELECT dataset, doc_id, page_id, space_id FROM page_map WHERE index_layer_id = ?",
                (layer_id,),
            )
        ]
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
        assert repo.page_map_counts(connection, layer_id) == {DATASET: 3, OTHER_DATASET: 3}

        # 续跑：两组都已导全，一条都不该重发。服务端的 Space 照实保留。
        second = FakeAkasha()
        second.spaces = fake.spaces
        _patch_client(monkeypatch, ingest_mod, second, config)
        assert ingest_mod.run(LABEL, [DATASET, OTHER_DATASET], path) == 0
        assert second.imported == []
    finally:
        connection.close()


def test_ingest_reports_import_failures(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """导入失败要计数并落库，不能静默跳过。"""
    fake = FakeAkasha()
    fake.fail_import_for.add("1")
    _patch_client(monkeypatch, ingest_mod, fake, config)
    assert ingest_mod.run(LABEL, [DATASET], staged.db_path) == 1

    failures = repo.import_failures(staged.connection, staged.layer_id)
    assert [f["doc_id"] for f in failures] == ["1"]
    # 缺篇必须让这一层进不了查询阶段。
    readiness = repo.index_layer_readiness(staged.connection, staged.layer_id)
    assert readiness["ready"] is False
    assert any("imported 2 of 3" in r for r in readiness["reasons"])


def test_ingest_detects_a_corrupt_subset_row(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """正文与它自己记录的 sha256 不符时终止，否则入库内容无从追溯。"""
    staged.connection.execute(
        "UPDATE subset_doc SET md_text = ? WHERE doc_id = ? AND index_layer_id = ?",
        ("# T1\n\ntampered\n", "1", staged.layer_id),
    )
    staged.connection.commit()

    fake = FakeAkasha()
    _patch_client(monkeypatch, ingest_mod, fake, config)
    exit_code = ingest_mod.main(
        ["--label", LABEL, "--dataset", DATASET, "--db", str(staged.db_path)]
    )
    assert exit_code == 1
    # 被改过的那篇绝不能进库。
    assert "1" not in {row["doc_id"] for row in fake.imported}


def test_ingest_detects_upstream_normalized_data_changing(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """§12.2 第 1 条：上游 sha256 链断了就报错。

    含义是「normalize 换过数据快照，而这一层的子集是照旧快照抽的」。带着这种
    状态导入，page_map 记的身份与库里的语料对不上，之后每个指标都失去可追溯性 ——
    而这种失效不会报错，只会给出一份看着正常的报告。
    """
    staged.connection.execute(
        "UPDATE dataset SET corpus_sha256 = ? WHERE name = ?", ("9" * 64, DATASET)
    )
    staged.connection.commit()

    fake = FakeAkasha()
    _patch_client(monkeypatch, ingest_mod, fake, config)
    exit_code = ingest_mod.main(
        ["--label", LABEL, "--dataset", DATASET, "--db", str(staged.db_path)]
    )
    assert exit_code == 1
    assert not fake.imported


def _prepare_query_stage(staged) -> None:
    """让索引层通过前置闸门：page_map 齐、质量四项全 0、编译已终态。

    这三项是查询阶段的**不可跳过**闸门（§10.1）。测试里必须显式把它们置成
    通过态，否则 run() 会（正确地）拒绝开跑。
    """
    connection = staged.connection
    repo.set_space(
        connection, staged.layer_id, DATASET, space_id="space-1", space_slug="sp", space_reused=False
    )
    for index, doc in enumerate(repo.subset_docs(connection, staged.layer_id, DATASET)):
        repo.record_page(
            connection,
            staged.layer_id,
            DATASET,
            doc_id=doc["doc_id"],
            page_id=f"page-{index}",
            space_id="space-1",
            title=None,
            md_sha256=doc["md_sha256"],
        )
    repo.record_quality_gate(
        connection,
        staged.layer_id,
        gates={
            "missingChunkPageCount": 0,
            "missingEmbeddingPageCount": 0,
            "missingSourcePageCount": 0,
            "stalePageCount": 0,
        },
        report={},
    )
    repo.record_compile_run(
        connection,
        staged.layer_id,
        accepted_run_count=1,
        coalesced_run_count=0,
        status_counts={"succeeded": 1},
        terminal={"succeeded": 1},
        timed_out=False,
        requested_at="2026-09-08T00:00:00Z",
        finished_at="2026-09-08T00:10:00Z",
    )
    repo.update_index_layer(
        connection,
        staged.layer_id,
        quality_passed=1,
        model_configs_json=repo.dumps(MOCK_MODEL_CONFIGS),
    )
    connection.commit()


def _query(staged, *, drift_ok: bool = False, retry_failed: bool = False) -> int:
    return rq_mod.run(
        LABEL, [DATASET], staged.db_path, None, None, drift_ok, retry_failed=retry_failed
    )


def _responses(staged) -> list[dict]:
    query_layer = repo.query_layer_by_label(staged.connection, QUERY_LABEL)
    return repo.responses_of(staged.connection, int(query_layer["id"]), DATASET)


def test_run_queries_refuses_a_layer_that_failed_the_quality_gate(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """前置闸门：质量没过就一条 query 都不准发（§10.1）。

    这一道以前只在 Makefile 里，Python 侧没有 —— 于是从平台或直接调 run()
    都能绕过去，拿半成品索引跑出一份看着像「配置差」的报告。
    """
    _prepare_query_stage(staged)
    repo.update_index_layer(staged.connection, staged.layer_id, quality_passed=0)
    staged.connection.execute(
        "UPDATE quality_gate SET passed = 0, missing_chunk_page_count = 4 WHERE index_layer_id = ?",
        (staged.layer_id,),
    )
    staged.connection.commit()

    fake = FakeAkasha()
    _patch_client(monkeypatch, rq_mod, fake, config)
    assert _query(staged) == 1
    assert fake.queries == []


def test_run_queries_refuses_a_layer_whose_compile_timed_out(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """编译超时意味着还有页在排队，chunk 没生成 —— 指标会偏低且不报错。"""
    _prepare_query_stage(staged)
    staged.connection.execute(
        "UPDATE compile_run SET timed_out = 1 WHERE index_layer_id = ?", (staged.layer_id,)
    )
    staged.connection.commit()

    fake = FakeAkasha()
    _patch_client(monkeypatch, rq_mod, fake, config)
    assert _query(staged) == 1
    assert fake.queries == []


def test_run_queries_writes_one_row_per_sample(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """每个 sample_id 恰好一行，且响应体完整落库。"""
    _prepare_query_stage(staged)
    fake = FakeAkasha()
    _patch_client(monkeypatch, rq_mod, fake, config)

    assert _query(staged) == 0

    rows = _responses(staged)
    assert {r["sample_id"] for r in rows} == {f"{DATASET}:s1", f"{DATASET}:s2"}
    assert all(r["http_status"] == 200 for r in rows)
    # 存的是**完整响应体**，不是当下用得到的那几个字段（§7.2）。
    assert all(r["response"]["answerMode"] == "knowledge" for r in rows)
    # answerMode 抽成列，UI 的默认切分靠它，不必每次解析 JSON。
    assert all(r["answer_mode"] == "knowledge" for r in rows)


def test_run_queries_honours_concurrency(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """并发 > 1 时每个 worker 各持一个客户端，所以登录次数等于并发度。

    共用一个实例的话限流器的 _last_request_at 会互相踩，request_interval
    退化成「一起睡、一起发」。行序变成完成顺序，因此按集合断言。
    """
    _prepare_query_stage(staged)
    config = replace(config, concurrency=3)
    fake = FakeAkasha()
    _patch_client(monkeypatch, rq_mod, fake, config)

    assert _query(staged) == 0

    rows = _responses(staged)
    assert {r["sample_id"] for r in rows} == {f"{DATASET}:s1", f"{DATASET}:s2"}
    assert all(r["http_status"] == 200 for r in rows)
    logins = [path for method, path in fake.requests if path == "/api/auth/login"]
    assert len(logins) == 3
    query_layer = repo.query_layer_by_label(staged.connection, QUERY_LABEL)
    assert query_layer["concurrency"] == 3


def test_run_queries_concurrent_resume_skips_completed_rows(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """并发路径同样要能续跑：已有行的 sample 不再重新请求。"""
    _prepare_query_stage(staged)
    config = replace(config, concurrency=3)
    fake = FakeAkasha()
    _patch_client(monkeypatch, rq_mod, fake, config)

    # 先只跑第一条，制造一个「跑了一半」的查询层。
    assert rq_mod.run(LABEL, [DATASET], staged.db_path, None, 1, False) == 0
    assert len(fake.queries) == 1

    second = FakeAkasha()
    _patch_client(monkeypatch, rq_mod, second, config)
    assert _query(staged) == 0
    assert [q["query"] for q in second.queries] == ["second question"]
    assert {r["sample_id"] for r in _responses(staged)} == {f"{DATASET}:s1", f"{DATASET}:s2"}


def test_run_queries_records_failures_as_rows(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch, no_sleep: None
):
    """失败也占一行。静默跳过失败会把后面所有均值算高。

    替身的 503 会先走完重试，``no_sleep`` 让这里不必真等退避。
    """
    _prepare_query_stage(staged)
    fake = FakeAkasha()
    fake.fail_query_for.add("second question")
    _patch_client(monkeypatch, rq_mod, fake, config)

    _query(staged)
    rows = _responses(staged)
    assert len(rows) == 2
    assert len([r for r in rows if r["http_status"] != 200]) == 1


def test_run_queries_records_transport_errors_instead_of_crashing(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch, no_sleep: None
):
    """断连/超时**重试用尽后**要落库成失败行，并且不能中断后面的样本。

    ``httpx.RequestError`` 不是 ``OSError`` 的子类，漏掉它的话跑到一半
    网络抖一下整个阶段就带 traceback 崩掉，那一行也不会落库。

    这里断言的错误串仍以 ``ConnectError:`` 开头 —— 客户端重试用尽后原样抛出
    传输层异常，不包成 ``AkashaError``，否则上面那条捕获通路就断了。
    """
    _prepare_query_stage(staged)
    fake = FakeAkasha()
    fake.raise_transport_for.add("first question")
    _patch_client(monkeypatch, rq_mod, fake, config)

    assert _query(staged) == 0

    rows = {r["sample_id"]: r for r in _responses(staged)}
    # 崩掉的那条照样占一行，后一条继续跑完。
    assert set(rows) == {f"{DATASET}:s1", f"{DATASET}:s2"}
    failed, ok = rows[f"{DATASET}:s1"], rows[f"{DATASET}:s2"]
    assert failed["http_status"] == 0
    assert failed["response"] is None
    assert failed["error"].startswith("ConnectError:")
    assert ok["http_status"] == 200


def test_retrying_failed_rows_needs_an_explicit_flag(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch, no_sleep: None
):
    """§10.1 的失败行恢复策略：默认跳过失败行，``--retry-failed`` 才重试。

    默认跳过是对的（重跑要烧 LLM 调用），但原实现没有任何重试入口，
    而追加 JSONL 又会造出重复 sample_id。现在主键拦住重复，重试走显式删除。
    """
    _prepare_query_stage(staged)
    fake = FakeAkasha()
    fake.fail_query_for.add("second question")
    _patch_client(monkeypatch, rq_mod, fake, config)
    _query(staged)
    assert len([r for r in _responses(staged) if r["http_status"] != 200]) == 1

    # 不带 flag 续跑：失败行被跳过，一条都不重发。
    second = FakeAkasha()
    _patch_client(monkeypatch, rq_mod, second, config)
    assert _query(staged) == 0
    assert second.queries == []

    # 带上 flag：失败行被删掉重试，这次替身不再失败。
    third = FakeAkasha()
    _patch_client(monkeypatch, rq_mod, third, config)
    assert _query(staged, retry_failed=True) == 0
    assert [q["query"] for q in third.queries] == ["second question"]
    rows = _responses(staged)
    assert len(rows) == 2
    assert all(r["http_status"] == 200 for r in rows)


def test_run_queries_resumes_from_existing_rows(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """断点续跑：已完成的 sample_id 不再请求。"""
    _prepare_query_stage(staged)
    fake = FakeAkasha()
    _patch_client(monkeypatch, rq_mod, fake, config)
    _query(staged)
    assert len(fake.queries) == 2

    second = FakeAkasha()
    _patch_client(monkeypatch, rq_mod, second, config)
    _query(staged)
    assert second.queries == []


def test_run_queries_stops_on_model_config_drift(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """模型配置与入库时不一致时终止，除非显式允许。

    这里漂的是 answer 模型：它改了不必重编译，所以属于「不可比」而非
    「静默失效」，可以用 --allow-config-drift 覆盖。
    """
    _prepare_query_stage(staged)
    drifted = {
        "configs": [
            {**c, "model": "OLD"} if c["feature"] == "answer" else c
            for c in MOCK_MODEL_CONFIGS["configs"]
        ]
    }
    repo.update_index_layer(
        staged.connection, staged.layer_id, model_configs_json=repo.dumps(drifted)
    )
    staged.connection.commit()

    fake = FakeAkasha()
    _patch_client(monkeypatch, rq_mod, fake, config)

    assert _query(staged) == 1
    assert fake.queries == []
    # 显式加 --allow-config-drift 才继续。
    assert _query(staged, drift_ok=True) == 0
    assert len(fake.queries) == 2


def test_run_queries_never_overrides_an_embedding_change(
    staged, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """embedding 漂移**拒绝执行**，--allow-config-drift 也不放行（§12.3）。

    换 embedding 后旧 chunk 的 ``embedding_profile`` 对不上，那些 chunk 永远
    召回不到，而评测会照常算出一份「recall 低、拒答率高」的报告 —— 看起来像
    配置差，实际是索引与 embedding 错配。这是唯一会静默失效的那一项。
    """
    _prepare_query_stage(staged)
    drifted = {
        "configs": [
            {**c, "model": "other-embedding"} if c["feature"] == "embedding" else c
            for c in MOCK_MODEL_CONFIGS["configs"]
        ]
    }
    repo.update_index_layer(
        staged.connection, staged.layer_id, model_configs_json=repo.dumps(drifted)
    )
    staged.connection.commit()

    fake = FakeAkasha()
    _patch_client(monkeypatch, rq_mod, fake, config)

    assert _query(staged) == 1
    # 即便显式允许漂移，这一项也必须拦住。
    assert _query(staged, drift_ok=True) == 1
    assert fake.queries == []
