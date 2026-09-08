"""轮次 3、4 跑在 mock 的 Akasha 上，覆盖请求形状与断点续跑。

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

from akasha_benchmark.akasha_client import AkashaClient, AkashaError
from akasha_benchmark.config import AkashaConfig
from akasha_benchmark.io_utils import atomic_write_json, atomic_write_jsonl, read_jsonl
from akasha_benchmark import ingest as ingest_mod
from akasha_benchmark import run_queries as rq_mod

DATASET = "hotpotqa"
RUN_ID = "r1"


class FakeAkasha:
    """轮次 3、4 用到的那几个端点的最小替身。

    ``role`` 和 ``quality_clean`` 用来构造两种失败场景，
    ``fail_import_for`` / ``fail_query_for`` 用来指定哪些条目要失败。
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
        self._next_page = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append((request.method, path))

        if path == "/api/auth/login":
            return httpx.Response(200, json=None, headers={"set-cookie": "authToken=jwt; Path=/"})
        if path == "/api/users/me":
            return httpx.Response(
                200,
                json={
                    "user": {"id": "u1", "role": self.role},
                    "workspace": {"id": "w1", "name": "bench"},
                },
            )
        if path == "/api/llm-wiki/admin/model-configs":
            return httpx.Response(200, json={"embedding": {"model": "m1"}})
        if path == "/api/spaces":
            return httpx.Response(200, json={"items": list(self.spaces.values()), "meta": {}})
        if path == "/api/spaces/create":
            body = json.loads(request.content)
            space = {"id": f"space-{body['slug']}", "slug": body["slug"], "name": body["name"]}
            self.spaces[body["slug"]] = space
            return httpx.Response(200, json=space)
        if path == "/api/pages/import":
            raw = request.content.decode("utf-8", "replace")
            # 文件名承担 doc_id，heading 承担 title，两者互不干扰。
            name = raw.split('filename="', 1)[1].split('"', 1)[0]
            doc_id = name.removesuffix(".md")
            if doc_id in self.fail_import_for:
                return httpx.Response(500, json={"message": "boom"})
            self._next_page += 1
            page = {"id": f"page-{doc_id}", "title": f"T{doc_id}"}
            self.imported.append({"doc_id": doc_id, "raw_has_space_id": "spaceId" in raw})
            return httpx.Response(200, json=page)
        if path == "/api/llm-wiki/admin/compile-spaces":
            return httpx.Response(
                200, json={"requestedSpaceCount": 1, "acceptedRunCount": 1, "coalescedRunCount": 0}
            )
        if path == "/api/llm-wiki/admin/diagnostics/summary":
            return httpx.Response(200, json={"statusCounts": {"succeeded": 1}})
        if path == "/api/llm-wiki/admin/diagnostics/quality":
            counts = 0 if self.quality_clean else 3
            return httpx.Response(
                200,
                json={
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
                },
            )
        if path == "/api/llm-wiki/query":
            body = json.loads(request.content)
            self.queries.append(body)
            if body["query"] in self.fail_query_for:
                return httpx.Response(503, json={"message": "unavailable"})
            return httpx.Response(
                200,
                json={
                    "answer": "an answer",
                    "answerMode": "knowledge",
                    "citations": [],
                    "citationEvidence": [],
                    "retrievedSources": [{"sourcePageId": "page-0"}],
                    "snippets": [],
                    "warnings": [],
                    "budget": {},
                    "completenessNotice": None,
                },
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


def test_login_sets_cookie_and_import_sends_multipart(tmp_path: Path, config: AkashaConfig):
    """导入必须以 multipart 发出，且带上 spaceId 字段。"""
    fake = FakeAkasha()
    md = tmp_path / "42.md"
    md.write_text("# Title\n\nBody\n", encoding="utf-8")

    with client_for(fake, config) as client:
        client.login()
        page = client.import_page(md, "space-1")

    assert page["id"] == "page-42"
    assert fake.imported[0] == {"doc_id": "42", "raw_has_space_id": True}


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


def _write_subset(data: Path, doc_ids: list[str], questions: list[tuple[str, str]]) -> None:
    """伪造一份轮次 2 的产出，含 md 文件与其 sha256。"""
    subset = data / "subsets" / RUN_ID / DATASET
    corpus = subset / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    hashes = {}
    from akasha_benchmark.io_utils import sha256_text

    for doc_id in doc_ids:
        markdown = f"# T{doc_id}\n\nbody {doc_id}\n"
        (corpus / f"{doc_id}.md").write_text(markdown, encoding="utf-8", newline="\n")
        hashes[doc_id] = sha256_text(markdown)

    atomic_write_jsonl(
        subset / "samples.jsonl",
        [
            {
                "dataset": DATASET,
                "sample_id": f"{DATASET}:{sid}",
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
    """准备好轮次 3 需要的输入。"""
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
    """质量报告非 0 时不能放行到轮次 4。"""
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
    """md 与轮次 2 记录的 sha256 不符时终止，否则入库内容无从追溯。"""
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


def _prepare_round4(data: Path) -> None:
    """伪造轮次 3 的 manifest，让轮次 4 能读到 space id 与模型配置。"""
    atomic_write_json(
        ingest_mod.ingest_dir(RUN_ID, data) / "manifest.json",
        {
            "spaces": {DATASET: {"id": "space-1"}},
            "model_configs": {"embedding": {"model": "m1"}},
        },
    )


def test_run_queries_writes_one_row_per_sample(
    staged: Path, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """验收标准：每个 sample_id 恰好一行，请求体形状固定。"""
    _prepare_round4(staged)
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
    _prepare_round4(staged)
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


def test_run_queries_resumes_from_existing_rows(
    staged: Path, config: AkashaConfig, monkeypatch: pytest.MonkeyPatch
):
    """断点续跑：已完成的 sample_id 不再请求。"""
    _prepare_round4(staged)
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
    """模型配置与轮次 3 不一致时终止，除非显式允许。"""
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
