from __future__ import annotations

import base64
import json
import threading
import time
from typing import Any

import pytest

from akasha_benchmark import model_configs
from akasha_benchmark import akasha_client
from akasha_benchmark.akasha_client import AkashaClient, unwrap_envelope
from akasha_benchmark.config import AkashaConfig
from akasha_benchmark.stages import compile, query
from akasha_benchmark.store import (
    compile_store,
    config_store,
    dumps,
    loads,
    query_store,
    run_store,
    task_store,
)
from akasha_benchmark.task import TaskContext, execute
from conftest import make_compile_run


def compile_with_pages(connection, run_id: str, page_ids: list[str]) -> tuple[int, int]:
    compile_id = make_compile_run(connection, run_id, ["hotpotqa"], qa_limit=1)
    compile.build_subset(
        connection, compile_id, "hotpotqa", seed=1, qa_limit=1, negatives_ratio=1.0
    )
    docs = compile_store.compile_docs(connection, compile_id)[: len(page_ids)]
    for doc, page_id in zip(docs, page_ids):
        compile_store.record_page(
            connection, compile_id, doc["dataset"], doc["doc_id"], page_id=page_id, error=None
        )
    task_id = bind_compile_task(connection, compile_id)
    return compile_id, task_id


def bind_compile_task(connection, compile_id: int, params: dict[str, Any] | None = None) -> int:
    task_id = task_store.create_task(connection, stage="compile", params=params or {})
    task_store.set_task_target(connection, task_id, "compile", compile_id)
    connection.commit()
    return task_id


@pytest.fixture
def client():
    connection = AkashaClient(AkashaConfig())
    yield connection
    connection.close()


def pages_client(items_by_run: dict[str, list[dict[str, Any]]], totals=None) -> FakeClient:
    """按 runId 返回预设 run_pages 分页。"""

    class Pages(FakeClient):
        def run_pages(self, run_id, *, page=1, limit=100):
            items = items_by_run.get(run_id, [])
            total = (totals or {}).get(run_id, len(items))
            return {"items": items, "total": total, "limit": limit}

    return Pages(None)


def context(connection, params: dict[str, Any], task_id=None) -> TaskContext:
    if task_id is None:
        task_id = task_store.create_task(connection, stage="test", params=params)
    connection.commit()
    return TaskContext(
        task_id=task_id,
        stage="test",
        params=params,
        connection=connection,
        pause_event=threading.Event(),
    )


CONFIGS = {
    "configs": [
        {"feature": "compiler", "provider": "p", "model": "c1", "baseUrl": "u", "parameters": {}},
        {"feature": "embedding", "provider": "p", "model": "e1", "baseUrl": "u", "parameters": {}},
        {"feature": "answer", "provider": "p", "model": "a1", "baseUrl": "u", "parameters": {}},
        {"feature": "image", "provider": "p", "model": "i1", "baseUrl": "u", "parameters": {}},
    ]
}


class FakeClient:
    def __init__(
        self,
        config,
        *,
        role: str = "owner",
        configs: Any = None,
        retrieved: list[str] | None = None,
        accepted_runs: int = 1,
        page_log_items: list[dict[str, Any]] | None = None,
        run_items: list[dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.role = role
        self.accepted_runs = accepted_runs
        self.compile_submitted = False
        self.page_log_items: list[dict[str, Any]] = page_log_items or []
        self.run_items: list[dict[str, Any]] = (
            run_items
            if run_items is not None
            else [
                {
                    "runId": "remote-run-1",
                    "status": "succeeded",
                    "runDurationMs": 8000,
                    "progress": {"text": {"expected": 4, "succeeded": 4, "failed": 0}},
                }
            ]
        )
        self.configs = configs if configs is not None else CONFIGS
        self.retrieved = retrieved or []
        self.imported: list[str] = []
        self.quality: dict[str, Any] = {
            "summary": {
                "missingChunkPageCount": 0,
                "missingEmbeddingPageCount": 0,
                "missingSourcePageCount": 0,
                "stalePageCount": 0,
            }
        }
        self.queries: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def close(self):
        pass

    def login(self):
        pass

    def current_user(self):
        return {"user": {"id": "u", "role": self.role}, "workspace": {"id": "w", "name": "W"}}

    def get_model_configs(self):
        return self.configs

    def create_space(self, name, slug, description=""):
        return {"id": "space-1", "slug": slug}

    def import_page_text(self, filename, markdown, space_id):
        self.imported.append(filename)
        return {"id": f"page-{len(self.imported)}"}

    def compile_spaces(self, space_ids):
        self.compile_submitted = True
        return {
            "acceptedRunCount": self.accepted_runs,
            "coalescedRunCount": 0,
            "runs": ([{"runId": "remote-run-1", "disposition": "created"}] if self.accepted_runs else []),
        }

    def run_diagnostics(self, space_ids, *, limit=50):
        return {"items": list(self.run_items) if self.compile_submitted else []}

    def quality_diagnostics(self, space_ids):
        return self.quality

    def page_log(self, space_ids, *, limit=100):
        return {"items": list(self.page_log_items)}

    def run_pages(self, run_id, *, page=1, limit=100):
        return {"items": [], "total": 0, "page": page, "limit": limit}

    def retryable_run_page_ids(self, run_ids):
        failed: dict[str, None] = {}
        for run_id in run_ids:
            page = 1
            while True:
                result = self.run_pages(run_id, page=page, limit=100)
                for item in result["items"]:
                    if item.get("status") == "failed" or item.get("mergeStatus") == "failed":
                        failed.setdefault(item["sourcePageId"], None)
                if page * int(result.get("limit") or 100) >= int(result.get("total") or 0):
                    break
                page += 1
        return list(failed)

    def retry_pages(self, page_ids):
        return {"queuedPageCount": len(page_ids), "jobIds": ["retry-run-1"]}

    def query(self, question, space_ids):
        from akasha_benchmark.akasha_client import Response

        self.queries.append(question)
        return Response(
            status=200,
            body={
                "answerMode": "knowledge",
                "answer": "a",
                "retrievedSources": [{"sourcePageId": page} for page in self.retrieved],
                "citations": [{"sourcePageId": page} for page in self.retrieved],
                "citationEvidence": [{"excerpts": ["x"]} for _ in self.retrieved],
                "snippets": [],
            },
            latency_ms=10,
        )


@pytest.fixture
def ready_connection(normalized):
    config_store.update_connection(normalized, base_url="http://x", email="e@x", password="p")
    normalized.commit()
    return normalized


def test_envelope_is_unwrapped_only_when_it_is_an_envelope():
    assert unwrap_envelope({"data": {"id": 1}, "success": True, "status": 200}) == {"id": 1}
    assert unwrap_envelope({"success": True, "status": 201}) is None
    payload = {"data": 1, "success": True, "status": 200, "extra": "x"}
    assert unwrap_envelope(payload) == payload
    assert unwrap_envelope([1, 2]) == [1, 2]


def _jwt(expiry: float) -> str:
    def encode(value: dict[str, Any]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode({'exp': expiry})}.signature"


def fake_http(token: str):
    """返回（假 httpx 客户端类, 登录计数）：登录发 cookie，其余请求首次 401 触发重登。"""
    calls = {"login": 0}
    lock = threading.Lock()

    class FakeHTTP:
        def __init__(self):
            self.cookies = akasha_client.httpx.Cookies()
            self.request_calls = 0

        def request(self, method, url, **kwargs):
            if url.endswith("/auth/login"):
                with lock:
                    calls["login"] += 1
                self.cookies.set(akasha_client.AUTH_TOKEN_COOKIE, token)
                return akasha_client.httpx.Response(201, json={"success": True, "status": 201})
            self.request_calls += 1
            status = 401 if self.request_calls == 1 else 200
            return akasha_client.httpx.Response(
                status, json={"data": {}, "success": status == 200, "status": status}
            )

        def close(self):
            pass

    return FakeHTTP, calls


def swap_transport(client: AkashaClient, fake_cls):
    client._client.close()
    client._client = fake_cls()
    return client


def test_clients_share_jwt_in_memory_and_from_local_cache(tmp_path, monkeypatch):
    token = _jwt(time.time() + 3600)
    FakeHTTP, calls = fake_http(token)

    monkeypatch.setattr(akasha_client, "_AUTH_CACHE_PATH", tmp_path / "auth.json")
    akasha_client._AUTH_TOKENS.clear()
    config = AkashaConfig(
        base_url="http://akasha", email="owner@example.com", password="pw"
    )
    clients = [swap_transport(AkashaClient(config), FakeHTTP) for _ in range(4)]

    threads = [threading.Thread(target=client.login) for client in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls["login"] == 1
    assert all(
        client._client.cookies.get(akasha_client.AUTH_TOKEN_COOKIE) == token
        for client in clients
    )

    akasha_client._AUTH_TOKENS.clear()
    restarted = swap_transport(AkashaClient(config), FakeHTTP)
    restarted.login()
    assert calls["login"] == 1
    assert restarted._client.cookies.get(akasha_client.AUTH_TOKEN_COOKIE) == token
    akasha_client._AUTH_TOKENS.clear()


def test_unauthorized_client_reuses_jwt_refreshed_by_another_client(
    tmp_path, monkeypatch
):
    stale = _jwt(time.time() + 3600)
    fresh = _jwt(time.time() + 7200)
    FakeHTTP, calls = fake_http(fresh)

    monkeypatch.setattr(akasha_client, "_AUTH_CACHE_PATH", tmp_path / "auth.json")
    config = AkashaConfig(base_url="http://akasha", email="e@x", password="pw")
    key = akasha_client._auth_cache_key(config)
    akasha_client._AUTH_TOKENS.clear()
    akasha_client._AUTH_TOKENS[key] = stale
    clients = [swap_transport(AkashaClient(config), FakeHTTP) for _ in range(2)]
    for client in clients:
        client._client.cookies.set(akasha_client.AUTH_TOKEN_COOKIE, stale)

    assert clients[0].get("users/me") == {}
    assert clients[1].get("users/me") == {}
    assert calls["login"] == 1
    assert all(
        client._client.cookies.get(akasha_client.AUTH_TOKEN_COOKIE) == fresh
        for client in clients
    )
    akasha_client._AUTH_TOKENS.clear()


def test_unauthorized_without_cookie_logs_in_and_saves_new_jwt(tmp_path, monkeypatch):
    fresh = _jwt(time.time() + 7200)
    FakeHTTP, calls = fake_http(fresh)

    cache_path = tmp_path / "auth.json"
    monkeypatch.setattr(akasha_client, "_AUTH_CACHE_PATH", cache_path)
    akasha_client._AUTH_TOKENS.clear()
    config = AkashaConfig(base_url="http://akasha", email="e@x", password="pw")
    client = swap_transport(AkashaClient(config), FakeHTTP)

    assert client.get("users/me") == {}
    assert calls["login"] == 1
    assert client._client.cookies.get(akasha_client.AUTH_TOKEN_COOKIE) == fresh
    assert json.loads(cache_path.read_text(encoding="utf-8"))[
        akasha_client._auth_cache_key(config)
    ] == fresh
    akasha_client._AUTH_TOKENS.clear()


def test_client_collects_failed_run_pages_across_pages(client, monkeypatch):
    calls: list[tuple[str, int, int]] = []

    def run_pages(run_id, *, page=1, limit=100):
        calls.append((run_id, page, limit))
        if page == 1:
            return {
                "items": [
                    {"sourcePageId": "ok", "status": "succeeded"},
                    {"sourcePageId": "text-failed", "status": "failed"},
                    {"sourcePageId": "ordinary-skip", "status": "skipped"},
                ],
                "total": 101,
                "limit": 100,
            }
        return {
            "items": [
                {
                    "sourcePageId": "merge-failed",
                    "status": "succeeded",
                    "mergeStatus": "failed",
                },
                {"sourcePageId": "text-failed", "status": "failed"},
                {
                    "sourcePageId": "cancelled",
                    "status": "skipped",
                    "errorCode": "manual_cancelled",
                },
            ],
            "total": 101,
            "limit": 100,
        }

    monkeypatch.setattr(client, "run_pages", run_pages)
    assert client.retryable_run_page_ids(["run-1"]) == [
        "text-failed",
        "merge-failed",
        "cancelled",
    ]
    assert calls == [("run-1", 1, 100), ("run-1", 2, 100)]


@pytest.mark.parametrize(
    ("pages", "run_ids", "expected"),
    [
        pytest.param(
            {
                "original": [
                    {"sourcePageId": "eventually-ok", "status": "failed"},
                    {"sourcePageId": "still-failed", "status": "failed"},
                    {"sourcePageId": "already-ok", "status": "succeeded"},
                ],
                "retry": [
                    {"sourcePageId": "eventually-ok", "status": "succeeded"},
                    {
                        "sourcePageId": "still-failed",
                        "status": "skipped",
                        "errorCode": "manual_cancelled",
                    },
                ],
            },
            ["original", "retry"],
            ["still-failed"],
            id="latest-status-across-runs",
        ),
        pytest.param(
            {
                "newer": [
                    {
                        "sourcePageId": "page",
                        "status": "succeeded",
                        "updatedAt": "2026-09-24T10:00:00Z",
                    }
                ],
                "older": [
                    {
                        "sourcePageId": "page",
                        "status": "skipped",
                        "errorCode": "manual_cancelled",
                        "updatedAt": "2026-09-24T09:00:00Z",
                    }
                ],
            },
            ["newer", "older"],
            [],
            id="remote-updated-at-beats-run-order",
        ),
    ],
)
def test_retryable_pages(client, monkeypatch, pages, run_ids, expected):
    monkeypatch.setattr(
        client,
        "run_pages",
        lambda run_id, **_: {"items": pages[run_id], "total": len(pages[run_id]), "limit": 100},
    )
    assert client.retryable_run_page_ids(run_ids) == expected


def test_client_requires_retry_batches_of_at_most_100(client, monkeypatch):
    batches: list[list[str]] = []

    def post(path, body):
        assert path == "llm-wiki/admin/retry-pages"
        batches.append(body["pageIds"])
        return {"queuedPageCount": 1, "jobIds": [f"run-{len(batches)}"]}

    monkeypatch.setattr(client, "post", post)
    with pytest.raises(ValueError, match="最多 100"):
        client.retry_pages([f"page-{index}" for index in range(101)])
    result = client.retry_pages([f"page-{index}" for index in range(100)])
    assert [len(batch) for batch in batches] == [100]
    assert result == {"queuedPageCount": 1, "jobIds": ["run-1"]}


def test_embedding_drift_is_detected():
    changed = {
        "configs": [
            {**entry, "model": "e2"} if entry["feature"] == "embedding" else entry
            for entry in CONFIGS["configs"]
        ]
    }
    assert model_configs.matches(CONFIGS, changed, "compiler") is True
    assert model_configs.matches(CONFIGS, changed, "embedding") is False
    assert model_configs.drift(changed, CONFIGS) == {
        "compiler": False,
        "embedding": True,
        "answer": False,
        "image": False,
    }


def test_normalize_model_configs_is_order_stable():
    reversed_configs = {"configs": list(reversed(CONFIGS["configs"]))}
    assert model_configs.normalize(CONFIGS) == model_configs.normalize(reversed_configs)


def test_compile_requires_owner(ready_connection, monkeypatch):
    monkeypatch.setattr(compile, "AkashaClient", lambda config: FakeClient(config, role="member"))
    ctx = context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2})
    with pytest.raises(RuntimeError, match="owner"):
        execute(compile.run, ctx)


def test_compile_fails_when_quality_gate_reports_nothing(ready_connection, monkeypatch):
    def factory(config):
        client = FakeClient(config)
        client.quality = {}
        return client

    monkeypatch.setattr(compile, "AkashaClient", factory)
    ctx = context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2})
    with pytest.raises(RuntimeError, match="质量闸门"):
        execute(compile.run, ctx)

    run = compile_store.list_compile_runs(ready_connection)[0]
    assert run["status"] == run_store.STATUS_FAILED


def test_compile_records_pace_estimate(ready_connection, monkeypatch):
    def factory(config):
        return FakeClient(
            config,
            run_items=[
                {
                    "runId": "remote-run-1",
                    "status": "succeeded",
                    "runDurationMs": 8000,
                    "progress": {"text": {"expected": 4}},
                },
            ],
        )

    monkeypatch.setattr(compile, "AkashaClient", factory)
    ctx = context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2, "run_id": "p1"})
    execute(compile.run, ctx)

    from akasha_benchmark.store import loads

    run = compile_store.compile_run_by_run_id(ready_connection, "p1")
    pace = loads(run["pace_json"])
    assert pace["pages"] == 4
    assert pace["total_ms"] == 8000
    assert pace["per_page_ms"] == 2000


def test_compile_reports_page_failure_reason(ready_connection, monkeypatch):
    pages = [
        {
            "status": "failed",
            "errorCode": "provider_error",
            "errorSummary": "Knowledge compiler provider request failed.",
            "title": f"Doc {index}",
        }
        for index in range(8)
    ]

    def factory(config):
        client = FakeClient(config, page_log_items=pages)
        client.quality = {"summary": {"missingChunkPageCount": 8, "missingSourcePageCount": 8}}
        return client

    monkeypatch.setattr(compile, "AkashaClient", factory)
    ctx = context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2})
    with pytest.raises(RuntimeError, match="provider_error") as caught:
        execute(compile.run, ctx)

    assert "8 篇" in str(caught.value)


def test_query_can_use_partially_successful_compile(ready_connection, monkeypatch):
    class Partial(FakeClient):
        def run_diagnostics(self, space_ids, *, limit=50):
            return {
                "items": [
                    {
                        "runId": "remote-run-1",
                        "status": "failed",
                        "runDurationMs": 8000,
                        "progress": {
                            "text": {
                                "expected": 4,
                                "succeeded": 3,
                                "failed": 1,
                                "skipped": 0,
                            }
                        },
                    }
                ]
            }

    def factory(config):
        client = Partial(config)
        client.quality = {
            "summary": {
                "missingChunkPageCount": 1,
                "missingEmbeddingPageCount": 1,
                "missingSourcePageCount": 1,
                "stalePageCount": 0,
            }
        }
        return client

    monkeypatch.setattr(compile, "AkashaClient", factory)
    ctx = context(
        ready_connection,
        {"datasets": ["hotpotqa"], "qa_limit": 2, "run_id": "partial"},
    )
    with pytest.raises(RuntimeError, match="质量闸门"):
        execute(compile.run, ctx)

    run = compile_store.compile_run_by_run_id(ready_connection, "partial")
    assert run["status"] == run_store.STATUS_FAILED
    compile_id = int(run["id"])
    assert compile_store.compile_ready(ready_connection, compile_id)["ready"] is True

    query_client = FakeClient(None)
    monkeypatch.setattr(query, "AkashaClient", lambda config: query_client)
    execute(query.run, context(ready_connection, {"compile_id": compile_id}))
    assert query_client.queries


def test_compile_fails_when_no_run_was_accepted(ready_connection, monkeypatch):
    def factory(config):
        return FakeClient(config, accepted_runs=0, run_items=[])

    monkeypatch.setattr(compile, "AkashaClient", factory)
    ctx = context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2})
    with pytest.raises(RuntimeError, match="编译没有启动"):
        execute(compile.run, ctx)

    run = compile_store.list_compile_runs(ready_connection)[0]
    assert run["status"] == run_store.STATUS_FAILED


def test_compile_waits_when_runs_are_still_active(ready_connection, monkeypatch):
    seen: list[dict[str, int]] = []

    class Slow(FakeClient):
        def run_diagnostics(self, space_ids, *, limit=50):
            seen.append({})
            status = "compiling" if len(seen) == 1 else "succeeded"
            return {
                "items": [
                    {
                        "runId": "remote-run-1",
                        "status": status,
                        "progress": {
                            "text": {"expected": 1, "succeeded": int(status == "succeeded"), "failed": 0, "skipped": 0}
                        },
                    }
                ]
            }

    monkeypatch.setattr(compile, "AkashaClient", lambda config: Slow(config))
    monkeypatch.setattr(compile.time, "sleep", lambda _seconds: None)
    ctx = context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2, "run_id": "r0"})
    execute(compile.run, ctx)

    assert len(seen) == 2
    run = compile_store.compile_run_by_run_id(ready_connection, "r0")
    assert run["status"] == run_store.STATUS_SUCCEEDED


def test_compile_progress_uses_current_run_not_space_history(ready_connection, monkeypatch):
    monkeypatch.setattr(compile, "POLL_INTERVAL_SECONDS", 0)

    class Runs(FakeClient):
        def __init__(self, config):
            super().__init__(config)
            self.polls = 0

        def run_diagnostics(self, space_ids, *, limit=50):
            self.polls += 1
            current_status = "compiling" if self.polls < 3 else "succeeded"
            return {
                "items": [
                    {
                        "runId": "old",
                        "spaceJobSequence": 1,
                        "status": "succeeded",
                        "runDurationMs": 999999,
                        "progress": {"text": {"expected": 100, "succeeded": 100}},
                    },
                    {
                        "runId": "current",
                        "spaceJobSequence": 2,
                        "status": current_status,
                        "runDurationMs": 2000,
                        "progress": {
                            "text": {
                                "expected": 4,
                                "succeeded": 2 if current_status == "compiling" else 4,
                                "failed": 0,
                                "skipped": 0,
                            }
                        },
                    },
                ]
            }

    client = Runs(None)
    result = compile._wait_for_compile(
        context(ready_connection, {}),
        client,
        "space-1",
        expect_runs=1,
        baseline_run_ids={"old"},
        baseline_sequence=1,
    )
    assert result["status_counts"] == {"succeeded": 1}
    assert result["runs"][0]["runId"] == "current"


@pytest.mark.parametrize(
    ("run_id", "page_ids", "items_by_run", "runs", "expected"),
    [
        pytest.param(
            "target-progress",
            ["target-0", "target-1"],
            {
                "run-1": [
                    {"sourcePageId": "target-0", "status": "succeeded"},
                    {"sourcePageId": "target-1", "status": "succeeded"},
                    {"sourcePageId": "unrelated", "status": "succeeded"},
                ]
            },
            [{"runId": "run-1"}],
            {"expected": 2, "succeeded": 2, "failed": 0, "skipped": 0},
            id="ignores-pages-outside-this-compile",
        ),
        pytest.param(
            "retry-progress",
            ["retry-0", "retry-1"],
            {
                "original": [
                    {"sourcePageId": "retry-0", "status": "succeeded"},
                    {"sourcePageId": "retry-1", "status": "failed"},
                ],
                "retry": [{"sourcePageId": "retry-1", "status": "succeeded"}],
            },
            [{"runId": "original"}, {"runId": "retry"}],
            {"expected": 2, "succeeded": 2, "failed": 0, "skipped": 0},
            id="latest-status-wins-across-retry-runs",
        ),
        pytest.param(
            "merge-progress",
            ["merge-page"],
            {
                "run": [
                    {"sourcePageId": "merge-page", "status": "succeeded", "mergeStatus": "failed"}
                ]
            },
            [{"runId": "run"}],
            {"expected": 1, "succeeded": 0, "failed": 1, "skipped": 0},
            id="merge-failure-counts-as-failed",
        ),
    ],
)
def test_compile_progress(ready_connection, run_id, page_ids, items_by_run, runs, expected):
    _, task_id = compile_with_pages(ready_connection, run_id, page_ids)
    progress = compile._target_run_progress(
        context(ready_connection, {}, task_id=task_id), pages_client(items_by_run), runs
    )
    assert progress == expected


def test_compile_progress_uses_remote_total_when_retrying(ready_connection, monkeypatch):
    compile_id = make_compile_run(ready_connection, "remote-total", ["hotpotqa"], qa_limit=1)
    task_id = bind_compile_task(ready_connection, compile_id)
    page_ids = [f"page-{index}" for index in range(244)]
    monkeypatch.setattr(
        compile.compile_store,
        "compile_docs",
        lambda *_args, **_kwargs: [{"page_id": page_id} for page_id in page_ids],
    )

    client = pages_client(
        {
            "original": [
                {"sourcePageId": page_id, "status": "succeeded"} for page_id in page_ids[:105]
            ],
            "retry": [
                {"sourcePageId": f"page-{index}", "status": "succeeded"}
                for index in (105, 106, 107)
            ],
        }
    )
    progress = compile._target_run_progress(
        context(ready_connection, {}, task_id=task_id),
        client,
        [
            {"runId": "original", "progress": {"text": {"expected": 244}}},
            {"runId": "retry", "progress": {"text": {"expected": 3}}},
        ],
    )

    assert progress == {"expected": 244, "succeeded": 108, "failed": 0, "skipped": 0}


def test_retry_batches_resume_current_run_before_submitting_pending(ready_connection, monkeypatch):
    compile_id = make_compile_run(ready_connection, "retry-state", ["hotpotqa"], qa_limit=1)
    current = [f"current-{index}" for index in range(100)]
    pending = [f"pending-{index}" for index in range(6)]
    retry_state = {
            "retry_current_page_ids": current,
            "retry_pending_page_ids": pending,
            "retry_current_run_ids": ["run-current"],
            "retry_run_ids": ["run-current"],
            "retry_completed": 0,
            "retry_total": 106,
            "retry_progress": {"expected": 106, "succeeded": 0, "failed": 0, "skipped": 0},
        }
    task_id = task_store.create_task(
        ready_connection, stage="compile", params=retry_state
    )
    task_store.set_task_target(ready_connection, task_id, "compile", compile_id)
    ready_connection.commit()
    ctx = context(ready_connection, retry_state, task_id=task_id)
    submitted: list[list[str]] = []

    class RetryClient(FakeClient):
        def retry_pages(self, page_ids):
            submitted.append(list(page_ids))
            return {"jobIds": ["run-pending"], "queuedPageCount": 1}

    def wait(_ctx, _client, _space, **kwargs):
        count = len(kwargs["progress_page_ids"])
        return {
            "status_counts": {"succeeded": 1}, "no_runs": False,
            "timed_out": False, "runs": [],
            "progress": {"expected": count, "succeeded": count, "failed": 0, "skipped": 0},
        }

    monkeypatch.setattr(compile, "_wait_for_compile", wait)
    result, run_count = compile._retry_batches(ctx, RetryClient(None), "space")

    assert submitted == [pending]
    assert run_count == 2
    assert result["progress"] == {
        "expected": 106, "succeeded": 106, "failed": 0, "skipped": 0
    }


def test_retry_batches_reject_inconsistent_remote_run_count(ready_connection):
    compile_id = make_compile_run(ready_connection, "bad-retry-count", ["hotpotqa"], qa_limit=1)
    task_id = bind_compile_task(ready_connection, compile_id)

    class BadCount(FakeClient):
        def retry_pages(self, page_ids):
            return {"queuedPageCount": 2, "jobIds": ["one-run"]}

    with pytest.raises(RuntimeError, match="重试响应计数不一致"):
        compile._retry_batches(
            context(ready_connection, {}, task_id=task_id),
            BadCount(None),
            "space",
            ["page-1"],
        )


def test_compile_does_not_finish_on_historical_run_before_new_run_appears(
    ready_connection, monkeypatch
):
    monkeypatch.setattr(compile, "POLL_INTERVAL_SECONDS", 0)

    class Delayed(FakeClient):
        def __init__(self, config):
            super().__init__(config)
            self.polls = 0

        def run_diagnostics(self, space_ids, *, limit=50):
            self.polls += 1
            items = [
                {
                    "runId": "old",
                    "spaceJobSequence": 1,
                    "status": "succeeded",
                    "progress": {"text": {"expected": 100, "succeeded": 100}},
                }
            ]
            if self.polls > 1:
                items.append(
                    {
                        "runId": "current",
                        "spaceJobSequence": 2,
                        "status": "succeeded",
                        "progress": {"text": {"expected": 4, "succeeded": 4}},
                    }
                )
            return {"items": items}

    client = Delayed(None)
    result = compile._wait_for_compile(
        context(ready_connection, {}),
        client,
        "space-1",
        expect_runs=1,
        baseline_run_ids={"old"},
        baseline_sequence=1,
    )
    assert client.polls == 2
    assert result["runs"][0]["runId"] == "current"


def test_compile_succeeds_and_records_pages(ready_connection, monkeypatch):
    clients: list[FakeClient] = []

    def factory(config):
        client = FakeClient(config)
        clients.append(client)
        return client

    monkeypatch.setattr(compile, "AkashaClient", factory)
    ctx = context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2, "run_id": "r1"})
    execute(compile.run, ctx)

    run = compile_store.compile_run_by_run_id(ready_connection, "r1")
    assert run["status"] == run_store.STATUS_SUCCEEDED
    assert run["space_id"] == "space-1"
    docs = compile_store.compile_docs(ready_connection, int(run["id"]))
    assert docs and all(doc["page_id"] for doc in docs)
    assert sum(len(client.imported) for client in clients) == len(docs)
    assert task_store.get_task(ready_connection, ctx.task_id)["params"][
        "remote_compile_run_ids"
    ] == ["remote-run-1"]
    assert compile_store.compile_ready(ready_connection, int(run["id"]))["ready"] is True
    task = task_store.get_task(ready_connection, ctx.task_id)
    assert (task["progress_done"], task["progress_total"], task["progress_note"]) == (
        len(docs),
        len(docs),
        "编译完成",
    )
    assert all("编译进度" not in entry["message"] for entry in task_store.task_logs(ready_connection, ctx.task_id))


def test_compile_resume_retries_only_failed_remote_pages(ready_connection, monkeypatch):
    clients: list[FakeClient] = []
    compile_calls = 0
    quality_checks = 0
    retried: list[list[str]] = []
    failed_page_id = "00000000-0000-0000-0000-000000000004"

    class Resumable(FakeClient):
        def compile_spaces(self, space_ids):
            nonlocal compile_calls
            compile_calls += 1
            if compile_calls > 1:
                pytest.fail("继续任务不应再次触发全空间编译")
            return {
                "acceptedRunCount": 1,
                "coalescedRunCount": 0,
                "runs": [{"runId": "remote-run-1", "disposition": "created"}],
            }

        def run_diagnostics(self, space_ids, *, limit=50):
            items = [
                {
                    "runId": "remote-run-1",
                    "status": "partial",
                    "progress": {
                        "text": {
                            "expected": 4,
                            "succeeded": 3,
                            "failed": 1,
                            "skipped": 0,
                        }
                    },
                }
            ]
            if retried:
                items.append(
                    {
                        "runId": "retry-run-1",
                        "status": "succeeded",
                        "progress": {
                            "text": {
                                "expected": 1,
                                "succeeded": 1,
                                "failed": 0,
                                "skipped": 0,
                            }
                        },
                    }
                )
            return {"items": items}

        def quality_diagnostics(self, space_ids):
            nonlocal quality_checks
            quality_checks += 1
            missing = 1 if quality_checks == 1 else 0
            return {
                "summary": {
                    "missingChunkPageCount": missing,
                    "missingEmbeddingPageCount": 0,
                    "missingSourcePageCount": missing,
                    "stalePageCount": 0,
                }
            }

        def run_pages(self, run_id, *, page=1, limit=100):
            if run_id == "retry-run-1":
                return {
                    "items": [{"sourcePageId": failed_page_id, "status": "succeeded"}],
                    "total": 1,
                    "page": page,
                    "limit": limit,
                }
            assert run_id == "remote-run-1"
            return {
                "items": [
                    {"sourcePageId": "00000000-0000-0000-0000-000000000001", "status": "succeeded"},
                    {"sourcePageId": failed_page_id, "status": "failed"},
                ],
                "total": 2,
                "page": page,
                "limit": limit,
            }

        def retry_pages(self, page_ids):
            retried.append(list(page_ids))
            return {"queuedPageCount": 1, "jobIds": ["retry-run-1"]}

    def factory(config):
        client = Resumable(config)
        clients.append(client)
        return client

    monkeypatch.setattr(compile, "AkashaClient", factory)
    params = {"datasets": ["hotpotqa"], "qa_limit": 2, "run_id": "r1"}
    ctx = context(ready_connection, params)
    with pytest.raises(RuntimeError, match="质量闸门"):
        execute(compile.run, ctx)
    first_run_clients = len(clients)
    imported = sum(len(client.imported) for client in clients)

    execute(compile.run, context(ready_connection, {}, task_id=ctx.task_id))

    assert imported > 0
    assert all(client.imported == [] for client in clients[first_run_clients:])
    assert compile_calls == 1
    assert retried == [[failed_page_id]]
    assert task_store.get_task(ready_connection, ctx.task_id)["params"][
        "remote_compile_run_ids"
    ] == ["remote-run-1", "retry-run-1"]


def test_compile_resume_adopts_active_remote_run(ready_connection, monkeypatch):
    compile_id = make_compile_run(ready_connection, "recover", ["hotpotqa"], qa_limit=1)
    compile.build_subset(
        ready_connection,
        compile_id,
        "hotpotqa",
        seed=1,
        qa_limit=1,
        negatives_ratio=1.0,
    )
    compile_store.update_compile_run(
        ready_connection,
        compile_id,
        space_id="space-1",
        workspace_id="w",
        model_configs_json=dumps(CONFIGS),
    )
    for index, doc in enumerate(compile_store.compile_docs(ready_connection, compile_id)):
        compile_store.record_page(
            ready_connection,
            compile_id,
            doc["dataset"],
            doc["doc_id"],
            page_id=f"page-{index}",
            error=None,
        )
    params = {
        "datasets": ["hotpotqa"],
        "seed": 1,
        "qa_limit": 1,
        "negatives_ratio": 1.0,
        "full_corpus": False,
        "import_concurrency": 1,
        "run_id": "recover",
        "remote_compile_run_ids": ["active-run"],
    }
    task_id = task_store.create_task(ready_connection, stage="test", params=params)
    task_store.set_task_target(ready_connection, task_id, "compile", compile_id)
    ready_connection.commit()
    polls = 0

    class Recovering(FakeClient):
        def put_model_config(self, feature, payload):
            pytest.fail("恢复编译任务不应修改远端模型配置")

        def run_diagnostics(self, space_ids, *, limit=50):
            nonlocal polls
            polls += 1
            status = "compiling" if polls == 1 else "succeeded"
            succeeded = 0 if status == "compiling" else 1
            return {
                "items": [
                    {
                        "runId": "active-run",
                        "status": status,
                        "progress": {
                            "text": {
                                "expected": 1,
                                "succeeded": succeeded,
                                "failed": 0,
                                "skipped": 0,
                            }
                        },
                    }
                ]
            }

        def compile_spaces(self, space_ids):
            pytest.fail("接管活动 Run 时不应新建全空间编译")

        def retry_pages(self, page_ids):
            pytest.fail("接管活动 Run 时不应提交页面重试")

    monkeypatch.setattr(compile, "AkashaClient", lambda config: Recovering(config))
    monkeypatch.setattr(compile.time, "sleep", lambda _seconds: None)

    execute(compile.run, context(ready_connection, {}, task_id=task_id))

    assert polls == 2
    assert task_store.get_task(ready_connection, task_id)["status"] == task_store.SUCCEEDED


def test_compile_resume_retries_target_pages_when_cancelled_before_page_initialization(
    ready_connection, monkeypatch
):
    monkeypatch.setattr(compile, "AkashaClient", lambda config: FakeClient(config))
    ctx = context(
        ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 1, "run_id": "early"}
    )
    execute(compile.run, ctx)
    task_store.transition(ready_connection, ctx.task_id, task_store.PAUSED)
    ready_connection.commit()
    retried: list[list[str]] = []

    class CancelledBeforeInit(FakeClient):
        def run_diagnostics(self, space_ids, *, limit=50):
            cancelled = {
                "runId": "remote-run-1",
                "status": "cancelled",
                "progress": {
                    "text": {"expected": 0, "succeeded": 0, "failed": 0, "skipped": 0}
                },
            }
            if not retried:
                return {"items": [cancelled]}
            return {
                "items": [
                    cancelled,
                    {
                        "runId": "new-run",
                        "status": "succeeded",
                        "progress": {
                            "text": {
                                "expected": 1,
                                "succeeded": 1,
                                "failed": 0,
                                "skipped": 0,
                            }
                        },
                    },
                ]
            }

        def compile_spaces(self, space_ids):
            pytest.fail("继续未初始化 Run 不应扫描整个 Space")

        def retry_pages(self, page_ids):
            retried.append(list(page_ids))
            return {"queuedPageCount": len(page_ids), "jobIds": ["new-run"]}

        def run_pages(self, run_id, *, page=1, limit=100):
            if not retried:
                return {"items": [], "total": 0, "limit": limit}
            items = [
                {"sourcePageId": page_id, "status": "succeeded"}
                for page_id in retried[0]
            ]
            return {"items": items, "total": len(items), "limit": limit}

    monkeypatch.setattr(compile, "AkashaClient", lambda config: CancelledBeforeInit(config))

    execute(compile.run, context(ready_connection, {}, task_id=ctx.task_id))

    expected = list(
        dict.fromkeys(
            str(row["page_id"])
            for row in compile_store.compile_docs(
                ready_connection, int(ctx.target("compile"))
            )
            if row.get("page_id")
        )
    )
    assert retried == [expected]
    assert task_store.get_task(ready_connection, ctx.task_id)["params"][
        "remote_compile_run_ids"
    ] == ["remote-run-1", "new-run"]


def test_compile_imports_concurrently(ready_connection, monkeypatch):
    lock = threading.Lock()
    active = 0
    peak = 0
    imported = 0

    class Concurrent(FakeClient):
        def import_page_text(self, filename, markdown, space_id):
            nonlocal active, peak, imported
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
                imported += 1
                page_id = f"page-{imported}"
            return {"id": page_id}

    monkeypatch.setattr(compile, "AkashaClient", lambda config: Concurrent(config))
    ctx = context(
        ready_connection,
        {
            "datasets": ["hotpotqa"],
            "qa_limit": 2,
            "run_id": "concurrent",
            "import_concurrency": 3,
        },
    )
    execute(compile.run, ctx)

    run = compile_store.compile_run_by_run_id(ready_connection, "concurrent")
    docs = compile_store.compile_docs(ready_connection, int(run["id"]))
    assert peak > 1
    assert imported == len(docs)
    assert task_store.get_task(ready_connection, ctx.task_id)["params"][
        "import_concurrency"
    ] == 3


def test_compile_retries_then_pauses_and_resume_only_imports_pending(
    ready_connection, monkeypatch
):
    attempts: dict[str, int] = {}
    imported_ids = 0
    compile_calls = 0
    keep_failing = True
    failed_filename: str | None = None
    lock = threading.Lock()

    class Retryable(FakeClient):
        def import_page_text(self, filename, markdown, space_id):
            nonlocal failed_filename, imported_ids
            with lock:
                failed_filename = failed_filename or filename
                attempts[filename] = attempts.get(filename, 0) + 1
                if keep_failing and filename == failed_filename:
                    return {}
                imported_ids += 1
                return {"id": f"page-{imported_ids}"}

        def compile_spaces(self, space_ids):
            nonlocal compile_calls
            compile_calls += 1
            return super().compile_spaces(space_ids)

    monkeypatch.setattr(compile, "AkashaClient", lambda config: Retryable(config))
    ctx = context(
        ready_connection,
        {
            "datasets": ["hotpotqa"],
            "qa_limit": 2,
            "run_id": "retry",
            "import_concurrency": 3,
        },
    )

    from akasha_benchmark.task import Paused

    with pytest.raises(Paused, match="重试后仍未导入"):
        execute(compile.run, ctx)

    run = compile_store.compile_run_by_run_id(ready_connection, "retry")
    compile_id = int(run["id"])
    docs = compile_store.compile_docs(ready_connection, compile_id)
    succeeded_filenames = {f"{doc['doc_id']}.md" for doc in docs if doc["page_id"]}
    assert attempts[failed_filename] == 2
    assert all(attempts[name] == 1 for name in succeeded_filenames)
    assert len(succeeded_filenames) == len(docs) - 1
    assert task_store.get_task(ready_connection, ctx.task_id)["status"] == task_store.PAUSED
    assert run["status"] == run_store.STATUS_PAUSED
    assert compile_calls == 0

    keep_failing = False
    execute(compile.run, context(ready_connection, {}, task_id=ctx.task_id))

    resumed_docs = compile_store.compile_docs(ready_connection, compile_id)
    assert all(doc["page_id"] for doc in resumed_docs)
    assert attempts[failed_filename] == 3
    assert all(attempts[name] == 1 for name in succeeded_filenames)
    assert compile_calls == 1


def _compiled(connection, monkeypatch) -> int:
    monkeypatch.setattr(compile, "AkashaClient", lambda config: FakeClient(config))
    execute(
        compile.run, context(connection, {"datasets": ["hotpotqa"], "qa_limit": 2, "run_id": "r1"})
    )
    return int(compile_store.compile_run_by_run_id(connection, "r1")["id"])


def test_query_blocks_embedding_model_change(ready_connection, monkeypatch):
    compile_id = _compiled(ready_connection, monkeypatch)
    changed = {
        "configs": [
            {**entry, "model": "e2"} if entry["feature"] == "embedding" else entry
            for entry in CONFIGS["configs"]
        ]
    }
    monkeypatch.setattr(query, "AkashaClient", lambda config: FakeClient(config, configs=changed))
    ctx = context(
        ready_connection, {"compile_id": compile_id, "name": "embedding-changed"}
    )
    with pytest.raises(RuntimeError, match="embedding.*必须重新编译"):
        execute(query.run, ctx)
    assert query_store.query_run_by_name(ready_connection, "embedding-changed") is None


def test_query_allows_endpoint_and_other_config_changes(
    ready_connection, monkeypatch
):
    compile_id = _compiled(ready_connection, monkeypatch)
    changed = {
        "configs": [
            entry
            if entry["feature"] == "embedding"
            else {**entry, "model": f"other-{entry['feature']}"}
            for entry in CONFIGS["configs"]
        ]
    }
    monkeypatch.setattr(
        query, "AkashaClient", lambda config: FakeClient(config, configs=changed)
    )

    execute(
        query.run,
        context(ready_connection, {"compile_id": compile_id, "name": "compatible"}),
    )

    run = query_store.query_run_by_name(ready_connection, "compatible")
    assert run is not None
    assert run["status"] == run_store.STATUS_SUCCEEDED


def test_query_uses_remote_answer_model_without_changing_it(ready_connection, monkeypatch):
    compile_id = _compiled(ready_connection, monkeypatch)
    pushed: list[tuple[str, dict[str, Any]]] = []
    active = CONFIGS

    class GroupClient(FakeClient):
        def put_model_config(self, feature, payload):
            pushed.append((feature, payload))
            return {}

        def get_model_configs(self):
            return active

    monkeypatch.setattr(query, "AkashaClient", lambda config: GroupClient(config))
    execute(
        query.run,
        context(
            ready_connection,
                {
                    "compile_id": compile_id,
                    "name": "query-own-group",
                "concurrency": 2,
            },
        ),
    )

    run = query_store.query_run_by_name(ready_connection, "query-own-group")
    assert run["concurrency"] == 2
    assert loads(run["model_configs_json"]) == active
    assert pushed == []


def test_compile_uses_remote_models_without_changing_them(ready_connection, monkeypatch):
    pushed: list[str] = []

    class SelectedClient(FakeClient):
        def put_model_config(self, feature, payload):
            pushed.append(feature)
            return {}

    monkeypatch.setattr(compile, "AkashaClient", lambda config: SelectedClient(config))
    execute(
        compile.run,
        context(
            ready_connection,
                {
                    "datasets": ["hotpotqa"],
                    "qa_limit": 1,
                    "run_id": "selected-models",
                },
        ),
    )

    run = compile_store.compile_run_by_run_id(ready_connection, "selected-models")
    assert pushed == []
    assert loads(run["model_configs_json"]) == CONFIGS


def test_query_refuses_on_workspace_mismatch(ready_connection, monkeypatch):
    compile_id = _compiled(ready_connection, monkeypatch)

    class OtherWorkspace(FakeClient):
        def current_user(self):
            return {
                "user": {"id": "u", "role": "owner"},
                "workspace": {"id": "w-other", "name": "Other"},
            }

    monkeypatch.setattr(query, "AkashaClient", lambda config: OtherWorkspace(config))
    with pytest.raises(RuntimeError, match="workspace"):
        execute(query.run, context(ready_connection, {"compile_id": compile_id}))
    assert query_store.list_query_runs(ready_connection, compile_id) == []


def test_compile_resume_refuses_on_workspace_mismatch(ready_connection, monkeypatch):
    compile_id = _compiled(ready_connection, monkeypatch)
    ready_connection.execute(
        "UPDATE compile_doc SET page_id = NULL WHERE compile_id = ?", (compile_id,)
    )
    ready_connection.commit()

    class OtherWorkspace(FakeClient):
        def current_user(self):
            return {
                "user": {"id": "u", "role": "owner"},
                "workspace": {"id": "w-other", "name": "Other"},
            }

    clients: list[FakeClient] = []

    def factory(config):
        client = OtherWorkspace(config)
        clients.append(client)
        return client

    monkeypatch.setattr(compile, "AkashaClient", factory)
    with pytest.raises(RuntimeError, match="workspace"):
        execute(compile.run, context(ready_connection, {}, task_id=1))
    assert clients[-1].imported == []


def test_workspace_mismatch_rejects_missing_record(ready_connection, monkeypatch):
    compile_id = _compiled(ready_connection, monkeypatch)
    ready_connection.execute(
        "UPDATE compile_run SET workspace_id = NULL WHERE id = ?", (compile_id,)
    )
    ready_connection.commit()
    assert "没有记录 workspace" in compile_store.workspace_mismatch(
        ready_connection, compile_id, "w-other"
    )


def test_query_refuses_unready_compile(ready_connection, monkeypatch):
    compile_id = make_compile_run(ready_connection, "half", ["hotpotqa"], qa_limit=1)
    ready_connection.commit()
    monkeypatch.setattr(query, "AkashaClient", lambda config: FakeClient(config))
    with pytest.raises(ValueError, match="不能用于查询"):
        execute(query.run, context(ready_connection, {"compile_id": compile_id}))


def test_query_records_responses_and_resumes(ready_connection, monkeypatch):
    compile_id = _compiled(ready_connection, monkeypatch)
    clients: list[FakeClient] = []

    def factory(config):
        client = FakeClient(config)
        clients.append(client)
        return client

    monkeypatch.setattr(query, "AkashaClient", factory)
    params = {"compile_id": compile_id, "name": "q1"}
    ctx = context(ready_connection, params)
    execute(query.run, ctx)

    run = query_store.query_run_by_name(ready_connection, "q1")
    assert run["status"] == run_store.STATUS_SUCCEEDED
    responses = query_store.responses_of(ready_connection, int(run["id"]))
    assert len(responses) == 2
    assert all(r["answer_mode"] == "knowledge" for r in responses)

    execute(query.run, ctx)
    assert clients[-1].queries == []


def test_query_quality_gate_rejects_generation_unavailable_answer(ready_connection, monkeypatch):
    compile_id = _compiled(ready_connection, monkeypatch)

    class EmptyAnswerClient(FakeClient):
        def query(self, question, space_ids):
            from akasha_benchmark.akasha_client import Response

            self.queries.append(question)
            return Response(
                status=200,
                body={
                    "answerMode": "knowledge",
                    "answer": query_store.ANSWER_GENERATION_UNAVAILABLE,
                    "retrievedSources": [{"sourcePageId": "p1"}],
                },
                latency_ms=10,
            )

    monkeypatch.setattr(query, "AkashaClient", lambda config: EmptyAnswerClient(config))
    ctx = context(ready_connection, {"compile_id": compile_id, "name": "empty-answer"})

    with pytest.raises(RuntimeError, match="质量阀门"):
        execute(query.run, ctx)

    run = query_store.query_run_by_name(ready_connection, "empty-answer")
    assert run["status"] == run_store.STATUS_FAILED
    assert query_store.retryable_response_count(ready_connection, int(run["id"])) == 2


def test_query_resume_refuses_remote_answer_config_drift(ready_connection, monkeypatch):
    compile_id = _compiled(ready_connection, monkeypatch)
    monkeypatch.setattr(query, "AkashaClient", lambda config: FakeClient(config))
    ctx = context(ready_connection, {"compile_id": compile_id, "name": "drift-query"})
    execute(query.run, ctx)

    changed = {
        "configs": [
            {**entry, **({"model": "changed-answer"} if entry["feature"] == "answer" else {})}
            for entry in CONFIGS["configs"]
        ]
    }

    class Drifted(FakeClient):
        def get_model_configs(self):
            return changed

        def put_model_config(self, feature, payload):
            pytest.fail("继续查询不应覆盖远端模型配置")

    monkeypatch.setattr(query, "AkashaClient", lambda config: Drifted(config))
    with pytest.raises(RuntimeError, match="不能继续原查询.*answer"):
        execute(query.run, context(ready_connection, {}, task_id=ctx.task_id))


def test_query_refuses_embedding_drift_from_compile_snapshot(ready_connection, monkeypatch):
    compile_id = _compiled(ready_connection, monkeypatch)
    changed = {
        "configs": [
            {**entry, **({"model": "changed-embedding"} if entry["feature"] == "embedding" else {})}
            for entry in CONFIGS["configs"]
        ]
    }

    class Drifted(FakeClient):
        def get_model_configs(self):
            return changed

    monkeypatch.setattr(query, "AkashaClient", lambda config: Drifted(config))
    with pytest.raises(RuntimeError, match="embedding.*必须重新编译"):
        execute(query.run, context(ready_connection, {"compile_id": compile_id}))


def test_query_uses_frozen_selection(ready_connection, monkeypatch):
    compile_id = _compiled(ready_connection, monkeypatch)
    monkeypatch.setattr(query, "AkashaClient", lambda config: FakeClient(config))
    execute(
        query.run,
        context(ready_connection, {"compile_id": compile_id, "name": "q1", "sample_limit": 1}),
    )
    run = query_store.query_run_by_name(ready_connection, "q1")
    assert len(query_store.query_samples(ready_connection, int(run["id"]))) == 1


def test_query_rejects_name_from_another_compile(ready_connection, monkeypatch):
    compile_id = _compiled(ready_connection, monkeypatch)
    other = make_compile_run(ready_connection, "other", [], qa_limit=1)
    query_store.create_query_run(
        ready_connection,
        name="taken",
        compile_id=other,
        concurrency=1,
        model_configs=CONFIGS,
    )
    ready_connection.commit()
    monkeypatch.setattr(query, "AkashaClient", lambda config: FakeClient(config))
    with pytest.raises(ValueError, match="已存在"):
        execute(query.run, context(ready_connection, {"compile_id": compile_id, "name": "taken"}))


def test_compile_snapshot_is_frozen_on_the_run(ready_connection, monkeypatch):
    compile_id = _compiled(ready_connection, monkeypatch)
    run = compile_store.get_compile_run(ready_connection, compile_id)
    assert run["model_configs_json"] == dumps(CONFIGS)


def test_same_name_does_not_resume_another_task(ready_connection, monkeypatch):
    _compiled(ready_connection, monkeypatch)
    with pytest.raises(ValueError, match="已存在"):
        execute(
            compile.run,
            context(
                ready_connection,
                {
                    "datasets": ["hotpotqa"],
                    "qa_limit": 1,
                    "run_id": "r1",
                },
            ),
        )
    assert len(compile_store.list_compile_runs(ready_connection)) == 1


def test_compile_resume_keeps_resolved_defaults(ready_connection, monkeypatch):
    clients = []

    def factory(config):
        client = FakeClient(config)
        clients.append(client)
        return client

    monkeypatch.setattr(compile, "AkashaClient", factory)
    monkeypatch.setattr(compile, "default_seed", lambda: 7)
    ctx = context(ready_connection, {"datasets": ["hotpotqa"], "qa_limit": 2})
    execute(compile.run, ctx)
    original = task_store.get_task(ready_connection, ctx.task_id)
    monkeypatch.setattr(compile, "default_seed", lambda: 99)
    resumed = context(ready_connection, {"seed": 99, "qa_limit": 1}, task_id=ctx.task_id)
    execute(compile.run, resumed)
    task = task_store.get_task(ready_connection, ctx.task_id)
    assert task["target_id"] == original["target_id"]
    assert task["params"] == original["params"]
    assert task["params"]["seed"] == 7
    assert clients[-1].imported == []


def test_deleted_run_cannot_be_recreated_by_resume(ready_connection, monkeypatch):
    compile_id = _compiled(ready_connection, monkeypatch)
    compile_store.delete_compile_run(ready_connection, compile_id)
    ready_connection.commit()
    with pytest.raises(ValueError, match="已被清理"):
        execute(compile.run, context(ready_connection, {}, task_id=1))
    assert compile_store.list_compile_runs(ready_connection) == []


def test_unnamed_queries_create_independent_runs(ready_connection, monkeypatch):
    compile_id = _compiled(ready_connection, monkeypatch)
    monkeypatch.setattr(query, "AkashaClient", lambda config: FakeClient(config))
    for _ in range(2):
        execute(query.run, context(ready_connection, {"compile_id": compile_id}))
    runs = query_store.list_query_runs(ready_connection, compile_id)
    assert len(runs) == 2
    assert len({run["name"] for run in runs}) == 2
