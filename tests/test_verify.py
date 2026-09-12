"""平台小样本验证复用真实阶段；仅 Akasha HTTP 使用替身。"""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from akasha_benchmark import ingest, run_queries
from akasha_benchmark.config import AkashaConfig
from akasha_benchmark.store import connect, repo
from akasha_benchmark.store.migrate import migrate
from akasha_platform import tasks, verify
from akasha_platform.main import create_app
from akasha_platform.settings import Settings
from test_client_and_stages import FakeAkasha, _patch_client, enveloped
from test_subset_and_io import _seed_normalized


def response():
    return {
        "answer": "answer",
        "answerMode": "knowledge",
        "warnings": [],
        "retrievedSources": [{"sourcePageId": "page-1"}],
        "citations": [{"sourcePageId": "page-1"}],
        "citationEvidence": [{"sourcePageId": "page-1", "excerpts": ["body"]}],
        "snippets": [],
        "retrievalReasons": [],
        "budget": {
            "maxContextLength": 12000,
            "perItemMaxLength": 12000,
            "responseReserve": 0,
            "usedContextLength": 4,
            "includedItemCount": 1,
            "omittedItemCount": 0,
        },
    }


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "verify.db"
    migrate(path, verbose=False)
    c = connect(path)
    _seed_normalized(
        c,
        "hotpotqa",
        [
            {
                "dataset": "hotpotqa",
                "sample_id": f"hotpotqa:{i}",
                "dataset_sample_id": str(i),
                "question": f"question {i}",
                "answers": ["answer"],
                "gold_doc_ids": [str(i)],
                "metadata": {"type": "bridge", "gold_count": 1},
            }
            for i in range(3)
        ],
        [{"doc_id": str(i), "title": f"T{i}", "text": "body"} for i in range(6)],
    )
    repo.update_connection(c, email="test@example.com", password="test")
    c.commit()
    c.close()
    return path


@pytest.mark.parametrize("broken", [False, True], ids=["success", "invalid-response"])
def test_platform_verification_pipeline(database, monkeypatch, broken):
    class Server(FakeAkasha):
        def handler(self, request):
            if request.url.path.endswith("/query"):
                self.queries.append({})
                body = response()
                if broken:
                    body["citations"] = [{"sourcePageId": "unknown-page"}]
                return enveloped(body)
            return super().handler(request)

    server = Server()
    config = AkashaConfig(
        email="test@example.com", password="test", request_interval_seconds=0
    )
    for module in (ingest, run_queries, verify):
        _patch_client(monkeypatch, module, server, config)
    code = verify.main(["--db", str(database), "--samples", "2"])
    assert code == (1 if broken else 0)
    c = connect(database)
    try:
        assert len(repo.list_index_layers(c)) == 1
        assert len(server.imported) == 4  # 2 gold + 2 negatives, no duplicate import
        assert len(server.queries) == 2  # query resume sends no extra requests
        layers = repo.list_eval_layers(c)
        assert len(layers) == (0 if broken else 1)
        if not broken:
            assert len(repo.sample_evals(c, layers[0]["id"])) == 2
    finally:
        c.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("answerMode", "unknown"),
        ("budget", {}),
        ("citations", []),
        ("snippets", [{"id": "not-a-uuid"}]),
        ("warnings", None),
    ],
)
def test_response_contract_rejects_malformed_fields(field, value):
    body = deepcopy(response())
    body[field] = value
    with pytest.raises(ValueError):
        verify.validate_response(body, {"page-1"})


def test_verify_stage_is_exposed_and_rejects_invalid_size(database, tmp_path):
    with TestClient(
        create_app(Settings(db_path=database))
    ) as api:
        assert any(s["stage"] == "verify" for s in api.get("/api/stages").json())
        assert api.post("/api/tasks/verify", json={"samples": 100}).status_code == 409


@pytest.mark.parametrize(
    "running,requested", [("verify", "ingest"), ("query", "verify")]
)
def test_verification_does_not_overlap_other_stages(database, running, requested):
    c = connect(database)
    task_id = repo.create_task(c, stage=running, argv=["test"])
    repo.start_task(c, task_id, pid=123, log_path=None)
    c.commit()
    c.close()
    with pytest.raises(tasks.TaskRejected, match="already running"):
        tasks.start(requested, {}, Settings(db_path=database))
