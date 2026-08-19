"""Integration tests for the FastAPI backend using TestClient.

Run against the local JSON store (DDB_LOCAL=1) and a locally trained model, so
no AWS or W&B is needed in CI.
"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

# Isolate the local store in a fresh temp dir so tests never depend on
# leftover files from a previous run.
_tmp = tempfile.mkdtemp()
os.environ["DDB_LOCAL"] = "1"
os.environ["MODEL_SOURCE"] = "local"
os.environ["CACHE_ENABLED"] = "1"
os.environ["DDB_LOGS_FILE"] = os.path.join(_tmp, "logs.json")
os.environ["DDB_CACHE_FILE"] = os.path.join(_tmp, "cache.json")

from backend.main import app  # noqa: E402

client = TestClient(app)


def _a_valid_title() -> str:
    return client.get("/catalog", params={"n": 5}).json()["titles"][0]


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_catalog_returns_titles():
    r = client.get("/catalog", params={"n": 5})
    assert r.status_code == 200
    titles = r.json()["titles"]
    assert isinstance(titles, list) and len(titles) > 0


def test_recommend_returns_ranked_books():
    title = _a_valid_title()
    r = client.post("/recommend", json={"favorite_titles": [title], "n": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["recommendations"]) == 5
    assert "request_id" in body
    assert body["cache_hit"] is False
    # A user's own favourite must not be recommended back.
    assert all(rec["title"] != title for rec in body["recommendations"])


def test_recommend_requires_at_least_one_title():
    r = client.post("/recommend", json={"favorite_titles": [], "n": 5})
    assert r.status_code == 422


def test_cache_hit_on_repeat_user():
    title = _a_valid_title()
    payload = {"favorite_titles": [title], "n": 5, "user_id": "cache-test-user"}
    first = client.post("/recommend", json=payload).json()
    assert first["cache_hit"] is False
    second = client.post("/recommend", json=payload).json()
    assert second["cache_hit"] is True
    assert second["source"] == "cache"


def test_feedback_roundtrip():
    title = _a_valid_title()
    rec = client.post("/recommend", json={"favorite_titles": [title], "n": 3}).json()
    r = client.post(
        "/feedback", json={"request_id": rec["request_id"], "helpful": True}
    )
    assert r.status_code == 200
    assert r.json()["status"] == "recorded"


def test_feedback_unknown_id_404():
    r = client.post("/feedback", json={"request_id": "nope", "helpful": False})
    assert r.status_code == 404


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
