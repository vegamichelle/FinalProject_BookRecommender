"""Persistence for recommendation logs and the per-user recommendation cache.

Two logical stores:
  * LOGS  (DDB_LOGS_TABLE): append-only record of every /recommend request —
    request_id, timestamp, latency_ms, input titles, returned book ids, the
    recommendation source (collaborative/content/popularity), whether it was a
    cache hit, and any later feedback. The monitoring dashboard reads this.
  * CACHE (DDB_CACHE_TABLE): user_id -> last recommendations, so repeat
    requests for the same user are served from the store instead of recomputed
    (the caching pattern called out in the project spec).

For local dev (`DDB_LOCAL=1`) both are backed by JSON files exposing the same
interface, so the whole app runs with no AWS setup.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from src import config

_LOGS_FILE = os.environ.get("DDB_LOGS_FILE", "artifacts/local_logs.json")
_CACHE_FILE = os.environ.get("DDB_CACHE_FILE", "artifacts/local_cache.json")
_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# --------------------------------------------------------------------------- #
# Local JSON-backed stores (development / CI)
# --------------------------------------------------------------------------- #
class _LocalBackend:
    def __init__(self, logs_file: str, cache_file: str) -> None:
        self.logs_file = logs_file
        self.cache_file = cache_file
        os.makedirs(os.path.dirname(logs_file) or ".", exist_ok=True)
        for f in (logs_file, cache_file):
            if not os.path.exists(f):
                with open(f, "w") as fh:
                    json.dump({} if f == cache_file else [], fh)

    def put_log(self, item: dict) -> None:
        with _lock:
            with open(self.logs_file) as f:
                rows = json.load(f)
            rows.append(item)
            with open(self.logs_file, "w") as f:
                json.dump(rows, f, default=str)

    def update_log_feedback(self, request_id: str, feedback: int) -> bool:
        with _lock:
            with open(self.logs_file) as f:
                rows = json.load(f)
            found = False
            for r in rows:
                if r["request_id"] == request_id:
                    r["feedback"] = feedback
                    found = True
            if found:
                with open(self.logs_file, "w") as f:
                    json.dump(rows, f, default=str)
            return found

    def scan_logs(self) -> list[dict]:
        with open(self.logs_file) as f:
            return json.load(f)

    def get_cache(self, user_id: str) -> list | None:
        with open(self.cache_file) as f:
            cache = json.load(f)
        return cache.get(user_id)

    def put_cache(self, user_id: str, recs: list) -> None:
        with _lock:
            with open(self.cache_file) as f:
                cache = json.load(f)
            cache[user_id] = recs
            with open(self.cache_file, "w") as f:
                json.dump(cache, f, default=str)


# --------------------------------------------------------------------------- #
# DynamoDB backend (production)
# --------------------------------------------------------------------------- #
class _DynamoBackend:
    def __init__(self) -> None:
        import boto3

        kwargs: dict[str, Any] = {"region_name": config.AWS_REGION}
        if config.DDB_ENDPOINT_URL:
            kwargs["endpoint_url"] = config.DDB_ENDPOINT_URL
        resource = boto3.resource("dynamodb", **kwargs)
        self.logs = resource.Table(config.DDB_LOGS_TABLE)
        self.cache = resource.Table(config.DDB_CACHE_TABLE)

    @staticmethod
    def _to_ddb(obj: Any) -> Any:
        if isinstance(obj, float):
            return Decimal(str(obj))
        if isinstance(obj, dict):
            return {k: _DynamoBackend._to_ddb(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_DynamoBackend._to_ddb(v) for v in obj]
        return obj

    @staticmethod
    def _from_ddb(obj: Any) -> Any:
        if isinstance(obj, Decimal):
            f = float(obj)
            return int(f) if f.is_integer() else f
        if isinstance(obj, dict):
            return {k: _DynamoBackend._from_ddb(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_DynamoBackend._from_ddb(v) for v in obj]
        return obj

    def put_log(self, item: dict) -> None:
        self.logs.put_item(Item=self._to_ddb(item))

    def update_log_feedback(self, request_id: str, feedback: int) -> bool:
        self.logs.update_item(
            Key={"request_id": request_id},
            UpdateExpression="SET feedback = :f",
            ExpressionAttributeValues={":f": int(feedback)},
        )
        return True

    def scan_logs(self) -> list[dict]:
        items: list[dict] = []
        resp = self.logs.scan()
        items.extend(resp.get("Items", []))
        while "LastEvaluatedKey" in resp:
            resp = self.logs.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
            items.extend(resp.get("Items", []))
        return [self._from_ddb(i) for i in items]

    def get_cache(self, user_id: str) -> list | None:
        resp = self.cache.get_item(Key={"user_id": user_id})
        item = resp.get("Item")
        return self._from_ddb(item["recommendations"]) if item else None

    def put_cache(self, user_id: str, recs: list) -> None:
        self.cache.put_item(
            Item=self._to_ddb(
                {"user_id": user_id, "recommendations": recs, "cached_at": _now_iso()}
            )
        )


_backend: _LocalBackend | _DynamoBackend | None = None


def _get_backend() -> _LocalBackend | _DynamoBackend:
    global _backend
    if _backend is None:
        _backend = (
            _LocalBackend(_LOGS_FILE, _CACHE_FILE)
            if config.DDB_LOCAL
            else _DynamoBackend()
        )
    return _backend


# --------------------------------------------------------------------------- #
# Public helpers used by the backend + dashboard
# --------------------------------------------------------------------------- #
def log_request(
    input_titles: list[str],
    recommendations: list[dict],
    latency_ms: float,
    source: str,
    cache_hit: bool,
    user_id: str | None,
) -> str:
    request_id = str(uuid.uuid4())
    _get_backend().put_log(
        {
            "request_id": request_id,
            "timestamp": _now_iso(),
            "user_id": user_id or "anonymous",
            "latency_ms": float(latency_ms),
            "input_titles": input_titles,
            "recommended_ids": [r["book_id"] for r in recommendations],
            "recommended_genres": [r.get("genre", "") for r in recommendations],
            "source": source,
            "cache_hit": cache_hit,
        }
    )
    return request_id


def record_feedback(request_id: str, feedback: int) -> bool:
    """feedback: 1 (helpful) or 0 (not helpful)."""
    return _get_backend().update_log_feedback(request_id, int(feedback))


def fetch_logs() -> list[dict]:
    return _get_backend().scan_logs()


def get_cached(user_id: str) -> list | None:
    return _get_backend().get_cache(user_id)


def set_cached(user_id: str, recommendations: list) -> None:
    _get_backend().put_cache(user_id, recommendations)
