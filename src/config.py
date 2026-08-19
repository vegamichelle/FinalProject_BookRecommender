"""Central configuration, read from environment variables.

Every component (training, backend, dashboard) imports from here so behaviour
is controlled entirely by env vars / the .env file — no code edits needed to
move from local dev to production on AWS.
"""
from __future__ import annotations

import os


def _get(key: str, default: str) -> str:
    return os.environ.get(key, default)


# --- AWS / DynamoDB ---------------------------------------------------------
AWS_REGION: str = _get("AWS_REGION", "us-east-1")
# Two tables: one append-only log for monitoring, one point-lookup rec cache.
DDB_LOGS_TABLE: str = _get("DDB_LOGS_TABLE", "book-rec-logs")
DDB_CACHE_TABLE: str = _get("DDB_CACHE_TABLE", "book-rec-cache")
# When "1", use a local JSON-backed store so the app runs with no AWS setup.
DDB_LOCAL: bool = _get("DDB_LOCAL", "1") == "1"
DDB_ENDPOINT_URL: str | None = os.environ.get("DDB_ENDPOINT_URL") or None

# Whether the backend should serve cached recommendations for repeat users.
CACHE_ENABLED: bool = _get("CACHE_ENABLED", "1") == "1"

# --- Model source -----------------------------------------------------------
# "local"  -> load a joblib file from MODEL_LOCAL_PATH (default, no creds)
# "wandb"  -> pull the Production model from the W&B Model Registry
MODEL_SOURCE: str = _get("MODEL_SOURCE", "local")
MODEL_LOCAL_PATH: str = _get("MODEL_LOCAL_PATH", "artifacts/recommender.joblib")

# --- Weights & Biases -------------------------------------------------------
WANDB_PROJECT: str = _get("WANDB_PROJECT", "mlops-book-recommender")
WANDB_ENTITY: str | None = os.environ.get("WANDB_ENTITY") or None
WANDB_MODEL_NAME: str = _get("WANDB_MODEL_NAME", "book-recommender")
WANDB_MODEL_ALIAS: str = _get("WANDB_MODEL_ALIAS", "production")

# --- Recommender defaults ---------------------------------------------------
DEFAULT_TOP_N: int = int(_get("DEFAULT_TOP_N", "10"))
SIM_TOP_K: int = int(_get("SIM_TOP_K", "50"))  # neighbours stored per item
MIN_INTERACTIONS: int = int(_get("MIN_INTERACTIONS", "3"))  # drop rare books

# --- Frontend ---------------------------------------------------------------
BACKEND_URL: str = _get("BACKEND_URL", "http://localhost:8000")
