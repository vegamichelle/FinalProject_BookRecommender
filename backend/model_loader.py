"""Load the recommender from either a local file or the W&B Model Registry.

Controlled by MODEL_SOURCE:
  * "local"  -> joblib.load(MODEL_LOCAL_PATH)
  * "wandb"  -> download {WANDB_MODEL_NAME}:{WANDB_MODEL_ALIAS} from the registry

Returns a dict: {"recommender": Recommender, "metrics": {...}}.
"""
from __future__ import annotations

import os

import joblib

from src import config


def _load_local() -> dict:
    if not os.path.exists(config.MODEL_LOCAL_PATH):
        raise FileNotFoundError(
            f"Model file {config.MODEL_LOCAL_PATH} missing. Train first: "
            f"python -m src.train --no-wandb"
        )
    return joblib.load(config.MODEL_LOCAL_PATH)


def _load_wandb() -> dict:
    import wandb

    api = wandb.Api()
    entity = config.WANDB_ENTITY or api.default_entity
    artifact = api.artifact(
        f"{entity}/model-registry/{config.WANDB_MODEL_NAME}:{config.WANDB_MODEL_ALIAS}"
    )
    download_dir = artifact.download()
    for fname in os.listdir(download_dir):
        if fname.endswith(".joblib"):
            return joblib.load(os.path.join(download_dir, fname))
    raise RuntimeError("No .joblib file found in the downloaded W&B artifact.")


def load_model() -> dict:
    if config.MODEL_SOURCE == "wandb":
        return _load_wandb()
    return _load_local()
