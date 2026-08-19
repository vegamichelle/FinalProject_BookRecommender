"""Train the book recommender with W&B tracking + registry.

Logs to W&B: git commit, hyperparameters, evaluation metrics
(hit_rate@k, precision@k, catalog coverage), and the data version. The fitted
Recommender is saved as an artifact and linked into the Model Registry.

Runs with or without W&B:
    python -m src.train                # uses W&B if logged in
    python -m src.train --no-wandb     # local only, still saves the artifact
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import build_interaction_matrix, hit_at_k, precision_at_k
from src.recommender import Recommender


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _data_version(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return f"{h.hexdigest()[:12]}-{len(df)}rows"


def evaluate(rec: Recommender, books: pd.DataFrame, ratings: pd.DataFrame,
             k: int, n_eval_users: int = 500, seed: int = 42) -> dict:
    """Leave-one-out evaluation.

    For each sampled user with >=2 liked books, hold out one liked book, use
    the rest as favourites, and check whether the held-out book appears in the
    top-k recommendations.
    """
    rng = np.random.default_rng(seed)
    _mat, users, _items = build_interaction_matrix(ratings)
    liked = ratings[ratings["rating"] >= 4.0]
    by_user = liked.groupby("user_id")["book_id"].apply(list)
    eligible = [u for u in users if len(by_user.get(u, [])) >= 2]
    if not eligible:
        return {"hit_rate_at_k": 0.0, "precision_at_k": 0.0, "coverage": 0.0}

    sample = rng.choice(eligible, size=min(n_eval_users, len(eligible)), replace=False)
    hits, precs = [], []
    recommended_catalog: set = set()
    id_to_title = dict(zip(books["book_id"], books["title"]))

    for u in sample:
        book_ids = list(by_user[u])
        held_out = int(rng.choice(book_ids))
        favourites = [id_to_title[b] for b in book_ids if b != held_out]
        recs = rec.recommend(favourites, n=k)
        rec_ids = [r["book_id"] for r in recs]
        recommended_catalog.update(rec_ids)
        relevant = {held_out}
        hits.append(hit_at_k(rec_ids, relevant, k))
        precs.append(precision_at_k(rec_ids, relevant, k))

    coverage = len(recommended_catalog) / max(len(rec.items), 1)
    return {
        "hit_rate_at_k": float(np.mean(hits)),
        "precision_at_k": float(np.mean(precs)),
        "coverage": float(coverage),
    }


def train(books_path: str, ratings_path: str, top_k: int, min_interactions: int,
          eval_k: int, use_wandb: bool) -> dict:
    if not (os.path.exists(books_path) and os.path.exists(ratings_path)):
        raise FileNotFoundError(
            "Missing data. Run: python data/generate_sample.py"
        )
    books = pd.read_csv(books_path)
    ratings = pd.read_csv(ratings_path)

    hyperparams = {
        "model": "item-item-CF + tfidf-fallback",
        "sim_top_k": top_k,
        "min_interactions": min_interactions,
        "eval_k": eval_k,
    }

    run = None
    if use_wandb:
        import wandb

        run = wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            config={
                **hyperparams,
                "git_commit": _git_commit(),
                "data_version": _data_version(ratings),
                "n_books": len(books),
                "n_ratings": len(ratings),
            },
        )

    rec = Recommender(top_k=top_k, min_interactions=min_interactions)
    rec.fit(books, ratings)

    metrics = evaluate(rec, books, ratings, k=eval_k)
    print(
        f"Metrics @ k={eval_k}: hit_rate={metrics['hit_rate_at_k']:.3f}  "
        f"precision={metrics['precision_at_k']:.3f}  "
        f"coverage={metrics['coverage']:.3f}"
    )

    os.makedirs(os.path.dirname(config.MODEL_LOCAL_PATH) or ".", exist_ok=True)
    joblib.dump({"recommender": rec, "metrics": metrics}, config.MODEL_LOCAL_PATH)
    print(f"Saved recommender to {config.MODEL_LOCAL_PATH}")

    if run is not None:
        import wandb

        wandb.log(metrics)
        art = wandb.Artifact(
            config.WANDB_MODEL_NAME, type="model", metadata={**hyperparams, **metrics}
        )
        art.add_file(config.MODEL_LOCAL_PATH)
        logged = run.log_artifact(art)
        logged.wait()
        try:
            run.link_artifact(
                logged,
                target_path=f"{config.WANDB_ENTITY or run.entity}/model-registry/"
                f"{config.WANDB_MODEL_NAME}",
                aliases=["staging"],
            )
            print("Linked artifact into Model Registry with alias 'staging'.")
        except Exception as exc:
            print(f"Registry link skipped: {exc}")
        wandb.finish()

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="data/books.csv")
    ap.add_argument("--ratings", default="data/ratings.csv")
    ap.add_argument("--top-k", type=int, default=config.SIM_TOP_K)
    ap.add_argument("--min-interactions", type=int, default=config.MIN_INTERACTIONS)
    ap.add_argument("--eval-k", type=int, default=10)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    env_disables = os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true")
    use_wandb = not args.no_wandb and not env_disables

    train(
        books_path=args.books,
        ratings_path=args.ratings,
        top_k=args.top_k,
        min_interactions=args.min_interactions,
        eval_k=args.eval_k,
        use_wandb=use_wandb,
    )


if __name__ == "__main__":
    main()
