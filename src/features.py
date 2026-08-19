"""Pure, dependency-light preprocessing helpers for the recommender.

These functions are deliberately side-effect-free so they are simple to
unit-test. Both training and serving import them, so title matching and the
interaction matrix can never drift between train and serve time.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Lets free-text user input ("The Silent  Forest!") match a catalogue entry
    ("the silent forest") without an exact string match.
    """
    if title is None:
        return ""
    t = str(title).lower().strip()
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


def build_interaction_matrix(
    ratings: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "book_id",
    like_threshold: float = 4.0,
    rating_col: str = "rating",
) -> tuple[sparse.csr_matrix, list, list]:
    """Build a sparse implicit (users x items) binary "liked" matrix.

    A cell is 1 when the user's rating >= like_threshold. Returns the matrix
    plus the ordered lists of user ids and item ids that index its rows/cols.
    """
    liked = ratings[ratings[rating_col] >= like_threshold]
    users = sorted(liked[user_col].unique().tolist())
    items = sorted(liked[item_col].unique().tolist())
    u_index = {u: i for i, u in enumerate(users)}
    i_index = {it: j for j, it in enumerate(items)}

    rows = liked[user_col].map(u_index).to_numpy()
    cols = liked[item_col].map(i_index).to_numpy()
    data = np.ones(len(liked), dtype=np.float32)

    mat = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(users), len(items))
    )
    # De-duplicate repeated (user, item) likes -> keep binary.
    mat.data[:] = 1.0
    return mat, users, items


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of the top-k recommendations that are relevant."""
    if k <= 0:
        return 0.0
    topk = recommended[:k]
    if not topk:
        return 0.0
    hits = sum(1 for r in topk if r in relevant)
    return hits / len(topk)


def hit_at_k(recommended: list, relevant: set, k: int) -> int:
    """1 if any of the top-k recommendations is relevant, else 0."""
    return int(any(r in relevant for r in recommended[:k]))
