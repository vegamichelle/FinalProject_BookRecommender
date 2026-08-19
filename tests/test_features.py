"""Unit tests for pure preprocessing helpers and the recommender logic."""
import pandas as pd

from src.features import (
    build_interaction_matrix,
    hit_at_k,
    normalize_title,
    precision_at_k,
)
from src.recommender import Recommender


# --- normalize_title -------------------------------------------------------
def test_normalize_lowercases_and_strips_punct():
    assert normalize_title("The Silent  Forest!") == "the silent forest"


def test_normalize_handles_none_and_empty():
    assert normalize_title(None) == ""
    assert normalize_title("   ") == ""


def test_normalize_is_idempotent():
    once = normalize_title("Shadows: of the North?")
    assert normalize_title(once) == once


# --- metrics ---------------------------------------------------------------
def test_precision_at_k():
    assert precision_at_k([1, 2, 3, 4], {2, 4}, 4) == 0.5
    assert precision_at_k([1, 2, 3], {9}, 3) == 0.0
    assert precision_at_k([], {1}, 3) == 0.0


def test_hit_at_k():
    assert hit_at_k([5, 6, 7], {7}, 3) == 1
    assert hit_at_k([5, 6, 7], {7}, 2) == 0
    assert hit_at_k([5, 6, 7], {99}, 3) == 0


# --- interaction matrix ----------------------------------------------------
def test_build_interaction_matrix_thresholds_and_shape():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "book_id": [10, 20, 10, 30, 20],
            "rating": [5.0, 2.0, 4.0, 5.0, 4.0],  # rating 2.0 dropped
        }
    )
    mat, users, items = build_interaction_matrix(ratings, like_threshold=4.0)
    # user 1 only liked book 10 (20 was rated 2.0)
    assert users == [1, 2, 3]
    assert items == [10, 20, 30]
    assert mat.shape == (3, 3)
    assert mat.sum() == 4  # four liked interactions survive the threshold


# --- recommender end to end (tiny in-memory dataset) -----------------------
def _tiny_dataset():
    books = pd.DataFrame(
        {
            "book_id": [1, 2, 3, 4],
            "title": ["Alpha", "Beta", "Gamma", "Delta"],
            "author": ["A", "A", "B", "B"],
            "genre": ["Fantasy", "Fantasy", "SciFi", "SciFi"],
        }
    )
    # Users who like Alpha also like Beta; Gamma-lovers also like Delta.
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "book_id": [1, 2, 1, 2, 3, 4, 3, 4],
            "rating": [5, 5, 5, 5, 5, 5, 5, 5],
        }
    )
    return books, ratings


def test_recommender_collaborative_pairing():
    books, ratings = _tiny_dataset()
    rec = Recommender(top_k=10, min_interactions=1).fit(books, ratings)
    recs = rec.recommend(["Alpha"], n=3)
    # Beta co-occurs with Alpha, so it should be the top recommendation.
    assert recs[0]["title"] == "Beta"
    assert recs[0]["source"] == "collaborative"
    # Alpha itself must never be recommended back.
    assert all(r["title"] != "Alpha" for r in recs)


def test_recommender_cold_start_falls_back():
    books, ratings = _tiny_dataset()
    rec = Recommender(top_k=10, min_interactions=1).fit(books, ratings)
    recs = rec.recommend(["Totally Unknown Title xyzzy"], n=2)
    assert len(recs) >= 1
    assert recs[0]["source"] in ("content", "popularity")


def test_recommender_respects_n():
    books, ratings = _tiny_dataset()
    rec = Recommender(top_k=10, min_interactions=1).fit(books, ratings)
    assert len(rec.recommend(["Gamma"], n=1)) == 1
