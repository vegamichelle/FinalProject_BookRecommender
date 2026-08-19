"""The recommendation model — a self-contained, picklable artifact.

Strategy (in priority order at serve time):
  1. Item-item collaborative filtering: books that co-occur in users' liked
     sets. Scores candidates by summed similarity to the user's favourites.
  2. Content-based fallback (TF-IDF over title+author+genre): used when none
     of the favourite titles are in the CF catalogue (cold-start item).
  3. Popularity fallback: used when we have no usable signal at all.

`fit` builds everything from a books catalogue + a ratings frame; `recommend`
takes a list of favourite titles and returns ranked (book_id, title, score).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.features import build_interaction_matrix, normalize_title


class Recommender:
    def __init__(self, top_k: int = 50, min_interactions: int = 3) -> None:
        self.top_k = top_k
        self.min_interactions = min_interactions
        # Populated by fit().
        self.items: list = []
        self.item_pos: dict = {}
        self.id_to_title: dict = {}
        self.norm_title_to_id: dict = {}
        self.id_to_meta: dict = {}
        self.sim_neighbours: dict = {}  # book_id -> [(neighbour_id, score)]
        self.popularity: dict = {}       # book_id -> like count
        self._tfidf: TfidfVectorizer | None = None
        self._tfidf_ids: list = []
        self._tfidf_matrix = None

    # ------------------------------------------------------------------ fit
    def fit(self, books: pd.DataFrame, ratings: pd.DataFrame) -> Recommender:
        # Metadata lookups (built from the full catalogue).
        self.id_to_title = dict(zip(books["book_id"], books["title"]))
        self.norm_title_to_id = {
            normalize_title(t): bid
            for bid, t in zip(books["book_id"], books["title"])
        }
        self.id_to_meta = {
            row["book_id"]: {
                "title": row["title"],
                "author": row.get("author", ""),
                "genre": row.get("genre", ""),
            }
            for _, row in books.iterrows()
        }

        # --- Collaborative filtering ---
        mat, _users, items = build_interaction_matrix(ratings)
        counts = np.asarray(mat.sum(axis=0)).ravel()
        keep = counts >= self.min_interactions
        items = [it for it, k in zip(items, keep) if k]
        mat = mat[:, keep]

        self.items = items
        self.item_pos = {it: j for j, it in enumerate(items)}
        self.popularity = {
            it: float(c) for it, c in zip(items, counts[keep])
        }

        # Cosine similarity between items (columns). item_norm: items x users.
        item_user = mat.T.tocsr().astype(np.float32)
        sim = cosine_similarity(item_user, dense_output=False)  # items x items

        # Keep only the top_k neighbours per item (excluding self).
        self.sim_neighbours = {}
        sim = sim.tolil()
        for j, book_id in enumerate(items):
            row = sim.rows[j]
            vals = sim.data[j]
            pairs = [
                (items[c], float(v))
                for c, v in zip(row, vals)
                if c != j and v > 0
            ]
            pairs.sort(key=lambda p: p[1], reverse=True)
            self.sim_neighbours[book_id] = pairs[: self.top_k]

        # --- Content-based index (over the whole catalogue) ---
        corpus = (
            books["title"].fillna("")
            + " "
            + books.get("author", pd.Series([""] * len(books))).fillna("")
            + " "
            + books.get("genre", pd.Series([""] * len(books))).fillna("")
        ).tolist()
        self._tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        self._tfidf_matrix = self._tfidf.fit_transform(corpus)
        self._tfidf_ids = books["book_id"].tolist()

        return self

    # ------------------------------------------------------------ recommend
    def _resolve_titles(self, favourite_titles: list[str]) -> list:
        ids = []
        for t in favourite_titles:
            bid = self.norm_title_to_id.get(normalize_title(t))
            if bid is not None:
                ids.append(bid)
        return ids

    def _cf_scores(self, liked_ids: list) -> dict:
        scores: dict = {}
        liked_in_cf = [b for b in liked_ids if b in self.sim_neighbours]
        for b in liked_in_cf:
            for neigh, s in self.sim_neighbours[b]:
                scores[neigh] = scores.get(neigh, 0.0) + s
        return scores

    def _content_scores(self, favourite_titles: list[str]) -> dict:
        if self._tfidf is None:
            return {}
        query = self._tfidf.transform([" ".join(favourite_titles)])
        sims = cosine_similarity(query, self._tfidf_matrix).ravel()
        return {
            self._tfidf_ids[i]: float(sims[i])
            for i in np.argsort(sims)[::-1][: self.top_k]
            if sims[i] > 0
        }

    def recommend(
        self, favourite_titles: list[str], n: int | None = None
    ) -> list[dict]:
        n = n or 10
        liked_ids = self._resolve_titles(favourite_titles)

        scores = self._cf_scores(liked_ids)
        source = "collaborative"
        if not scores:
            scores = self._content_scores(favourite_titles)
            source = "content"
        if not scores:
            scores = dict(self.popularity)  # cold-start: most popular
            source = "popularity"

        # Never recommend the books the user already listed.
        for b in liked_ids:
            scores.pop(b, None)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [
            {
                "book_id": bid,
                "title": self.id_to_title.get(bid, str(bid)),
                "author": self.id_to_meta.get(bid, {}).get("author", ""),
                "genre": self.id_to_meta.get(bid, {}).get("genre", ""),
                "score": round(float(score), 4),
                "source": source,
            }
            for bid, score in ranked
        ]

    def sample_titles(self, n: int = 30) -> list[str]:
        """A few catalogue titles, most-popular first, for the UI dropdown."""
        popular = sorted(
            self.popularity.items(), key=lambda kv: kv[1], reverse=True
        )
        return [self.id_to_title[b] for b, _ in popular[:n]]
