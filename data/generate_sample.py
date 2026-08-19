"""Generate a synthetic *books catalogue* + *user ratings* with real signal.

The real project uses the Amazon Review Data (Books subset); those files are
large, so for local dev / CI we synthesise a dataset with the same shape. Users
have latent genre preferences and rate books accordingly, so genre- and
author-correlated co-occurrence emerges and item-item CF actually learns
something (leave-one-out hit-rate lands well above random).

Outputs two CSVs:
    data/books.csv    : book_id,title,author,genre
    data/ratings.csv  : user_id,book_id,rating

Usage:
    python data/generate_sample.py --n-books 400 --n-users 3000
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

GENRES = [
    "Fantasy", "SciFi", "Mystery", "Romance",
    "Thriller", "History", "Biography", "Horror",
]
ADJ = [
    "Silent", "Crimson", "Broken", "Hidden", "Last", "Golden", "Frozen",
    "Burning", "Secret", "Lost", "Distant", "Shattered", "Wandering", "Pale",
    "Endless", "Forgotten", "Savage", "Gentle", "Restless", "Iron",
]
NOUN = [
    "Forest", "Crown", "River", "Empire", "Shadow", "Garden", "Storm",
    "Harbor", "Kingdom", "Mirror", "Ember", "Throne", "Wolf", "Lantern",
    "Cathedral", "Machine", "Orbit", "Cipher", "Meadow", "Requiem",
]


def _make_catalogue(n_books: int, rng: np.random.Generator) -> pd.DataFrame:
    titles: set[str] = set()
    rows = []
    authors = [f"Author {chr(65 + i % 26)}{i // 26}" for i in range(60)]
    book_id = 0
    while len(rows) < n_books:
        genre = GENRES[book_id % len(GENRES)]
        title = f"The {rng.choice(ADJ)} {rng.choice(NOUN)}"
        if title in titles:
            # add a numeral to keep titles unique
            title = f"{title} {len(titles)}"
        titles.add(title)
        rows.append(
            {
                "book_id": book_id,
                "title": title,
                "author": rng.choice(authors),
                "genre": genre,
            }
        )
        book_id += 1
    return pd.DataFrame(rows)


def _make_ratings(
    books: pd.DataFrame, n_users: int, rng: np.random.Generator
) -> pd.DataFrame:
    by_genre = {g: books[books["genre"] == g]["book_id"].to_numpy() for g in GENRES}
    records = []
    for user_id in range(n_users):
        # Each user favours 1-2 genres.
        n_fav = rng.integers(1, 3)
        fav_genres = rng.choice(GENRES, size=n_fav, replace=False)
        n_rated = rng.integers(5, 25)
        for _ in range(n_rated):
            if rng.random() < 0.8:  # mostly rate within favourite genres
                g = rng.choice(fav_genres)
                pool = by_genre[g]
                base = 4.0
            else:  # some exploration
                g = rng.choice(GENRES)
                pool = by_genre[g]
                base = 3.0
            book_id = int(rng.choice(pool))
            rating = float(np.clip(round(base + rng.normal(0, 0.8)), 1, 5))
            records.append({"user_id": user_id, "book_id": book_id, "rating": rating})
    df = pd.DataFrame(records).drop_duplicates(subset=["user_id", "book_id"])
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-books", type=int, default=400)
    ap.add_argument("--n-users", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    books = _make_catalogue(args.n_books, rng)
    ratings = _make_ratings(books, args.n_users, rng)

    os.makedirs(args.outdir, exist_ok=True)
    books.to_csv(os.path.join(args.outdir, "books.csv"), index=False)
    ratings.to_csv(os.path.join(args.outdir, "ratings.csv"), index=False)
    print(
        f"Wrote {len(books):,} books and {len(ratings):,} ratings "
        f"({ratings['user_id'].nunique():,} users) to {args.outdir}/"
    )


if __name__ == "__main__":
    main()
