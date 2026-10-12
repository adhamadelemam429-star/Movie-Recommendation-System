from __future__ import annotations

from typing import Dict

import pandas as pd


def validate_movie_rating_integrity(movies: pd.DataFrame, ratings: pd.DataFrame) -> Dict[str, object]:
    """Check the integrity of movie and rating tables."""
    movie_ids_valid = set(ratings["movieId"].dropna().astype(int).tolist()).issubset(set(movies["movieId"].dropna().astype(int).tolist()))
    rating_min = ratings["rating"].min() if not ratings.empty else 0.0
    rating_max = ratings["rating"].max() if not ratings.empty else 0.0
    rating_range_valid = rating_min >= 0.5 and rating_max <= 5.0

    missing_movies = {
        column: int(movies[column].isna().sum()) if column in movies.columns else 0
        for column in ["movieId", "title", "genres"]
    }
    missing_ratings = {
        column: int(ratings[column].isna().sum()) if column in ratings.columns else 0
        for column in ["userId", "movieId", "rating", "timestamp"]
    }

    duplicates = {
        "movies": int(movies.duplicated(subset=["movieId"]).sum()),
        "ratings": int(ratings.duplicated(subset=["userId", "movieId"]).sum()),
    }

    return {
        "movie_ids_valid": bool(movie_ids_valid),
        "rating_range_valid": bool(rating_range_valid),
        "missing_values": sum(missing_movies.values()) + sum(missing_ratings.values()),
        "missing_movies": missing_movies,
        "missing_ratings": missing_ratings,
        "duplicates": duplicates,
        "rating_min": float(rating_min),
        "rating_max": float(rating_max),
    }


def preprocess_movies_and_ratings(movies: pd.DataFrame, ratings: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Clean the movie and rating tables before recommendation training."""
    movies_clean = movies.copy()
    ratings_clean = ratings.copy()

    movies_clean["movieId"] = pd.to_numeric(movies_clean["movieId"], errors="coerce").astype("Int64")
    movies_clean["title"] = movies_clean["title"].fillna("Unknown Title").astype(str)
    movies_clean["genres"] = movies_clean["genres"].fillna("Unknown").astype(str)
    movies_clean["year"] = pd.to_numeric(movies_clean.get("year", pd.Series([None] * len(movies_clean))), errors="coerce")
    movies_clean = movies_clean.dropna(subset=["movieId"]).drop_duplicates(subset=["movieId"], keep="first").reset_index(drop=True)

    ratings_clean["userId"] = pd.to_numeric(ratings_clean["userId"], errors="coerce")
    ratings_clean["movieId"] = pd.to_numeric(ratings_clean["movieId"], errors="coerce")
    ratings_clean["rating"] = pd.to_numeric(ratings_clean["rating"], errors="coerce")
    ratings_clean["timestamp"] = pd.to_numeric(ratings_clean.get("timestamp", 0), errors="coerce")
    ratings_clean = ratings_clean.dropna(subset=["userId", "movieId", "rating"]).reset_index(drop=True)
    ratings_clean = ratings_clean[ratings_clean["rating"].between(0.5, 5.0)].copy()
    ratings_clean = ratings_clean.drop_duplicates(subset=["userId", "movieId"], keep="first").reset_index(drop=True)

    return {"movies": movies_clean, "ratings": ratings_clean}
