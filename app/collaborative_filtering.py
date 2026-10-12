from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Build a sparse-like user-item matrix using the ratings table."""
    return ratings.pivot_table(index="userId", columns="movieId", values="rating", aggfunc="mean").fillna(0.0)


def recommend_for_user(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Recommend unseen movies for an existing user using item-based collaborative filtering."""
    if ratings.empty or movies.empty:
        return pd.DataFrame(columns=["movieId", "title", "score"])

    rated_movies = set(ratings[ratings["userId"] == user_id]["movieId"].astype(int).tolist())
    all_movies = set(movies["movieId"].astype(int).tolist())
    unseen_movies = sorted(all_movies - rated_movies)

    if not unseen_movies:
        return pd.DataFrame(columns=["movieId", "title", "score"])

    user_ratings = ratings[ratings["userId"] == user_id].copy()
    if user_ratings.empty:
        return pd.DataFrame(columns=["movieId", "title", "score"])

    movie_matrix = ratings.pivot_table(index="movieId", columns="userId", values="rating", aggfunc="mean").fillna(0.0)
    similarity = cosine_similarity(movie_matrix.values)
    movie_index = {movie_id: idx for idx, movie_id in enumerate(movie_matrix.index.tolist())}
    user_scores = []

    for candidate_movie in unseen_movies:
        idx = movie_index.get(int(candidate_movie))
        if idx is None:
            continue
        candidate_similarities = similarity[idx]
        candidate_scores = []
        for rated_movie in user_ratings["movieId"].astype(int).tolist():
            rated_idx = movie_index.get(int(rated_movie))
            if rated_idx is None:
                continue
            weight = candidate_similarities[rated_idx]
            if abs(weight) < 1e-9:
                continue
            candidate_scores.append((weight, float(user_ratings[user_ratings["movieId"] == rated_movie]["rating"].iloc[0])))

        if not candidate_scores:
            continue
        weighted = sum(weight * rating for weight, rating in candidate_scores)
        denom = sum(abs(weight) for weight, _ in candidate_scores)
        score = weighted / denom if denom else 0.0
        user_scores.append({"movieId": int(candidate_movie), "score": float(score)})

    if not user_scores:
        return pd.DataFrame(columns=["movieId", "title", "score"])

    recommendations = pd.DataFrame(user_scores).sort_values("score", ascending=False).head(top_n)
    if recommendations.empty:
        return pd.DataFrame(columns=["movieId", "title", "score"])

    recommendations = recommendations.merge(movies[["movieId", "title"]], on="movieId", how="left")
    return recommendations[["movieId", "title", "score"]].reset_index(drop=True)

