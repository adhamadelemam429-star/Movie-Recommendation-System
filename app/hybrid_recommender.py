from __future__ import annotations

import pandas as pd

from app.content_based import get_similar_movies
from app.collaborative_filtering import recommend_for_user


def hybrid_recommend(user_id: int, movies: pd.DataFrame, ratings: pd.DataFrame, top_n: int = 5, content_weight: float = 0.5, collaborative_weight: float = 0.5) -> pd.DataFrame:
    """Combine content-based and collaborative scores into a single recommendation list."""
    if movies.empty or ratings.empty:
        return pd.DataFrame(columns=["movieId", "title", "score", "method"])

    user_rated = ratings[ratings["userId"] == user_id]["movieId"].astype(int).tolist()
    user_rated_movies = movies[movies["movieId"].isin(user_rated)]
    content_scores = []

    for movie_id in user_rated_movies["movieId"].astype(int).tolist():
        similar = get_similar_movies(int(movie_id), movies, top_n=10)
        for _, row in similar.iterrows():
            content_scores.append({"movieId": int(row["movieId"]), "score": float(row["score"])})

    content_df = pd.DataFrame(content_scores).groupby("movieId", as_index=False)["score"].max()
    content_df.rename(columns={"score": "content_score"}, inplace=True)

    collaborative = recommend_for_user(user_id, ratings, movies, top_n=max(top_n * 3, 10))
    collaborative_df = collaborative.rename(columns={"score": "collaborative_score"}) if not collaborative.empty else pd.DataFrame(columns=["movieId", "collaborative_score"])

    combined = collaborative_df.merge(content_df, on="movieId", how="outer") if not collaborative_df.empty else content_df
    if combined.empty:
        return pd.DataFrame(columns=["movieId", "title", "score", "method"])

    if "collaborative_score" not in combined.columns:
        combined["collaborative_score"] = 0.0
    if "content_score" not in combined.columns:
        combined["content_score"] = 0.0

    combined["score"] = (
        content_weight * combined["content_score"].fillna(0.0)
        + collaborative_weight * combined["collaborative_score"].fillna(0.0)
    )
    combined = combined.merge(movies[["movieId", "title"]], on="movieId", how="left", suffixes=("_x", "_y"))
    if "title_x" in combined.columns and "title_y" in combined.columns:
        combined["title"] = combined["title_y"].fillna(combined["title_x"])
        combined = combined.drop(columns=["title_x", "title_y"])
    elif "title_x" in combined.columns:
        combined = combined.rename(columns={"title_x": "title"})
    combined = combined.sort_values("score", ascending=False).head(top_n)
    combined["method"] = "hybrid"
    return combined[["movieId", "title", "score", "method"]].reset_index(drop=True)

