from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _normalize_feature_text(row: pd.Series) -> str:
    title = str(row.get("title", "")).strip()
    genres = str(row.get("genres", "")).replace("|", " ")
    return f"{title} {genres}".strip()


def build_content_model(movies: pd.DataFrame) -> Dict[str, Any]:
    """Build a TF-IDF content model for movie titles and genres."""
    movies = movies.copy()
    movies["feature_text"] = movies.apply(_normalize_feature_text, axis=1)
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(movies["feature_text"])
    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "movie_ids": list(movies["movieId"].astype(int)),
        "movies": movies,
    }


def get_similar_movies(
    movie_id: int,
    movies: pd.DataFrame,
    model: Optional[Dict[str, Any]] = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return movies similar to the selected movie using TF-IDF and cosine similarity."""
    if movies.empty:
        return pd.DataFrame(columns=["movieId", "title", "score"])

    if model is None:
        model = build_content_model(movies)

    movie_ids = list(model["movies"]["movieId"].astype(int))
    if movie_id not in movie_ids:
        raise ValueError(f"Movie id {movie_id} was not found in the dataset.")

    movie_index = movie_ids.index(int(movie_id))
    similarities = cosine_similarity(model["matrix"][movie_index], model["matrix"])[0]
    ranking = sorted(enumerate(similarities), key=lambda item: item[1], reverse=True)

    recommended = []
    for idx, score in ranking:
        same_movie = int(model["movies"].iloc[idx]["movieId"]) == int(movie_id)
        if same_movie:
            continue
        recommended.append(
            {
                "movieId": int(model["movies"].iloc[idx]["movieId"]),
                "title": model["movies"].iloc[idx]["title"],
                "score": round(float(score), 4),
            }
        )
        if len(recommended) >= top_n:
            break

    return pd.DataFrame(recommended, columns=["movieId", "title", "score"])

