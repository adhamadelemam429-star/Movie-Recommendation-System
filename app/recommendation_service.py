from __future__ import annotations

import os
from typing import Iterable, List, Optional

import pandas as pd

from app.content_based import build_content_model, get_similar_movies
from app.collaborative_filtering import recommend_for_user
from app.data_loader import _ensure_poster_assets, ensure_sample_data, fetch_movie_lens_dataset
from app.hybrid_recommender import hybrid_recommend
from app.preprocessing import preprocess_movies_and_ratings


class RecommendationService:
    """High-level service that exposes a single API for recommendation tasks."""

    def __init__(self, movies: Optional[pd.DataFrame] = None, ratings: Optional[pd.DataFrame] = None):
        self.movies, self.ratings = self._prepare_data(movies, ratings)
        self.content_model = build_content_model(self.movies)

    def _prepare_data(self, movies: Optional[pd.DataFrame], ratings: Optional[pd.DataFrame]):
        if movies is None or ratings is None:
            dataset_mode = os.getenv("DATA_SOURCE_MODE", "sample").lower()
            dataset_name = os.getenv("MOVIELENS_DATASET", "ml-latest-small")
            if dataset_mode in {"movielens", "ml-latest", "ml-latest-small", "movie_lens"}:
                movies, ratings = fetch_movie_lens_dataset(dataset_name)
            else:
                movies, ratings = ensure_sample_data()
        cleaned = preprocess_movies_and_ratings(movies, ratings)
        cleaned_movies = _ensure_poster_assets(cleaned["movies"])
        return cleaned_movies, cleaned["ratings"]

    def search_movies(self, query: str, foreign_only: bool = False) -> List[dict]:
        query = (query or "").strip().lower()
        matches = self.movies.copy()
        if foreign_only:
            matches = matches[matches.get("is_foreign", False) == True]
        if query:
            matches = matches[matches["title"].str.lower().str.contains(query, na=False)]
        return [self.get_movie(int(row["movieId"])) for _, row in matches.head(10).iterrows()]

    def _recommendation_item(self, row: pd.Series, method: str, reason: str) -> dict:
        movie = self.get_movie(int(row["movieId"])) or {}
        return {**movie, "score": float(row["score"]), "method": method, "reason": reason}

    def recommend_similar_movies(self, movie_id: int, top_n: int = 5) -> List[dict]:
        try:
            similar = get_similar_movies(int(movie_id), self.movies, model=self.content_model, top_n=top_n)
        except ValueError:
            return []
        return [self._recommendation_item(row, "content_based", "Matched on shared title and genre signals.") for _, row in similar.iterrows()]

    def recommend_for_user(self, user_id: int, top_n: int = 5, method: str = "hybrid") -> List[dict]:
        user_id = int(user_id)
        if user_id not in set(self.ratings["userId"].astype(int).tolist()):
            return []

        if method == "content_based":
            watched = self.ratings[self.ratings["userId"] == user_id].sort_values("rating", ascending=False)
            if watched.empty:
                return []
            favorite = int(watched.iloc[0]["movieId"])
            similar = get_similar_movies(favorite, self.movies, model=self.content_model, top_n=top_n)
            return [self._recommendation_item(row, "content_based", "Based on the user's highest-rated movie and shared characteristics.") for _, row in similar.iterrows()]

        if method == "collaborative":
            recs = recommend_for_user(user_id, self.ratings, self.movies, top_n=top_n)
            return [self._recommendation_item(row, "collaborative", "Derived from ratings similarity among users.") for _, row in recs.iterrows()]

        if method == "hybrid":
            recs = hybrid_recommend(user_id, self.movies, self.ratings, top_n=top_n)
            return [self._recommendation_item(row, "hybrid", "Weighted mix of content similarity and collaborative signals.") for _, row in recs.iterrows()]

        return self.get_popular_movies(top_n=top_n)

    def recommend_by_genre(self, genres: Iterable[str], top_n: int = 5) -> List[dict]:
        targets = [genre.strip().lower() for genre in genres if genre]
        if not targets:
            return []

        filtered = self.movies[self.movies["genres"].str.lower().str.contains("|".join(targets), na=False)]
        popular = (
            self.ratings.merge(filtered[["movieId"]], on="movieId", how="inner")
            .groupby("movieId", as_index=False)["rating"]
            .mean()
            .merge(filtered[["movieId", "title", "genres"]], on="movieId", how="inner")
            .sort_values("rating", ascending=False)
            .head(top_n)
        )

        return [self._recommendation_item(row.rename({"rating": "score"}), "genre", "Selected for matching the requested genres.") for _, row in popular.iterrows()]

    def get_popular_movies(self, top_n: int = 5) -> List[dict]:
        popularity = (
            self.ratings.groupby("movieId", as_index=False)["rating"]
            .mean()
            .merge(self.movies, on="movieId", how="inner")
            .sort_values("rating", ascending=False)
            .head(top_n)
        )
        return [self._recommendation_item(row.rename({"rating": "score"}), "popularity", "Popular among users, but not necessarily the best or most personalized recommendation.") for _, row in popularity.iterrows()]

    def explain_recommendation(self, movie_id: int, method: str = "content_based") -> dict:
        movie = self.movies[self.movies["movieId"] == int(movie_id)]
        if movie.empty:
            return {"movieId": int(movie_id), "method": method, "reason": "Movie not found."}
        metadata = movie.iloc[0]
        if method == "content_based":
            reason = f"This recommendation matches the requested movie using title/genre overlap: {metadata['genres']}."
        elif method == "collaborative":
            reason = "This recommendation was inferred from users with similar rating patterns."
        elif method == "hybrid":
            reason = "This recommendation combines content similarity with collaborative rating patterns."
        else:
            reason = "This recommendation is based on overall popularity and average user preference."
        return {"movieId": int(movie_id), "title": metadata["title"], "method": method, "reason": reason}

    def get_movie(self, movie_id: int) -> Optional[dict]:
        match = self.movies[self.movies["movieId"] == int(movie_id)]
        if match.empty:
            return None
        row = match.iloc[0]
        year = row.get("year")
        runtime = row.get("runtime")
        vote_average = row.get("vote_average")
        return {
            "movieId": int(row["movieId"]),
            "title": row["title"],
            "genres": row["genres"],
            "year": int(year) if pd.notna(year) else None,
            "language": row.get("language", "Unknown"),
            "country": row.get("country", "Unknown"),
            "is_foreign": bool(row.get("is_foreign", False)),
            "poster_url": row.get("poster_url"),
            "backdrop_url": row.get("backdrop_url", row.get("poster_url")),
            "overview": row.get("overview", "No synopsis is available for this title."),
            "tagline": row.get("tagline", ""),
            "runtime": int(runtime) if pd.notna(runtime) else None,
            "vote_average": float(vote_average) if pd.notna(vote_average) else None,
        }
