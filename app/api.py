from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from app.recommendation_service import RecommendationService

app = FastAPI(title="Movie Recommendation API")
service = RecommendationService()


class RecommendationRequest(BaseModel):
    user_id: Optional[int] = None
    movie_id: Optional[int] = None
    top_n: int = 5
    method: str = "hybrid"
    genres: List[str] = []


class RecommendationResponse(BaseModel):
    movieId: int
    title: str
    score: Optional[float] = None
    method: str
    reason: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "movie-recommendation-system"}


@app.get("/movies/search")
def search_movies(q: str = Query(..., min_length=1)) -> list[dict]:
    results = service.search_movies(q)
    return results


@app.get("/movies/{movie_id}/similar")
def similar_movies(movie_id: int, top_n: int = Query(5, ge=1, le=25)) -> list[dict]:
    results = service.recommend_similar_movies(movie_id, top_n=top_n)
    if not results:
        raise HTTPException(status_code=404, detail="Movie not found or no similar content available.")
    return results


@app.get("/recommendations/popular")
def popular(top_n: int = Query(5, ge=1, le=25)) -> list[dict]:
    return service.get_popular_movies(top_n=top_n)


@app.get("/recommendations/user/{user_id}")
def user_recommendations(user_id: int, top_n: int = Query(5, ge=1, le=25), method: str = "hybrid") -> list[dict]:
    results = service.recommend_for_user(user_id, top_n=top_n, method=method)
    if not results:
        fallback = service.get_popular_movies(top_n=top_n)
        if fallback:
            return fallback
        raise HTTPException(status_code=404, detail="No recommendations available for the requested user.")
    return results


@app.post("/recommendations/custom")
def custom_recommendation(request: RecommendationRequest) -> list[dict]:
    if request.user_id is not None:
        return service.recommend_for_user(request.user_id, top_n=request.top_n, method=request.method)
    if request.movie_id is not None:
        return service.recommend_similar_movies(request.movie_id, top_n=request.top_n)
    if request.genres:
        return service.recommend_by_genre(request.genres, top_n=request.top_n)
    raise HTTPException(status_code=400, detail="Provide user_id, movie_id, or genres.")
