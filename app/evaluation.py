from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from app.content_based import get_similar_movies
from app.collaborative_filtering import recommend_for_user
from app.hybrid_recommender import hybrid_recommend


def _create_holdout_split(ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_rows = []
    test_rows = []

    for user_id, df in ratings.groupby("userId", sort=True):
        if len(df) < 2:
            train_rows.append(df)
            continue
        ordered = df.sort_values("timestamp").copy()
        test = ordered.iloc[[-1]]
        train = ordered.iloc[:-1]
        test_rows.append(test)
        train_rows.append(train)

    return pd.concat(train_rows, ignore_index=True), pd.concat(test_rows, ignore_index=True)


def _predict_for_method(method: str, user_id: int, movies: pd.DataFrame, ratings: pd.DataFrame, top_n: int = 5) -> List[int]:
    if method == "content_based":
        watched = ratings[ratings["userId"] == user_id]["movieId"].astype(int).tolist()
        if not watched:
            return []
        top_movie = max(watched, key=lambda movie_id: ratings[(ratings["userId"] == user_id) & (ratings["movieId"] == movie_id)]["rating"].sum())
        similar = get_similar_movies(int(top_movie), movies, top_n=top_n)
        return similar["movieId"].astype(int).tolist()

    if method == "collaborative":
        recommendations = recommend_for_user(user_id, ratings, movies, top_n=top_n)
        return recommendations["movieId"].astype(int).tolist()

    if method == "hybrid":
        recommendations = hybrid_recommend(user_id, movies, ratings, top_n=top_n)
        return recommendations["movieId"].astype(int).tolist()

    if method == "popularity":
        popularity = (
            ratings.groupby("movieId")
            .agg(avg_rating=("rating", "mean"), count=("rating", "size"))
            .sort_values(["count", "avg_rating"], ascending=[False, False])
            .reset_index()
        )
        return popularity["movieId"].astype(int).head(top_n).tolist()

    raise ValueError(f"Unsupported method: {method}")


def _precision_recall_ndcg(actual_ids: Iterable[int], recommended_ids: Iterable[int], top_n: int) -> Tuple[float, float, float]:
    actual = set(int(item) for item in actual_ids)
    recommended = list(int(item) for item in recommended_ids[:top_n])
    if not recommended:
        return 0.0, 0.0, 0.0

    hits = set(actual).intersection(recommended)
    precision = len(hits) / len(recommended)
    recall = len(hits) / len(actual) if actual else 0.0

    dcg = 0.0
    for index, movie_id in enumerate(recommended, start=1):
        if movie_id in actual:
            dcg += 1.0 / math.log2(index + 1)

    ideal = min(len(actual), top_n)
    idcg = 0.0
    for index in range(1, ideal + 1):
        idcg += 1.0 / math.log2(index + 1)

    ndcg = dcg / idcg if idcg > 0 else 0.0
    return precision, recall, ndcg


def compute_detailed_metrics(movies: pd.DataFrame, ratings: pd.DataFrame, top_n: int = 5) -> Dict[str, Dict[str, float]]:
    """Return professional ranking metrics for each recommender at a given cutoff."""
    train, test = _create_holdout_split(ratings)
    metrics: Dict[str, Dict[str, float]] = {}
    methods = ["content_based", "collaborative", "hybrid", "popularity"]

    for method in methods:
        precision_total = 0.0
        recall_total = 0.0
        ndcg_total = 0.0
        hit_total = 0.0
        user_count = 0

        for user_id, group in test.groupby("userId"):
            actual = set(int(item) for item in group["movieId"].astype(int).tolist())
            if not actual:
                continue

            recommendations = _predict_for_method(method, int(user_id), movies, train, top_n=top_n)
            if not recommendations:
                continue

            precision, recall, ndcg = _precision_recall_ndcg(actual, recommendations, top_n)
            precision_total += precision
            recall_total += recall
            ndcg_total += ndcg
            hit_total += 1.0 if actual.intersection(int(item) for item in recommendations[:top_n]) else 0.0
            user_count += 1

        if user_count == 0:
            metrics[method] = {"precision@5": 0.0, "recall@5": 0.0, "ndcg@5": 0.0, "hit_rate@5": 0.0}
            continue

        metrics[method] = {
            "precision@5": round(precision_total / user_count, 4),
            "recall@5": round(recall_total / user_count, 4),
            "ndcg@5": round(ndcg_total / user_count, 4),
            "hit_rate@5": round(hit_total / user_count, 4),
        }

    return metrics


def evaluate_recommenders(movies: pd.DataFrame, ratings: pd.DataFrame) -> Dict[str, float]:
    """Return the primary summary metric used by the project while keeping compatibility."""
    detailed = compute_detailed_metrics(movies, ratings, top_n=5)
    return {method: round(values["ndcg@5"], 4) for method, values in detailed.items()}


def export_evaluation_report(movies: pd.DataFrame, ratings: pd.DataFrame, output_path: str = "reports/evaluation_report.csv") -> pd.DataFrame:
    """Save the evaluation metrics to a CSV file for reporting."""
    detailed = compute_detailed_metrics(movies, ratings, top_n=5)
    summary_rows = []
    for algorithm, metrics in detailed.items():
        summary_rows.append(
            {
                "algorithm": algorithm,
                "score": round(metrics["ndcg@5"], 4),
                "precision@5": metrics["precision@5"],
                "recall@5": metrics["recall@5"],
                "ndcg@5": metrics["ndcg@5"],
                "hit_rate@5": metrics["hit_rate@5"],
            }
        )

    output_path = str(output_path)
    path = Path(output_path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_path, index=False)
    return summary
