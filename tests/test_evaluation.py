from app.data_loader import ensure_sample_data
from app.evaluation import compute_detailed_metrics, evaluate_recommenders


def test_evaluation_returns_metrics():
    movies, ratings = ensure_sample_data()
    results = evaluate_recommenders(movies, ratings)

    assert isinstance(results, dict)
    assert {"content_based", "collaborative", "hybrid", "popularity"}.issubset(results.keys())
    assert all(isinstance(value, (int, float)) for value in results.values())


def test_compute_detailed_metrics_includes_professional_metrics():
    movies, ratings = ensure_sample_data()
    results = compute_detailed_metrics(movies, ratings, top_n=5)

    assert {"content_based", "collaborative", "hybrid", "popularity"}.issubset(results.keys())
    for method_metrics in results.values():
        for key in ["precision@5", "recall@5", "ndcg@5", "hit_rate@5"]:
            assert key in method_metrics
            assert isinstance(method_metrics[key], float)
