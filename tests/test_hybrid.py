from app.data_loader import ensure_sample_data
from app.hybrid_recommender import hybrid_recommend


def test_hybrid_recommendations():
    movies, ratings = ensure_sample_data()
    result = hybrid_recommend(1, movies, ratings, top_n=3)

    assert len(result) == 3
    assert {"movieId", "title", "score", "method"}.issubset(result.columns)
    assert not result.empty
