from app.content_based import build_content_model, get_similar_movies
from app.data_loader import ensure_sample_data


def test_content_based_recommendations_exclude_self():
    movies, ratings = ensure_sample_data()
    model = build_content_model(movies)
    result = get_similar_movies(1, movies, model=model, top_n=3)

    assert len(result) == 3
    assert 1 not in result["movieId"].tolist()
    assert {"movieId", "title", "score"}.issubset(result.columns)
