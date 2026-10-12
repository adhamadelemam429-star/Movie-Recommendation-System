from app.collaborative_filtering import build_user_item_matrix, recommend_for_user
from app.data_loader import ensure_sample_data


def test_collaborative_filtering_recommendations_for_user():
    movies, ratings = ensure_sample_data()
    matrix = build_user_item_matrix(ratings)
    recommendations = recommend_for_user(1, ratings, movies, top_n=3)

    assert len(recommendations) == 3
    assert {"movieId", "title", "score"}.issubset(recommendations.columns)
    assert not recommendations.empty
    assert matrix is not None
