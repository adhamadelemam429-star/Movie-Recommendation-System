from app.data_loader import ensure_sample_data
from app.preprocessing import preprocess_movies_and_ratings, validate_movie_rating_integrity


def test_preprocessing_and_integrity():
    movies, ratings = ensure_sample_data()
    report = validate_movie_rating_integrity(movies, ratings)

    assert report["movie_ids_valid"] is True
    assert report["rating_range_valid"] is True
    assert report["missing_values"] >= 0

    cleaned = preprocess_movies_and_ratings(movies, ratings)
    assert {"movies", "ratings"}.issubset(cleaned.keys())
    assert not cleaned["movies"].empty
    assert not cleaned["ratings"].empty
