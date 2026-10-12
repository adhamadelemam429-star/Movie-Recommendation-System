from app.data_loader import ensure_sample_data, load_movies_data, load_ratings_data


def test_ensure_sample_data_creates_data():
    movies, ratings = ensure_sample_data()

    assert not movies.empty
    assert not ratings.empty
    assert {"movieId", "title", "genres"}.issubset(movies.columns)
    assert {"userId", "movieId", "rating", "timestamp"}.issubset(ratings.columns)

    loaded_movies = load_movies_data()
    loaded_ratings = load_ratings_data()

    assert not loaded_movies.empty
    assert not loaded_ratings.empty


def test_sample_data_includes_poster_and_language_fields():
    movies, _ = ensure_sample_data()

    assert "poster_url" in movies.columns
    assert "language" in movies.columns
    assert len(movies) >= 20
    assert movies["poster_url"].notna().all()


def test_known_movies_use_tmdb_poster_artwork():
    movies, _ = ensure_sample_data()
    inception = movies.loc[movies["title"] == "Inception", "poster_url"].iloc[0]

    assert inception.startswith("https://media.themoviedb.org/t/p/")
