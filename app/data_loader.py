from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
POSTERS_DIR = BASE_DIR / "assets" / "posters"
TMDB_POSTERS = {
    "inception": "xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg",
    "the dark knight": "qJ2tW6WMUDux911r6m7haRef0WH.jpg",
    "interstellar": "yQvGrMoipbRoddT0ZR8tPoR7NfX.jpg",
    "the matrix": "dXNAPwY7VrqMAo51EKhhCJfaGb5.jpg",
    "pulp fiction": "vQWk5YBFWF4bZaofAbv0tShwBvQ.jpg",
    "the lord of the rings": "6oom5QYQ2yQTMJIbnvbkBL9cHo6.jpg",
    "toy story": "uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
    "finding nemo": "eHuGQ10FUzK1mdOY69wF5pGgEf5.jpg",
    "the godfather": "3bhkrj58Vtu7enYsRolD1fZdja1.jpg",
    "avatar": "gKY6q7SjCkAU6FqvqWybDYgUKIF.jpg",
    "forrest gump": "saHP97rTPS5eLmrLQEcANmKrsFl.jpg",
    "shrek": "iB64vpL3dIObOtMZgX3RqdVdQDc.jpg",
    "parasite": "7IiTTgloJzvGI1TAYymCfbfl3vT.jpg",
    "spirited away": "39wmItIWsg5sZMyRUHLkWBcuVCM.jpg",
    "amélie": "nSxDa3M9aMvGVLoItzWTepQ5h5d.jpg",
    "the lives of others": "cVUDMnskSc01rdbyH0tLATTJUdP.jpg",
    "the intouchables": "1QU7HKgsQbGpzsJbJK4pAVQV9F5.jpg",
    "your name": "vfJFJPepRKapMd5G2ro7klIRysq.jpg",
    "roma": "dtIIyQyALk57ko5bjac7hi01YQ.jpg",
    "the boy and the heron": "f4oZTcfGrVTXKTWg157AwikXqmP.jpg",
}
TMDB_IMAGE_BASE = "https://media.themoviedb.org/t/p/w300_and_h450_face/"


def _ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(value: str) -> str:
    cleaned = ''.join(ch.lower() if ch.isalnum() else '-' for ch in value)
    while '--' in cleaned:
        cleaned = cleaned.replace('--', '-')
    return cleaned.strip('-') or 'movie'


def _normalized_title(value: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", value).strip().casefold()


def _ensure_poster_assets(movies: pd.DataFrame) -> pd.DataFrame:
    POSTERS_DIR.mkdir(parents=True, exist_ok=True)
    for row_index, row in movies.iterrows():
        title = str(row.get("title", "Movie"))
        movie_id = int(row["movieId"])
        poster_path = TMDB_POSTERS.get(_normalized_title(title))
        if poster_path:
            movies.at[row_index, "poster_url"] = f"{TMDB_IMAGE_BASE}{poster_path}"
            continue

        path = POSTERS_DIR / f"{movie_id}-{_slugify(title)}.svg"
        if path.exists():
            continue

        palette = [
            ("#0f172a", "#1d4ed8"),
            ("#1f2937", "#7c3aed"),
            ("#111827", "#f59e0b"),
            ("#0f766e", "#14b8a6"),
            ("#7c2d12", "#f97316"),
            ("#312e81", "#a78bfa"),
            ("#4c1d95", "#c084fc"),
            ("#1e293b", "#38bdf8"),
        ]
        start, end = palette[(movie_id - 1) % len(palette)]
        safe_title = title.replace("&", "and").replace("<", "").replace(">", "")
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="400" height="600" viewBox="0 0 400 600">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{start}"/>
      <stop offset="100%" stop-color="{end}"/>
    </linearGradient>
  </defs>
  <rect width="400" height="600" fill="url(#grad)"/>
  <rect x="18" y="18" width="364" height="564" rx="18" fill="rgba(15,23,42,0.18)" stroke="rgba(255,255,255,0.22)"/>
  <text x="200" y="270" font-size="32" font-weight="700" text-anchor="middle" fill="white" font-family="Arial, sans-serif">{safe_title}</text>
  <text x="200" y="330" font-size="16" text-anchor="middle" fill="rgba(255,255,255,0.78)" font-family="Arial, sans-serif">Movie Recommendation</text>
</svg>'''
        path.write_text(svg, encoding="utf-8")

    for row_index, row in movies.iterrows():
        movie_id = int(row["movieId"])
        title = str(row.get("title", "Movie"))
        poster_path = TMDB_POSTERS.get(_normalized_title(title))
        movies.at[row_index, "poster_url"] = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else f"assets/posters/{movie_id}-{_slugify(title)}.svg"

    return movies


def _sample_movies() -> pd.DataFrame:
    movies = [
        {"movieId": 1, "title": "Inception", "genres": "Action|Sci-Fi", "year": 2010, "language": "English", "country": "USA", "is_foreign": False},
        {"movieId": 2, "title": "The Dark Knight", "genres": "Action|Crime|Drama", "year": 2008, "language": "English", "country": "USA", "is_foreign": False},
        {"movieId": 3, "title": "Interstellar", "genres": "Adventure|Sci-Fi|Drama", "year": 2014, "language": "English", "country": "USA", "is_foreign": False},
        {"movieId": 4, "title": "The Matrix", "genres": "Action|Sci-Fi", "year": 1999, "language": "English", "country": "USA", "is_foreign": False},
        {"movieId": 5, "title": "Pulp Fiction", "genres": "Crime|Drama", "year": 1994, "language": "English", "country": "USA", "is_foreign": False},
        {"movieId": 6, "title": "The Lord of the Rings", "genres": "Adventure|Fantasy", "year": 2001, "language": "English", "country": "New Zealand", "is_foreign": False},
        {"movieId": 7, "title": "Toy Story", "genres": "Animation|Children|Comedy", "year": 1995, "language": "English", "country": "USA", "is_foreign": False},
        {"movieId": 8, "title": "Finding Nemo", "genres": "Animation|Adventure|Children", "year": 2003, "language": "English", "country": "USA", "is_foreign": False},
        {"movieId": 9, "title": "The Godfather", "genres": "Crime|Drama", "year": 1972, "language": "English", "country": "USA", "is_foreign": False},
        {"movieId": 10, "title": "Avatar", "genres": "Action|Adventure|Sci-Fi", "year": 2009, "language": "English", "country": "USA", "is_foreign": False},
        {"movieId": 11, "title": "Forrest Gump", "genres": "Comedy|Drama|Romance", "year": 1994, "language": "English", "country": "USA", "is_foreign": False},
        {"movieId": 12, "title": "Shrek", "genres": "Animation|Comedy|Fantasy", "year": 2001, "language": "English", "country": "USA", "is_foreign": False},
        {"movieId": 13, "title": "Parasite", "genres": "Drama|Thriller", "year": 2019, "language": "Korean", "country": "South Korea", "is_foreign": True},
        {"movieId": 14, "title": "Spirited Away", "genres": "Animation|Fantasy", "year": 2001, "language": "Japanese", "country": "Japan", "is_foreign": True},
        {"movieId": 15, "title": "Amélie", "genres": "Comedy|Romance", "year": 2001, "language": "French", "country": "France", "is_foreign": True},
        {"movieId": 16, "title": "The Lives of Others", "genres": "Drama|History", "year": 2006, "language": "German", "country": "Germany", "is_foreign": True},
        {"movieId": 17, "title": "The Intouchables", "genres": "Comedy|Drama", "year": 2011, "language": "French", "country": "France", "is_foreign": True},
        {"movieId": 18, "title": "Your Name", "genres": "Animation|Drama|Romance", "year": 2016, "language": "Japanese", "country": "Japan", "is_foreign": True},
        {"movieId": 19, "title": "Roma", "genres": "Drama", "year": 2018, "language": "Spanish", "country": "Mexico", "is_foreign": True},
        {"movieId": 20, "title": "The Silent Sea", "genres": "Drama|Sci-Fi|Thriller", "year": 2021, "language": "Korean", "country": "South Korea", "is_foreign": True},
        {"movieId": 21, "title": "A Quiet Place: Day One", "genres": "Drama|Horror|Sci-Fi", "year": 2024, "language": "English", "country": "USA", "is_foreign": False},
        {"movieId": 22, "title": "The Book of Clarence", "genres": "Comedy|Drama", "year": 2024, "language": "English", "country": "USA", "is_foreign": False},
        {"movieId": 23, "title": "La Dolce Villa", "genres": "Comedy|Romance", "year": 2025, "language": "Italian", "country": "Italy", "is_foreign": True},
        {"movieId": 24, "title": "The Boy and the Heron", "genres": "Animation|Adventure|Drama", "year": 2023, "language": "Japanese", "country": "Japan", "is_foreign": True},
        {"movieId": 25, "title": "A Glass of Water", "genres": "Drama|Romance", "year": 2025, "language": "Arabic", "country": "Egypt", "is_foreign": True},
        {"movieId": 26, "title": "The Last Supper", "genres": "Drama|Mystery", "year": 2025, "language": "Spanish", "country": "Spain", "is_foreign": True},
        {"movieId": 27, "title": "Snow and the Bear", "genres": "Drama|Adventure", "year": 2025, "language": "French", "country": "France", "is_foreign": True},
        {"movieId": 28, "title": "Summer Palace 2025", "genres": "Drama|Romance", "year": 2025, "language": "Chinese", "country": "China", "is_foreign": True},
        {"movieId": 29, "title": "After the Rain", "genres": "Drama|Romance", "year": 2025, "language": "Korean", "country": "South Korea", "is_foreign": True},
        {"movieId": 30, "title": "Metro 2035", "genres": "Sci-Fi|Thriller", "year": 2025, "language": "German", "country": "Germany", "is_foreign": True},
    ]
    movies_df = pd.DataFrame(movies)
    return _ensure_poster_assets(movies_df)


def _sample_ratings() -> pd.DataFrame:
    records = [
        (1, 1, 5.0, 1717189200), (1, 2, 4.0, 1717189201), (1, 3, 5.0, 1717189202),
        (1, 10, 4.5, 1717189203), (1, 6, 3.5, 1717189204),
        (2, 1, 4.5, 1717189205), (2, 4, 5.0, 1717189206), (2, 3, 4.0, 1717189207),
        (2, 10, 5.0, 1717189208), (2, 12, 3.0, 1717189209),
        (3, 2, 5.0, 1717189210), (3, 5, 4.0, 1717189211), (3, 9, 4.5, 1717189212),
        (3, 11, 3.5, 1717189213), (3, 12, 4.0, 1717189214),
        (4, 7, 5.0, 1717189215), (4, 8, 4.5, 1717189216), (4, 12, 4.0, 1717189217),
        (4, 1, 2.0, 1717189218), (4, 6, 2.5, 1717189219),
        (5, 7, 3.5, 1717189220), (5, 8, 5.0, 1717189221), (5, 12, 4.5, 1717189222),
        (5, 11, 4.0, 1717189223), (5, 10, 2.0, 1717189224),
        (6, 3, 5.0, 1717189225), (6, 4, 4.5, 1717189226), (6, 10, 4.0, 1717189227),
        (6, 1, 4.0, 1717189228), (6, 2, 3.5, 1717189229),
    ]
    return pd.DataFrame(records, columns=["userId", "movieId", "rating", "timestamp"]) 


def generate_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_directories()
    movies_df = _sample_movies()
    ratings_df = _sample_ratings()

    movies_df.to_csv(RAW_DIR / "movies.csv", index=False)
    ratings_df.to_csv(RAW_DIR / "ratings.csv", index=False)
    movies_df.to_csv(SAMPLE_DIR / "movies.csv", index=False)
    ratings_df.to_csv(SAMPLE_DIR / "ratings.csv", index=False)
    return movies_df, ratings_df


def ensure_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    movie_path = RAW_DIR / "movies.csv"
    ratings_path = RAW_DIR / "ratings.csv"

    if not movie_path.exists() or not ratings_path.exists():
        return generate_sample_data()

    movies_df = pd.read_csv(movie_path)
    ratings_df = pd.read_csv(ratings_path)

    required_movie_columns = {"movieId", "title", "genres", "year", "language", "country", "is_foreign", "poster_url"}
    required_rating_columns = {"userId", "movieId", "rating", "timestamp"}

    if not required_movie_columns.issubset(movies_df.columns) or not required_rating_columns.issubset(ratings_df.columns):
        return generate_sample_data()

    if len(movies_df) < 20:
        return generate_sample_data()

    movies_df = _ensure_poster_assets(movies_df)
    movies_df.to_csv(movie_path, index=False)
    movies_df.to_csv(SAMPLE_DIR / "movies.csv", index=False)
    return movies_df, ratings_df


def load_movies_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        ensure_sample_data()
        path = RAW_DIR / "movies.csv"
    path = Path(path)
    if not path.exists():
        return ensure_sample_data()[0]
    return pd.read_csv(path)


def load_ratings_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        ensure_sample_data()
        path = RAW_DIR / "ratings.csv"
    path = Path(path)
    if not path.exists():
        return ensure_sample_data()[1]
    return pd.read_csv(path)


def save_processed_data(movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
    _ensure_directories()
    movies.to_csv(PROCESSED_DIR / "movies_processed.csv", index=False)
    ratings.to_csv(PROCESSED_DIR / "ratings_processed.csv", index=False)


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    movies_path = PROCESSED_DIR / "movies_processed.csv"
    ratings_path = PROCESSED_DIR / "ratings_processed.csv"
    if not movies_path.exists() or not ratings_path.exists():
        movies, ratings = ensure_sample_data()
        return preprocess_data(movies, ratings)
    return pd.read_csv(movies_path), pd.read_csv(ratings_path)


def preprocess_data(movies: pd.DataFrame, ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from app.preprocessing import preprocess_movies_and_ratings

    processed = preprocess_movies_and_ratings(movies, ratings)
    save_processed_data(processed["movies"], processed["ratings"])
    return processed["movies"], processed["ratings"]


def fetch_movie_lens_dataset(dataset_name: str = "ml-latest-small", output_dir: str | Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download a GroupLens MovieLens dataset when available.

    The function supports both the lightweight small dataset and the larger latest dataset,
    while falling back to the bundled synthetic sample data when download is unavailable.
    """

    dataset_name = dataset_name.strip().lower()
    supported = {"ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip", "ml-latest": "https://files.grouplens.org/datasets/movielens/ml-latest.zip"}
    if dataset_name not in supported:
        raise ValueError(f"Unsupported MovieLens dataset: {dataset_name}. Supported values: {sorted(supported)}")

    output_dir = Path(output_dir) if output_dir else RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = output_dir / dataset_name
    movies_path = extracted_dir / "movies.csv"
    ratings_path = extracted_dir / "ratings.csv"
    if movies_path.exists() and ratings_path.exists():
        return pd.read_csv(movies_path), pd.read_csv(ratings_path)

    try:
        import io
        import zipfile

        import requests

        response = requests.get(supported[dataset_name], timeout=30)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zipped:
            zipped.extractall(output_dir)

        if movies_path.exists() and ratings_path.exists():
            return pd.read_csv(movies_path), pd.read_csv(ratings_path)
    except Exception:
        pass

    return generate_sample_data()


def fetch_movie_lens_small(output_dir: str | Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible wrapper for the small MovieLens dataset."""
    return fetch_movie_lens_dataset("ml-latest-small", output_dir=output_dir)

