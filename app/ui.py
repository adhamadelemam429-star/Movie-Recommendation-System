from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.data_loader import TMDB_IMAGE_BASE, fetch_movie_lens_dataset
from app.evaluation import compute_detailed_metrics
from app.recommendation_service import RecommendationService


@st.cache_resource
def build_service(dataset_mode: str) -> RecommendationService:
    if dataset_mode == "sample":
        return RecommendationService()
    if dataset_mode in {"ml-latest-small", "ml-latest"}:
        movies, ratings = fetch_movie_lens_dataset(dataset_mode)
        return RecommendationService(movies, ratings)
    return RecommendationService()


def render_movie_cards(items: list[dict], empty_message: str = "No movies found.", key_prefix: str = "cards") -> None:
    if not items:
        st.info(empty_message)
        return

    cols = st.columns(min(5, len(items)))
    for col, item in zip(cols, items):
        with col:
            poster = item.get("poster_url") or "assets/posters/default-movie.svg"
            st.image(poster, width="stretch")
            st.markdown(f"**{item.get('title', 'Unknown')}**")
            st.caption(f"{item.get('year', 'N/A')} • {item.get('language', 'Unknown')}")
            genres = item.get("genres") or ""
            st.caption(genres)
            score = item.get("score")
            if score is not None:
                st.caption(f"Score: {float(score):.2f}")
            if st.button("View film", key=f"{key_prefix}_{item.get('movieId')}", width="stretch"):
                st.session_state.selected_movie_id = int(item["movieId"])
                st.rerun()


st.set_page_config(page_title="FRAME | Movie Discovery", page_icon="🎬", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#f4f1ec; --muted:#a7a29c; --accent:#e50914; --line:#33302e; --panel:#171615; }
.stApp { background: radial-gradient(ellipse at 78% 0%, #33211f 0%, #171514 35%, #11100f 76%); color:var(--ink); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:0; }
h1 { font-size:2.35rem!important; font-weight:800!important; }
.block-container { padding-top:2rem; max-width:1500px; }
[data-testid="stSidebar"] { background:#171615; border-right:1px solid var(--line); }
[data-testid="stMetric"] { background:#1d1b19; border:1px solid var(--line); padding:14px 18px; border-radius:6px; }
[data-testid="stMetricValue"] { color:#fff; }
[data-testid="stImage"] img { border-radius:5px; }
.stButton > button { border-radius:4px; border:1px solid #56504b; background:#211f1d; color:#f4f1ec; font-weight:600; }
.stButton > button:hover { border-color:#e50914; color:#fff; background:#30201f; }
[data-baseweb="tab-list"] { gap:18px; border-bottom:1px solid var(--line); }
[data-baseweb="tab"] { background:transparent; }
.frame-kicker { color:#e50914; text-transform:uppercase; font-size:12px; font-weight:700; letter-spacing:2px; }
.frame-hero { padding:24px 0 8px; }
.frame-rule { border-top:1px solid var(--line); margin:24px 0; }
@media (max-width:700px) { .block-container {padding:1rem;} h1 {font-size:1.8rem!important;} }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="frame-kicker">A better way to find your next film</div>', unsafe_allow_html=True)
st.title("FRAME")
st.caption("Curated discovery, built around what you love.")

with st.sidebar:
    st.header("Controls")
    dataset_mode = st.selectbox("Dataset source", ["sample", "ml-latest-small", "ml-latest"])
    algorithm = st.selectbox("Algorithm", ["hybrid", "content_based", "collaborative", "popularity"])
    top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    user_id = st.number_input("User ID", min_value=1, max_value=250, value=1, step=1)

service = build_service(dataset_mode)
all_movies = service.movies.copy()
available_movies = all_movies
if dataset_mode == "sample":
    has_real_poster = all_movies["poster_url"].fillna("").str.startswith(TMDB_IMAGE_BASE)
    available_movies = all_movies[has_real_poster].copy()
with st.sidebar:
    st.divider()
    st.subheader("Refine the catalog")
    foreign_only = st.toggle("International films only", value=False)
    language_options = ["Any language", *sorted(available_movies.get("language", pd.Series(dtype=str)).dropna().astype(str).unique())]
    country_options = ["Any country", *sorted(available_movies.get("country", pd.Series(dtype=str)).dropna().astype(str).unique())]
    genre_options = ["Any genre", *sorted({genre for value in available_movies["genres"].dropna() for genre in str(value).split("|")})]
    language_filter = st.selectbox("Language", language_options)
    country_filter = st.selectbox("Country", country_options)
    genre_filter = st.selectbox("Genre", genre_options)
    years = pd.to_numeric(available_movies.get("year", pd.Series(dtype=float)), errors="coerce").dropna()
    year_range = st.slider("Release years", int(years.min()), int(years.max()), (int(years.min()), int(years.max()))) if not years.empty else None

catalog = available_movies
if foreign_only and "is_foreign" in catalog:
    catalog = catalog[catalog["is_foreign"].fillna(False).astype(bool)]
if language_filter != "Any language":
    catalog = catalog[catalog["language"] == language_filter]
if country_filter != "Any country":
    catalog = catalog[catalog["country"] == country_filter]
if genre_filter != "Any genre":
    catalog = catalog[catalog["genres"].str.contains(genre_filter, case=False, na=False)]
if year_range and "year" in catalog:
    numeric_year = pd.to_numeric(catalog["year"], errors="coerce")
    catalog = catalog[numeric_year.between(*year_range)]

stats = service.ratings
movies_count = len(service.movies)
users_count = stats["userId"].nunique()
ratings_count = len(stats)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Movies", len(available_movies))
col2.metric("Users", users_count)
col3.metric("Ratings", ratings_count)
col4.metric("Dataset", dataset_mode)

search_query = st.text_input("Search the catalog", placeholder="Search by title...")
selected_movie_id = None
if search_query:
    results = [service.get_movie(int(row["movieId"])) for _, row in catalog[catalog["title"].str.contains(search_query, case=False, na=False)].head(10).iterrows()]
    if results:
        movie_options = {f"{item['movieId']} - {item['title']}": item["movieId"] for item in results}
        selected_label = st.selectbox("Matching movies", list(movie_options.keys()))
        selected_movie_id = movie_options[selected_label]
        st.session_state.selected_movie_id = selected_movie_id

international_picks = [
    service.get_movie(int(row["movieId"]))
    for _, row in catalog[catalog.get("is_foreign", pd.Series([False] * len(catalog))).fillna(False).astype(bool)]
    .sort_values("year", ascending=False, na_position="last")
    .head(8)
    .iterrows()
]

overview_tab, search_tab, evaluation_tab = st.tabs(["Discover", "For You", "Model Insights"])

with overview_tab:
    if international_picks:
        st.subheader("International favorites")
        render_movie_cards(international_picks, empty_message="No international titles match these filters.", key_prefix="international-picks")

    st.subheader("Popular right now")
    allowed_ids = set(catalog["movieId"].astype(int))
    popular = [item for item in service.get_popular_movies(top_n=max(top_n * 2, 10)) if item["movieId"] in allowed_ids][:top_n]
    render_movie_cards(popular, empty_message="No popular movies available.", key_prefix="popular")

    st.subheader("Browse the collection")
    catalog_items = [service.get_movie(int(row["movieId"])) for _, row in catalog.sort_values("title").head(20).iterrows()]
    render_movie_cards(catalog_items, empty_message="No films match the selected filters.", key_prefix="catalog")

with search_tab:
    if genre_filter != "Any genre":
        genre_results = service.recommend_by_genre([genre_filter], top_n=top_n)
        genre_results = [item for item in genre_results if item["movieId"] in set(catalog["movieId"].astype(int))]
        st.subheader("Genre-based suggestions")
        render_movie_cards(genre_results, empty_message="No movies match the selected genre.", key_prefix="genre")

    if selected_movie_id:
        st.subheader("Similar movies")
        similar = service.recommend_similar_movies(selected_movie_id, top_n=top_n)
        render_movie_cards(similar, empty_message="No similar movies were found for the selected movie.", key_prefix="search-similar")

    st.subheader("User recommendations")
    user_recs = service.recommend_for_user(int(user_id), top_n=top_n, method=algorithm)
    allowed_ids = set(catalog["movieId"].astype(int))
    user_recs = [item for item in user_recs if item["movieId"] in allowed_ids]
    render_movie_cards(user_recs, empty_message="No recommendations available for this user yet. Try a different user ID or use a popular fallback.", key_prefix="user-recs")

selected_movie_id = st.session_state.get("selected_movie_id", selected_movie_id)
if selected_movie_id:
    detail = service.get_movie(int(selected_movie_id))
    if detail:
        st.markdown('<div class="frame-rule"></div>', unsafe_allow_html=True)
        st.markdown('<div class="frame-kicker">Film details</div>', unsafe_allow_html=True)
        detail_poster, detail_info = st.columns([1, 3], gap="large")
        with detail_poster:
            st.image(detail.get("poster_url") or "assets/posters/default-movie.svg", width="stretch")
        with detail_info:
            st.header(detail["title"])
            st.markdown(f"*{detail.get('tagline') or 'Selected for your next movie night.'}*")
            rating = detail.get("vote_average")
            rating_text = f"★ {float(rating):.1f}/10" if pd.notna(rating) else "Rating unavailable"
            runtime = detail.get("runtime")
            runtime_text = f"{int(runtime)} min" if pd.notna(runtime) else "Runtime unavailable"
            st.caption(f"{detail.get('year', 'Year unknown')}  ·  {runtime_text}  ·  {rating_text}")
            st.caption(f"{detail.get('genres', 'Genre unavailable').replace('|', '  /  ')}  ·  {detail.get('language', 'Unknown')}  ·  {detail.get('country', 'Unknown')}")
            st.write(detail.get("overview") or "No synopsis is available for this title.")
            st.caption(service.explain_recommendation(int(selected_movie_id), method=algorithm).get("reason", ""))
            if st.button("Find similar films", key="detail_similar"):
                st.session_state.show_similar_to = int(selected_movie_id)
                st.rerun()
        if st.session_state.get("show_similar_to") == int(selected_movie_id):
            st.subheader("Because you chose this")
            render_movie_cards(service.recommend_similar_movies(int(selected_movie_id), top_n=top_n), key_prefix="detail-similar")

with evaluation_tab:
    evaluation_metrics = compute_detailed_metrics(service.movies, service.ratings, top_n=top_n)
    metric_df = pd.DataFrame(
        [
            {
                "Algorithm": name,
                "precision@5": values["precision@5"],
                "recall@5": values["recall@5"],
                "ndcg@5": values["ndcg@5"],
                "hit_rate@5": values["hit_rate@5"],
            }
            for name, values in evaluation_metrics.items()
        ]
    )
    st.dataframe(metric_df, width="stretch")
    st.bar_chart(metric_df.set_index("Algorithm"), width="stretch")

st.markdown("FRAME · Poster artwork via [TMDB](https://www.themoviedb.org/). This product uses TMDB data but is not endorsed or certified by TMDB.")
