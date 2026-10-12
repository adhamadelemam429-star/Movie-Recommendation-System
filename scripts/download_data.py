from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_loader import ensure_sample_data, fetch_movie_lens_small


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official MovieLens data or create a local sample fallback.")
    parser.add_argument("--force-sample", action="store_true")
    args = parser.parse_args()

    if args.force_sample:
        movies, ratings = ensure_sample_data()
        print(f"Sample data prepared: {len(movies)} movies and {len(ratings)} ratings.")
        return

    try:
        movies, ratings = fetch_movie_lens_small()
        print(f"Downloaded dataset: {len(movies)} movies and {len(ratings)} ratings.")
    except Exception as exc:  # pragma: no cover
        print(f"Falling back to synthetic sample data because the official dataset could not be downloaded: {exc}")
        movies, ratings = ensure_sample_data()
        print(f"Sample data prepared: {len(movies)} movies and {len(ratings)} ratings.")


if __name__ == "__main__":
    main()
