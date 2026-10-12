from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_loader import ensure_sample_data, preprocess_data


if __name__ == "__main__":
    movies, ratings = ensure_sample_data()
    prepared = preprocess_data(movies, ratings)
    print(f"Prepared data saved successfully: {len(prepared[0])} movies, {len(prepared[1])} ratings.")
