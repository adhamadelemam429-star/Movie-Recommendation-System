from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_loader import ensure_sample_data
from app.recommendation_service import RecommendationService


if __name__ == "__main__":
    movies, ratings = ensure_sample_data()
    service = RecommendationService(movies, ratings)
    metadata = {
        "movies": len(movies),
        "users": len(ratings["userId"].unique()),
        "ratings": len(ratings),
        "sample_mode": True,
        "algorithms": ["content_based", "collaborative", "hybrid", "popularity"],
    }
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    (models_dir / "service_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
