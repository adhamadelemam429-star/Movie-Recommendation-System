from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_loader import ensure_sample_data
from app.evaluation import export_evaluation_report


if __name__ == "__main__":
    movies, ratings = ensure_sample_data()
    summary = export_evaluation_report(movies, ratings, output_path="reports/evaluation_report.csv")
    print(summary.to_string(index=False))
