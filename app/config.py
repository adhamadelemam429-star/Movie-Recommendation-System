from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

load_dotenv(BASE_DIR / ".env")

APP_NAME = "Movie Recommendation System"
DEFAULT_TOP_N = 5
MAX_TOP_N = 25

DATA_SOURCE_MODE = os.getenv("DATA_SOURCE_MODE", "sample")
VALIDATION_RATING_MIN = 0.5
VALIDATION_RATING_MAX = 5.0
