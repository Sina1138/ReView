"""
Centralized configuration for ReView data pipeline.
"""

import re
from pathlib import Path


class Config:
    """Configuration for ReView data processing."""

    # Directories (BASE_DIR = project root, one level above pipeline/)
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "glimpse" / "data"
    PROCESSED_DIR = DATA_DIR / "processed"
    OUTPUT_DIR = BASE_DIR / "data"
    POLARITY_DIR = OUTPUT_DIR / "polarity_scored"
    TOPIC_DIR = OUTPUT_DIR / "topic_scored"

    # Output files
    LEGACY_PREPROCESSED = OUTPUT_DIR / "preprocessed_scored_reviews.csv"

    # Feature flags
    INCLUDE_REBUTTALS = True
    REBUTTAL_MIN_LENGTH = 10  # Minimum characters for valid rebuttal

    # OpenReview API
    OPENREVIEW_BASE_URL = 'https://api2.openreview.net'
    VENUE_TEMPLATE = 'ICLR.cc/{year}/Conference'

    # Model paths (HuggingFace)
    POLARITY_MODEL = "Sina1138/Scibert_polarity_Review"
    TOPIC_MODEL = "Sina1138/SciDeberta_Review"
    RSA_MODEL = "sshleifer/distilbart-cnn-12-3"  # For GLIMPSE

    @classmethod
    def find_available_years(cls, data_dir: Path = None) -> list:
        """Auto-detect available years from all_reviews_YYYY.csv files in data_dir."""
        if data_dir is None:
            data_dir = cls.OUTPUT_DIR
        years = []
        for f in data_dir.glob("all_reviews_*.csv"):
            match = re.search(r'all_reviews_(\d{4})\.csv', f.name)
            if match:
                years.append(int(match.group(1)))
        return sorted(years)

    @classmethod
    def get_preprocessed_path(cls, years: list = None) -> Path:
        """Get output CSV path based on the year range.

        If years is None, auto-detects from data directory.
        Produces filenames like preprocessed_scored_reviews_2020-2025.csv
        """
        if years is None:
            years = cls.find_available_years()
        if not years:
            # Fallback to a generic name if no years detected
            return cls.OUTPUT_DIR / "preprocessed_scored_reviews.csv"
        return cls.OUTPUT_DIR / f"preprocessed_scored_reviews_{min(years)}-{max(years)}.csv"

    @classmethod
    def find_preprocessed_csv(cls) -> Path:
        """Find the most recent preprocessed_scored_reviews_*.csv in the output dir.

        Useful for the interface to locate the dataset without knowing the year range.
        """
        candidates = sorted(cls.OUTPUT_DIR.glob("preprocessed_scored_reviews_*.csv"))
        if candidates:
            return candidates[-1]  # Last alphabetically = latest year range
        # Fallback to legacy
        if cls.LEGACY_PREPROCESSED.exists():
            return cls.LEGACY_PREPROCESSED
        return cls.OUTPUT_DIR / "preprocessed_scored_reviews.csv"
