"""
Centralized configuration for ReView data pipeline.
"""

from pathlib import Path


class Config:
    """Configuration for ReView data processing."""

    # Directories
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "glimpse" / "data"
    PROCESSED_DIR = DATA_DIR / "processed"
    OUTPUT_DIR = BASE_DIR / "data"
    POLARITY_DIR = OUTPUT_DIR / "polarity_scored"
    TOPIC_DIR = OUTPUT_DIR / "topic_scored"

    # Year ranges - LEGACY DATA (reference only, for documentation)
    LEGACY_START_YEAR = 2017
    LEGACY_END_YEAR = 2021

    # Year ranges - NEW DATA (reference only, actual years are auto-detected)
    # The system can work with any year - these are just for documentation
    NEW_DATA_START_YEAR = 2020
    NEW_DATA_END_YEAR = 2025

    # Output files
    LEGACY_PREPROCESSED = OUTPUT_DIR / "preprocessed_scored_reviews.csv"
    NEW_PREPROCESSED = OUTPUT_DIR / "preprocessed_scored_reviews_2020-2025.csv"

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
    def get_legacy_years(cls):
        """Get legacy year range (2017-2021)."""
        return range(cls.LEGACY_START_YEAR, cls.LEGACY_END_YEAR + 1)

    @classmethod
    def get_new_years(cls):
        """Get new data year range (2022-2025)."""
        return range(cls.NEW_DATA_START_YEAR, cls.NEW_DATA_END_YEAR + 1)

    @classmethod
    def get_all_years(cls):
        """Get complete year range (2017-2025)."""
        return range(cls.LEGACY_START_YEAR, cls.NEW_DATA_END_YEAR + 1)
