#!/usr/bin/env python3
"""
Standalone preprocessing script for ICLR data with rebuttal support.
Keeps glimpse-ui independent from the glimpse repository.
"""

import pandas as pd
import os
from pathlib import Path
from config import NEW_DATA_START_YEAR, NEW_DATA_END_YEAR, BASE_DIR


def preprocess_reviews_with_rebuttals(year: int,
                                       input_dir: Path = None,
                                       output_dir: Path = None):
    """
    Preprocess raw review data for a given year, including rebuttals.

    Args:
        year: Year to process
        input_dir: Directory containing raw all_reviews_{year}.csv files
        output_dir: Directory to write processed files
    """
    if input_dir is None:
        input_dir = BASE_DIR / "data"
    if output_dir is None:
        output_dir = BASE_DIR / "data" / "processed"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = input_dir / f"all_reviews_{year}.csv"
    output_file = output_dir / f"all_reviews_{year}.csv"

    if not input_file.exists():
        print(f"⚠️  Skipping {year}: {input_file} not found")
        return False

    print(f"Processing {year}...")
    dataset = pd.read_csv(input_file)

    # Check if rebuttal column exists
    if 'rebuttal' in dataset.columns:
        sub_dataset = dataset[['id', 'review', 'metareview', 'rebuttal']]
        sub_dataset.rename(columns={
            "review": "text",
            "metareview": "gold",
            "rebuttal": "rebuttal"
        }, inplace=True)
        print(f"  ✓ Found {len(dataset)} reviews with rebuttals")
    else:
        # Fallback for data without rebuttals (legacy compatibility)
        sub_dataset = dataset[['id', 'review', 'metareview']]
        sub_dataset.rename(columns={
            "review": "text",
            "metareview": "gold"
        }, inplace=True)
        sub_dataset['rebuttal'] = ''
        print(f"  ✓ Found {len(dataset)} reviews (no rebuttals)")

    sub_dataset.to_csv(output_file, index=False)
    print(f"  → Saved to {output_file}")
    return True


def main():
    """Preprocess all years in the configured range."""
    print(f"\n{'='*60}")
    print(f"Preprocessing ICLR data ({NEW_DATA_START_YEAR}-{NEW_DATA_END_YEAR})")
    print(f"{'='*60}\n")

    processed_count = 0
    for year in range(NEW_DATA_START_YEAR, NEW_DATA_END_YEAR + 1):
        if preprocess_reviews_with_rebuttals(year):
            processed_count += 1

    print(f"\n{'='*60}")
    print(f"✓ Preprocessing complete: {processed_count} years processed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
