#!/usr/bin/env python3
"""
Unified scoring pipeline - End-to-end data pipeline for ICLR review analysis.
Runs all steps including preprocessing, text cleaning, and all scoring (GLIMPSE, polarity, topic).
Builds final integrated dataset. Automatically skips existing results unless --force is used.

Pipeline steps:
    [0] Preprocessing & text cleaning (removes formatting artifacts)
    [1] GLIMPSE scoring (consensuality/agreement)
    [2] Polarity scoring (sentiment)
    [3] Topic scoring (aspects)
    [4] Final integration with all scores merged

Usage:
    python run_scoring.py --year 2020              # Full pipeline for single year
    python run_scoring.py                          # Auto-detect all available years
    python run_scoring.py --force                  # Reprocess everything
    python run_scoring.py --skip-preprocess        # Skip preprocessing (if already done)
    python run_scoring.py --skip-glimpse           # Skip GLIMPSE, just polarity/topic
    python run_scoring.py --year 2020 --limit 10   # Test with 10 reviews
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from dependencies.scoring_utils import find_available_years

# Import scoring functions
from preprocess_data import preprocess_reviews_with_rebuttals
from run_glimpse_scoring import run_glimpse_pipeline
from run_polarity_scoring import score_reviews_polarity
from run_topic_scoring import score_reviews_topic
from scored_reviews_builder import build_dataset


def run_full_pipeline(
    year: int,
    model_variant_polarity: str = "scibert",
    model_variant_topic: str = "scibert",
    device: str = "cuda",
    skip_if_exists: bool = True,
    skip_glimpse: bool = False,
    skip_preprocess: bool = False,
    limit: int = None,
) -> bool:
    """
    Run complete scoring pipeline for a single year.

    Args:
        year: Year to process
        model_variant_polarity: Polarity model ("scibert", "deberta", "scideberta")
        model_variant_topic: Topic model ("scibert", "deberta", "scideberta")
        device: Device for computation ("cuda" or "cpu")
        skip_if_exists: Skip if results already exist
        skip_glimpse: Skip GLIMPSE scoring step
        skip_preprocess: Skip preprocessing step (if already done)
        limit: Limit to first N reviews (None = process all)

    Returns:
        True if successful, False if failed
    """

    limit_str = f" (limit: {limit})" if limit else ""
    print(f"\n{'#'*60}")
    print(f"# Full Scoring Pipeline: {year}{limit_str}")
    print(f"{'#'*60}")

    try:
        # Step 0: Preprocessing (with text cleaning)
        if not skip_preprocess:
            print(f"\n[0/5] Preprocessing & Text Cleaning...")
            preprocess_reviews_with_rebuttals(year)
        else:
            print(f"\n[0/5] Skipping preprocessing (--skip-preprocess)")

        # Step 1: GLIMPSE Scoring
        if not skip_glimpse:
            print(f"\n[1/5] GLIMPSE Scoring...")
            run_glimpse_pipeline(
                year,
                model_name="facebook/bart-large-cnn",
                device=device,
                skip_if_exists=skip_if_exists,
            )
        else:
            print(f"\n[1/5] Skipping GLIMPSE scoring (--skip-glimpse)")

        # Step 2: Polarity Scoring
        print(f"\n[2/5] Polarity Scoring ({model_variant_polarity})...")
        score_reviews_polarity(
            year,
            model_variant=model_variant_polarity,
            device=device,
            skip_if_exists=skip_if_exists,
            limit=limit,
        )

        # Step 3: Topic Scoring
        print(f"\n[3/5] Topic Scoring ({model_variant_topic})...")
        score_reviews_topic(
            year,
            model_variant=model_variant_topic,
            device=device,
            skip_if_exists=skip_if_exists,
            limit=limit,
        )

        # Step 4: Build Final Dataset (always rebuild to ensure latest data)
        print(f"\n[4/5] Building Final Integrated Dataset...")
        build_dataset()

        print(f"\n[5/5] ✓ Pipeline Complete")
        
        print(f"\n{'='*60}")
        print(f"✓ Pipeline complete for {year}")
        print(f"{'='*60}")
        
        return True
    
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Pipeline failed for {year}: {e}")
        print(f"{'='*60}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Unified scoring pipeline - End-to-end processing for all review data"
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Single year to process (if not specified, auto-detects all available years)",
    )
    parser.add_argument(
        "--model-polarity",
        type=str,
        default="scibert",
        choices=["scibert", "deberta", "scideberta"],
        help="Model variant for polarity scoring (default: scibert)",
    )
    parser.add_argument(
        "--model-topic",
        type=str,
        default="scibert",
        choices=["scibert", "deberta", "scideberta"],
        help="Model variant for topic scoring (default: scibert)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for computation (default: cuda)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if results exist",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing (assume data already preprocessed)",
    )
    parser.add_argument(
        "--skip-glimpse",
        action="store_true",
        help="Skip GLIMPSE scoring (assume results already exist)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N reviews (None = process all)",
    )
    
    args = parser.parse_args()
    
    # Determine years to process
    if args.year:
        years = [args.year]
    else:
        # Auto-detect from raw data if preprocessing will be run
        if not args.skip_preprocess:
            from preprocess_data import find_available_years as find_raw_years
            raw_dir = Config.BASE_DIR / "data"
            years = find_raw_years(raw_dir)
            if not years:
                print("⚠️  No raw data found in data/")
                print("   Run fetch_iclr_data.py first")
                return
        else:
            # Auto-detect from preprocessed data if skipping preprocessing
            processed_dir = Config.BASE_DIR / "data" / "processed"
            years = find_available_years(processed_dir)
            if not years:
                print("⚠️  No preprocessed data found in data/processed/")
                print("   Run preprocess_data.py first or remove --skip-preprocess")
                return
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Unified Scoring Pipeline (with Preprocessing)")
    print(f"{'='*60}")
    print(f"Years: {years}")
    print(f"Include preprocessing: {not args.skip_preprocess}")
    print(f"Include GLIMPSE: {not args.skip_glimpse}")
    print(f"Polarity model: {args.model_polarity}")
    print(f"Topic model: {args.model_topic}")
    print(f"Device: {args.device}")
    print(f"Skip if exists: {not args.force}")
    if args.limit:
        print(f"Limit: {args.limit} reviews per year")
    print(f"{'='*60}")
    
    # Process each year
    success_count = 0
    failed_years = []
    
    for year in years:
        success = run_full_pipeline(
            year,
            model_variant_polarity=args.model_polarity,
            model_variant_topic=args.model_topic,
            device=args.device,
            skip_if_exists=not args.force,
            skip_preprocess=args.skip_preprocess,
            skip_glimpse=args.skip_glimpse,
            limit=args.limit,
        )
        
        if success:
            success_count += 1
        else:
            failed_years.append(year)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Pipeline Summary")
    print(f"{'='*60}")
    print(f"✓ Successful: {success_count}/{len(years)} years")
    if failed_years:
        print(f"✗ Failed: {failed_years}")
    output_path = Config.find_preprocessed_csv()
    print(f"\n📊 Final dataset: {output_path.relative_to(Config.BASE_DIR)}")
    print(f"   Ready for interface: python interface/Demo.py")
    print(f"{'='*60}\n")
    
    # Exit with error if any failed
    if failed_years:
        sys.exit(1)


if __name__ == "__main__":
    main()
