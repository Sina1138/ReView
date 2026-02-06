#!/usr/bin/env python3
"""
Clean polarity scoring pipeline for ICLR review data.
Supports multiple model variants (SciBERT, DeBERTa, SciBERTa) and auto-detects available years.
"""

import argparse
import sys
import torch
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from dependencies.Glimpse_tokenizer import glimpse_tokenizer
from dependencies.scoring_utils import (
    find_available_years,
    load_polarity_model,
    predict_batch,
    save_polarity_results,
    validate_input_file,
)


def score_reviews_polarity(
    year: int,
    model_variant: str = "scibert",
    device: str = "cuda",
    input_dir: Path = None,
    output_dir: Path = None,
    skip_if_exists: bool = True,
    limit: int = None,
) -> Path:
    """
    Score reviews for polarity using specified model variant.
    
    Args:
        year: Year of reviews to score
        model_variant: Model to use ("scibert", "deberta", "scideberta")
        device: Device for computation ("cuda" or "cpu")
        input_dir: Directory containing preprocessed reviews
        output_dir: Directory to save scored results
        skip_if_exists: Skip if output already exists
        limit: Limit to first N reviews (None = process all)
        
    Returns:
        Path to output CSV file
    """
    if input_dir is None:
        input_dir = Config.BASE_DIR / "data" / "processed"
    if output_dir is None:
        output_dir = Config.POLARITY_DIR
    
    output_path = output_dir / f"polarity_scored_reviews_{year}.csv"
    
    # Skip if already exists and not forced
    if skip_if_exists and output_path.exists():
        print(f"⏩ Polarity scores already exist for {year}: {output_path}")
        return output_path
    
    print(f"\n{'='*60}")
    print(f"Polarity Scoring: {year}")
    print(f"  Model: {model_variant}")
    print(f"  Device: {device}")
    if limit:
        print(f"  Limit: {limit} reviews")
    print(f"{'='*60}")
    
    # Validate input file
    input_path = input_dir / f"all_reviews_{year}.csv"
    try:
        df = validate_input_file(input_path, required_columns=["id", "text"])
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ Input validation failed: {e}")
        raise
    
    # Apply limit if specified
    if limit:
        df = df.head(limit)
        print(f"Limited to {len(df)} reviews")
    
    # Load model
    try:
        print(f"Loading {model_variant} model...")
        tokenizer, model, device_obj = load_polarity_model(
            model_variant, Config.BASE_DIR, device
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"✗ Model loading failed: {e}")
        raise
    
    # Process reviews
    all_results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
        review_id = row["id"]
        text = row["text"]
        
        # Tokenize into sentences
        sentences = glimpse_tokenizer(text)
        if not sentences:
            continue
        
        # Predict polarity for all sentences in batch
        try:
            predictions = predict_batch(sentences, tokenizer, model, device_obj)
        except RuntimeError as e:
            print(f"✗ Prediction failed for review {review_id}: {e}")
            raise
        
        # Store results
        for sentence, polarity_label in zip(sentences, predictions):
            all_results.append({
                "id": review_id,
                "sentence": sentence,
                "polarity": polarity_label,
            })
    
    # Save results
    try:
        save_polarity_results(output_path, all_results)
        print(f"✓ Polarity scores saved: {output_path}")
        print(f"  Scored sentences: {len(all_results)}")
    except Exception as e:
        print(f"✗ Failed to save results: {e}")
        raise
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Polarity scoring pipeline for ICLR review data"
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Single year to process (if not specified, auto-detects all available years)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="scibert",
        choices=["scibert", "deberta", "scideberta"],
        help="Model variant to use (default: scibert)",
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
    
    args = parser.parse_args()
    
    # Determine years to process
    if args.year:
        years = [args.year]
    else:
        processed_dir = Config.BASE_DIR / "data" / "processed"
        years = find_available_years(processed_dir)
        if not years:
            print("⚠️  No preprocessed data found in data/processed/")
            print("   Run preprocess_data.py first")
            return
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Polarity Scoring Pipeline")
    print(f"Years: {years}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"{'='*60}")
    
    # Process each year
    success_count = 0
    failed_years = []
    
    for year in years:
        try:
            score_reviews_polarity(
                year,
                model_variant=args.model,
                device=args.device,
                skip_if_exists=not args.force,
            )
            success_count += 1
        except Exception as e:
            print(f"\n⚠️  Failed to process {year}: {e}")
            failed_years.append(year)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Pipeline Summary")
    print(f"{'='*60}")
    print(f"✓ Successful: {success_count}/{len(years)} years")
    if failed_years:
        print(f"✗ Failed: {failed_years}")
    print(f"{'='*60}\n")
    
    # Exit with error if any failed
    if failed_years:
        sys.exit(1)


if __name__ == "__main__":
    main()
