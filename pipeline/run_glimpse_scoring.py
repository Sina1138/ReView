#!/usr/bin/env python3
"""
Clean GLIMPSE scoring pipeline - independent from glimpse directory mess.
Handles candidate generation, RSA scoring, and CSV conversion automatically.
"""

import argparse
import subprocess
import sys
import pickle
import pandas as pd
import json
import re
from pathlib import Path
from tqdm import tqdm

# Ensure sibling modules and project root are importable
_dir = Path(__file__).resolve().parent
sys.path[:0] = [str(_dir), str(_dir.parent)]

from config import Config

# Convenience alias
BASE_DIR = Config.BASE_DIR


def run_candidate_generation(year: int,
                              input_dir: Path = None,
                              output_dir: Path = None) -> Path:
    """
    Step 1: Generate extractive sentence candidates from preprocessed data.

    Returns: Path to generated candidates CSV
    """
    if input_dir is None:
        input_dir = BASE_DIR / "data" / "processed"
    if output_dir is None:
        output_dir = BASE_DIR / "data" / "glimpse_candidates"

    input_file = input_dir / f"all_reviews_{year}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Preprocessed file not found: {input_file}")

    print(f"\n{'='*60}")
    print(f"Step 1: Generating extractive candidates for {year}...")
    print(f"{'='*60}")

    # Run the glimpse candidate generation script
    cmd = [
        "python", str(Config.BASE_DIR / "glimpse/glimpse/data_loading/generate_extractive_candidates.py"),
        "--dataset_path", str(input_file),
        "--output_dir", str(output_dir),
        "--scripted-run"  # This makes it print the output path
    ]

    # Capture stdout only - let stderr stream through for progress bars
    result = subprocess.run(cmd, capture_output=False, stdout=subprocess.PIPE, text=True, check=True)

    # Extract the output path from stdout (last non-empty line)
    stdout_lines = [line for line in result.stdout.strip().split('\n') if line.strip()]
    if not stdout_lines:
        raise ValueError("No output from candidate generation script")
    output_path = Path(stdout_lines[-1].strip())

    # Rename to a clean, predictable filename
    clean_path = output_dir / f"candidates_{year}.csv"
    output_path.rename(clean_path)

    print(f"✓ Candidates saved to: {clean_path}")
    return clean_path


def run_rsa_scoring(candidates_csv: Path,
                    year: int,
                    model_name: str = "facebook/bart-large-cnn",
                    output_dir: Path = None,
                    device: str = "cuda") -> Path:
    """
    Step 2: Run RSA reranking on candidates.

    Returns: Path to generated pickle file
    """
    if output_dir is None:
        output_dir = BASE_DIR / "data" / "glimpse_output"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Step 2: Running RSA scoring for {year}...")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Output pickle file with clean name
    output_pk = output_dir / f"rsa_results_{year}.pk"

    cmd = [
        "python", str(Config.BASE_DIR / "glimpse/glimpse/src/compute_rsa.py"),
        "--summaries", str(candidates_csv),
        "--model_name", model_name,
        "--output_dir", str(output_dir),
        "--device", device,
        "--scripted-run"  # Get output path from stdout
    ]

    try:
        # Capture stdout only - let stderr stream through for progress bars
        result = subprocess.run(cmd, capture_output=False, stdout=subprocess.PIPE, text=True, check=True)

        # Extract the output path from stdout (last non-empty line)
        stdout_lines = [line for line in result.stdout.strip().split('\n') if line.strip()]
        if not stdout_lines:
            raise ValueError("No output from RSA scoring script")
        generated_pk = Path(stdout_lines[-1].strip())

        if not generated_pk.exists():
            raise FileNotFoundError(f"Generated pickle not found: {generated_pk}")

        # Rename to clean name
        generated_pk.rename(output_pk)
        print(f"✓ RSA results saved to: {output_pk}")

        return output_pk

    except subprocess.CalledProcessError as e:
        print(f"✗ RSA scoring failed for {year}")
        raise


def convert_pk_to_csv(pickle_path: Path,
                      year: int,
                      output_dir: Path = None) -> Path:
    """
    Step 3: Convert pickle results to clean CSV format.

    Returns: Path to GLIMPSE_results_{year}.csv
    """
    if output_dir is None:
        output_dir = BASE_DIR / "data"

    print(f"\n{'='*60}")
    print(f"Step 3: Converting pickle to CSV for {year}...")
    print(f"{'='*60}")

    # Load pickle file
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    results = data.get('results')

    if not isinstance(results, list):
        raise ValueError("Unexpected pickle structure")

    # Extract and flatten results — include listener/speaker distributions
    # for rich agreement visualization in the UI (R% bars, divergent cards)
    csv_data = []
    for index, result in enumerate(tqdm(results, desc="Converting")):
        row = {
            'index': index,
            'id': str(result.get('id')[0]),
            'gold': result.get('gold'),
            'consensuality_scores': json.dumps(result.get('consensuality_scores').to_dict())
                if isinstance(result.get('consensuality_scores'), pd.Series) else None,
        }

        # Save listener_df: DataFrame (N_reviews × K_sentences) of log-probs
        # Stored as JSON: {sentence: {R1: logprob, R2: logprob, ...}}
        listener_df = result.get('listener_df')
        if listener_df is not None and isinstance(listener_df, pd.DataFrame):
            row['listener_df'] = listener_df.to_json()
        else:
            row['listener_df'] = None

        # Save speaker_df: DataFrame (N_reviews × K_sentences) of log-probs
        speaker_df = result.get('speaker_df')
        if speaker_df is not None and isinstance(speaker_df, pd.DataFrame):
            row['speaker_df'] = speaker_df.to_json()
        else:
            row['speaker_df'] = None

        csv_data.append(row)

    # Save to expected location
    output_csv = output_dir / f"GLIMPSE_results_{year}.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv, index=False)

    print(f"✓ GLIMPSE results saved to: {output_csv}")
    print(f"  Rows: {len(df)}")

    return output_csv


def run_glimpse_pipeline(year: int,
                         model_name: str = "facebook/bart-large-cnn",
                         device: str = "cuda",
                         skip_if_exists: bool = True):
    """
    Run the complete GLIMPSE scoring pipeline for a single year.
    """
    final_csv = BASE_DIR / "data" / f"GLIMPSE_results_{year}.csv"

    if skip_if_exists and final_csv.exists():
        print(f"\n⏩ Skipping {year} - GLIMPSE results already exist at {final_csv}")
        return final_csv

    print(f"\n{'#'*60}")
    print(f"# GLIMPSE Scoring Pipeline: {year}")
    print(f"{'#'*60}")

    try:
        # Step 1: Generate candidates
        candidates_csv = run_candidate_generation(year)

        # Step 2: Run RSA scoring
        pickle_path = run_rsa_scoring(candidates_csv, year, model_name, device=device)

        # Step 3: Convert to CSV
        output_csv = convert_pk_to_csv(pickle_path, year)

        print(f"\n{'='*60}")
        print(f"✓ GLIMPSE pipeline complete for {year}")
        print(f"  Output: {output_csv}")
        print(f"{'='*60}")

        return output_csv

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ GLIMPSE pipeline failed for {year}: {e}")
        print(f"{'='*60}")
        raise


def find_available_preprocessed_years(data_dir: Path = None):
    """Auto-detect years by scanning processed data directory."""
    if data_dir is None:
        data_dir = BASE_DIR / "data" / "processed"

    years = []
    if data_dir.exists():
        for file in data_dir.glob("all_reviews_*.csv"):
            match = re.search(r'all_reviews_(\d{4})\.csv', file.name)
            if match:
                years.append(int(match.group(1)))

    return sorted(years)


def main():
    parser = argparse.ArgumentParser(
        description="Clean GLIMPSE scoring pipeline for ICLR review data"
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Single year to process (if not specified, auto-detects all available years)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/bart-large-cnn",
        help="Model for RSA scoring (default: facebook/bart-large-cnn)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (default: cuda)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if results exist"
    )

    args = parser.parse_args()

    # Process single year or auto-detect all available years
    if args.year:
        years = [args.year]
    else:
        years = find_available_preprocessed_years()
        if not years:
            print("⚠️  No preprocessed data found in data/processed/")
            print("   Run preprocess_data.py first")
            return

    print(f"\n{'='*60}")
    print(f"GLIMPSE Scoring Pipeline")
    print(f"Years: {list(years)}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"{'='*60}")

    success_count = 0
    failed_years = []

    for year in years:
        try:
            run_glimpse_pipeline(
                year,
                model_name=args.model,
                device=args.device,
                skip_if_exists=not args.force
            )
            success_count += 1
        except Exception as e:
            failed_years.append(year)
            print(f"\n⚠️  Failed to process {year}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Pipeline Summary")
    print(f"{'='*60}")
    print(f"✓ Successful: {success_count}/{len(years)} years")
    if failed_years:
        print(f"✗ Failed: {failed_years}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
