import sys
import pandas as pd
import nltk
import ast
from pathlib import Path
from tqdm import tqdm
import json

_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_dir))
sys.path.insert(0, str(_dir.parent))

from config import Config
from dependencies.Glimpse_tokenizer import glimpse_tokenizer

BASE_DIR = Config.BASE_DIR

# def tokenize_sentences(text: str) -> list:
#     # same tokenization as in the original glimpse code
#     text = text.replace('-----', '\n')
#     sentences = nltk.sent_tokenize(text)
#     sentences = [sentence for sentence in sentences if sentence != ""]
#     return sentences        
        
        
def preprocessed_scores(
        original_csv_path: Path,
        scored_csv_path: Path,
        polarity_csv_path: Path,
        topic_csv_path: Path,
        raw_data_csv_path: Path = None,
    ) -> dict:

    original_df = pd.read_csv(original_csv_path)
    scored_df = pd.read_csv(scored_csv_path)
    polarity_df = pd.read_csv(polarity_csv_path)
    topic_df = pd.read_csv(topic_csv_path)

    # Load raw data for rebuttals if available
    rebuttals_df = None
    if raw_data_csv_path and raw_data_csv_path.exists():
        try:
            rebuttals_df = pd.read_csv(raw_data_csv_path)
            rebuttal_count = rebuttals_df['rebuttal'].notna().sum() if 'rebuttal' in rebuttals_df.columns else 0
            print(f"Loaded rebuttals from {raw_data_csv_path}")
            print(f"  Found {len(rebuttals_df)} rows, {rebuttal_count} with non-null rebuttals")
        except Exception as e:
            print(f"Warning: Could not load rebuttals from {raw_data_csv_path}: {e}")

    scored_reviews = {}
    submission_review_counters = {}  # Track which review # we're on for each submission

    for _, row in tqdm(original_df.iterrows(), total=len(original_df)):
        review_id = row["id"]
        review_text = row["text"]

        if review_id not in scored_df["id"].values or review_id not in polarity_df["id"].values:
            continue

        if review_id not in scored_reviews:
            scored_reviews[review_id] = []

        # Track review number for this submission
        if review_id not in submission_review_counters:
            submission_review_counters[review_id] = 0
        else:
            submission_review_counters[review_id] += 1

        review_num = submission_review_counters[review_id]

        # Get consensuality scores
        consensuality_scores_str = scored_df[scored_df["id"] == review_id]["consensuality_scores"].iloc[0]
        try:
            consensuality_scores_dict = json.loads(consensuality_scores_str)
        except Exception as e:
            print(f"Error parsing consensuality scores for ID {review_id}: {e}")
            print("Problematic string:", consensuality_scores_str)
            continue  # skip this problematic entry

        # Get polarity scores
        polarity_rows = polarity_df[polarity_df["id"] == review_id]
        polarity_dict = dict(zip(polarity_rows["sentence"], polarity_rows["polarity"]))

        # Get topic scores
        topic_rows = topic_df[topic_df["id"] == review_id]
        topic_dict = dict(zip(topic_rows["sentence"], topic_rows["topic"]))

        # Get rebuttal if available - match by review number within submission
        rebuttal = ""
        if rebuttals_df is not None and review_id in rebuttals_df["id"].values:
            submission_reviews = rebuttals_df[rebuttals_df["id"] == review_id]
            if not submission_reviews.empty and "rebuttal" in submission_reviews.columns:
                if review_num < len(submission_reviews):
                    rebuttal_val = submission_reviews["rebuttal"].iloc[review_num]
                    rebuttal = str(rebuttal_val) if pd.notna(rebuttal_val) else ""
                    # Debug output
                    if review_id == "https://openreview.net/forum?id=ryxz8CVYDH":
                        print(f"DEBUG: {review_id[:50]}... review #{review_num+1}/{len(submission_reviews)}")
                        print(f"       rebuttal_val type: {type(rebuttal_val)}, notna: {pd.notna(rebuttal_val)}")
                        print(f"       rebuttal length: {len(rebuttal)}")

        scored_sentences = {}
        for sentence in glimpse_tokenizer(review_text):
            sentence_data = {}
            if sentence in consensuality_scores_dict:
                sentence_data["consensuality"] = consensuality_scores_dict[sentence]
            if sentence in polarity_dict:
                sentence_data["polarity"] = polarity_dict[sentence]
            if sentence in topic_dict:
                sentence_data["topic"] = topic_dict[sentence]
            if sentence_data:
                scored_sentences[sentence] = sentence_data

        scored_reviews[review_id].append({
            "sentences": scored_sentences,
            "rebuttal": rebuttal
        })

    return scored_reviews


def save_all_scored_reviews(
        start_year: int = 2017,
        end_year: int = 2021,
        input_dir: Path = BASE_DIR / "glimpse" / "data" / "processed",
        raw_data_dir: Path = BASE_DIR / "data",
        scored_csv_dir: Path = BASE_DIR / "data",
        polarity_dir: Path = BASE_DIR / "data" / "polarity_scored",
        topic_dir: Path = BASE_DIR / "data" / "topic_scored",
        output_csv_path: Path = BASE_DIR / "data" / "preprocessed_scored_reviews.csv",
    ):

    all_scored_reviews = []

    for year in range(start_year, end_year + 1):
        print(f"Processing {year}...")
        try:
            original_csv_path = input_dir / f"all_reviews_{year}.csv"
            raw_data_csv_path = raw_data_dir / f"all_reviews_{year}.csv"
            polarity_csv_path = polarity_dir / f"polarity_scored_reviews_{year}.csv"
            topic_csv_path = topic_dir / f"topic_scored_reviews_{year}.csv"
            scored_csv_path = scored_csv_dir / f"GLIMPSE_results_{year}.csv"
            scored_reviews = preprocessed_scores(
                original_csv_path,
                scored_csv_path,
                polarity_csv_path,
                topic_csv_path,
                raw_data_csv_path
            )
            all_scored_reviews.append({
                "year": year,
                "scored_dict": scored_reviews
            })

        except Exception as e:
            print(f"Skipped {year} due to error: {e}")

    df = pd.DataFrame(all_scored_reviews)
    df.to_csv(output_csv_path, index=False)
    print(f"All scored reviews saved to '{output_csv_path}'.")


def load_scored_reviews(csv_path: Path = BASE_DIR / "data" / "preprocessed_scored_reviews.csv") -> tuple:
    df = pd.read_csv(csv_path)
    tqdm.pandas(desc="Parsing scored_dict")
    df["scored_dict"] = df["scored_dict"].progress_apply(ast.literal_eval)
    years = df["year"].tolist()
    return years, df


def build_dataset(
    years: list = None,
    input_dir: Path = BASE_DIR / "data" / "processed",
    scored_csv_dir: Path = BASE_DIR / "data",
    polarity_dir: Path = BASE_DIR / "data" / "polarity_scored",
    topic_dir: Path = BASE_DIR / "data" / "topic_scored",
    output_csv_path: Path = None,
):
    """
    Build preprocessed dataset with rebuttals for any set of years.
    Auto-detects available years if none specified.
    """
    if years is None:
        # Auto-detect from processed data directory
        import re
        years = sorted(
            int(re.search(r'all_reviews_(\d{4})\.csv', f.name).group(1))
            for f in input_dir.glob("all_reviews_*.csv")
            if re.search(r'all_reviews_(\d{4})\.csv', f.name)
        )
        if not years:
            print("⚠ No processed data files found to build dataset from")
            return

    if output_csv_path is None:
        output_csv_path = BASE_DIR / "data" / f"preprocessed_scored_reviews_{min(years)}-{max(years)}.csv"

    all_scored_reviews = []

    for year in years:
        print(f"Processing {year}...")
        try:
            original_csv_path = input_dir / f"all_reviews_{year}.csv"

            # Check if files exist
            if not original_csv_path.exists():
                print(f"⚠ Skipping {year} - no processed data file found")
                continue

            polarity_csv_path = polarity_dir / f"polarity_scored_reviews_{year}.csv"
            topic_csv_path = topic_dir / f"topic_scored_reviews_{year}.csv"
            scored_csv_path = scored_csv_dir / f"GLIMPSE_results_{year}.csv"
            raw_data_csv_path = BASE_DIR / "data" / f"all_reviews_{year}.csv"

            # Use existing preprocessed_scores function
            scored_reviews = preprocessed_scores(
                original_csv_path,
                scored_csv_path,
                polarity_csv_path,
                topic_csv_path,
                raw_data_csv_path,
            )

            # Load original data to extract rebuttals
            original_df = pd.read_csv(original_csv_path)

            # Build metadata dict with rebuttals
            review_metadata = {}
            for _, row in original_df.iterrows():
                review_id = row["id"]
                rebuttal = row.get('rebuttal', '') if 'rebuttal' in original_df.columns else ''
                # Handle NaN values from pandas
                if pd.isna(rebuttal):
                    rebuttal = ''
                rebuttal_str = str(rebuttal) if rebuttal else ''

                review_metadata[review_id] = {
                    'rebuttal': rebuttal_str,
                    'paper_title': row.get('paper_title', '') if 'paper_title' in original_df.columns else '',
                    'has_rebuttal': bool(rebuttal_str.strip()) if rebuttal_str else False,
                }

            all_scored_reviews.append({
                "year": year,
                "scored_dict": scored_reviews,
                "metadata": review_metadata,
            })

        except Exception as e:
            print(f"✗ Error processing {year}: {e}")
            import traceback
            traceback.print_exc()

    # Save to CSV
    df = pd.DataFrame(all_scored_reviews)
    df.to_csv(output_csv_path, index=False)
    print(f"\n✓ Dataset saved to: {output_csv_path}")
    print(f"  Years included: {min(years)}-{max(years)}")
    print(f"  File size: {output_csv_path.stat().st_size / 1024 / 1024:.1f} MB")


# Backwards-compatible alias
build_2020_2025_dataset = build_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build preprocessed scored reviews dataset")
    parser.add_argument("--new-data", action="store_true", help="Build dataset (auto-detects years)")
    parser.add_argument("--years", type=int, nargs="+", help="Specific years to include (default: auto-detect)")
    args = parser.parse_args()

    if args.new_data or args.years:
        build_dataset(years=args.years)
    else:
        # Original behavior for legacy data
        years, all_scored_reviews_df = load_scored_reviews()
        print(years)

