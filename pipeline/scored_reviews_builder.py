import sys
import pandas as pd
import ast
from pathlib import Path
from tqdm import tqdm
import json

_dir = Path(__file__).resolve().parent
sys.path[:0] = [str(_dir), str(_dir.parent)]

from config import Config
from dependencies.Glimpse_tokenizer import glimpse_tokenizer

BASE_DIR = Config.BASE_DIR

def _parse_rsa_distributions(scored_df: pd.DataFrame, review_id: str) -> dict:
    """
    Parse listener/speaker DataFrames from the GLIMPSE results CSV.

    Returns dict with:
      listener: {sentence: {R1: prob, R2: prob, ...}} — normalized probabilities
      speaker:  {R1: {sentence: prob}, ...} — normalized probabilities
    Returns empty dict if data not available (backward compat with older CSVs).
    """
    import numpy as np

    row = scored_df[scored_df["id"] == review_id].iloc[0]

    listener_json = row.get("listener_df") if "listener_df" in scored_df.columns else None
    speaker_json = row.get("speaker_df") if "speaker_df" in scored_df.columns else None

    if not listener_json or not speaker_json or pd.isna(listener_json) or pd.isna(speaker_json):
        return {}

    try:
        listener_df = pd.read_json(listener_json)
        speaker_df = pd.read_json(speaker_json)
    except Exception:
        return {}

    num_reviews = len(listener_df)
    review_labels = [f"R{i+1}" for i in range(num_reviews)]

    # Listener: exponentiate log-probs, normalize per column (per sentence)
    listener_probs = np.exp(listener_df.values)
    col_sums = listener_probs.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    listener_probs = listener_probs / col_sums
    listener = {
        sent: {review_labels[i]: float(listener_probs[i, j]) for i in range(num_reviews)}
        for j, sent in enumerate(listener_df.columns)
    }

    # Speaker: exponentiate log-probs, normalize per row (per review)
    speaker_probs = np.exp(speaker_df.values)
    row_sums = speaker_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    speaker_probs = speaker_probs / row_sums
    speaker = {
        review_labels[i]: {sent: float(speaker_probs[i, j]) for j, sent in enumerate(speaker_df.columns)}
        for i in range(num_reviews)
    }

    return {"listener": listener, "speaker": speaker}


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

    # Pre-parse RSA distributions per submission (listener/speaker are shared across reviews)
    rsa_cache = {}

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

        # Parse RSA distributions (once per submission)
        if review_id not in rsa_cache:
            rsa_cache[review_id] = _parse_rsa_distributions(scored_df, review_id)

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

    return scored_reviews, rsa_cache


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
            scored_reviews, rsa_cache = preprocessed_scores(
                original_csv_path,
                scored_csv_path,
                polarity_csv_path,
                topic_csv_path,
                raw_data_csv_path,
            )

            # Load original data to extract rebuttals
            original_df = pd.read_csv(original_csv_path)

            # Load paper titles from raw data CSV (processed CSVs lack paper_title)
            paper_titles = {}
            if raw_data_csv_path.exists():
                try:
                    raw_df = pd.read_csv(raw_data_csv_path, usecols=["id", "paper_title"])
                    paper_titles = {
                        row["id"]: str(row["paper_title"])
                        for _, row in raw_df.iterrows()
                        if pd.notna(row.get("paper_title", ""))
                    }
                except Exception as e:
                    print(f"Warning: Could not load paper titles from {raw_data_csv_path}: {e}")

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
                    'paper_title': paper_titles.get(review_id, ''),
                    'has_rebuttal': bool(rebuttal_str.strip()) if rebuttal_str else False,
                }

            # Merge RSA distributions into metadata (listener/speaker per submission)
            for review_id, rsa_data in rsa_cache.items():
                if rsa_data and review_id in review_metadata:
                    review_metadata[review_id]['rsa'] = rsa_data

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

