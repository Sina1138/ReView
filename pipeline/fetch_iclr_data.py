"""
Fetch ICLR reviews and rebuttals from OpenReview API.
Based on DISAPERE data_prep_lib.py patterns.
"""

import openreview
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import os
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICLRDataFetcher:
    """Fetch ICLR reviews and rebuttals from OpenReview API."""

    def __init__(self, base_url='https://api.openreview.net'):
        """
        Initialize OpenReview client.

        Note: No authentication needed for public ICLR data.
        Uses guest client (V1 API) following DISAPERE patterns.
        """
        logger.info(f"Connecting to OpenReview API at {base_url}")

        # Clear environment credentials to force guest access
        # (V1 Client also picks up these environment variables)
        env_backup = {}
        for key in ['OPENREVIEW_USERNAME', 'OPENREVIEW_PASSWORD']:
            if key in os.environ:
                env_backup[key] = os.environ.pop(key)

        try:
            # Use V1 API client (same as DISAPERE)
            # No credentials needed for guest access to public data
            self.client = openreview.Client(baseurl=base_url)
        finally:
            # Restore environment variables
            for key, value in env_backup.items():
                os.environ[key] = value

    def get_venue_id(self, year: int) -> str:
        """Get OpenReview venue ID for ICLR in a given year."""
        return f'ICLR.cc/{year}/Conference'

    def fetch_submissions(self, venue_id: str) -> List:
        """Fetch all submissions for a venue."""
        logger.info(f"Fetching submissions for {venue_id}")

        # Get all submissions (papers) using V1 API
        # Try different invitation patterns (format changed over years)
        invitation_patterns = [
            f'{venue_id}/-/Blind_Submission',
            f'{venue_id}/-/Submission',
        ]

        submissions = []
        for pattern in invitation_patterns:
            try:
                submissions = list(openreview.tools.iterget_notes(
                    self.client,
                    invitation=pattern
                ))
                if submissions:
                    logger.info(f"Found {len(submissions)} submissions with pattern: {pattern}")
                    break
            except Exception as e:
                logger.debug(f"Pattern {pattern} failed: {e}")
                continue

        if not submissions:
            logger.warning(f"No submissions found for {venue_id}")

        return submissions

    def extract_reviews_and_rebuttals(self, submission) -> List[Dict]:
        """
        Extract reviews and rebuttals from a submission.

        Returns list of dicts, one per review, with schema:
        - id: forum URL
        - paper_title: title
        - abstract: abstract
        - reviewer: reviewer name/ID
        - review: review text
        - rating: numerical rating
        - conf_rev: confidence
        - metareview: meta-review text (if available)
        - conf_meta: meta-review confidence
        - recommendation: accept/reject
        - rebuttal: author response to this review (NEW)
        """
        rows = []

        forum_id = submission.id
        forum_url = f"https://openreview.net/forum?id={forum_id}"

        # Get paper metadata (V1 API format - direct values, not nested dicts)
        paper_title = submission.content.get('title', '')
        abstract = submission.content.get('abstract', '')

        # Get all notes for this forum to find reviews, meta-reviews, and rebuttals
        forum_notes = self.client.get_notes(forum=forum_id)

        # Find meta-review
        metareview_text = ''
        metareview_conf = ''
        decision = ''

        for note in forum_notes:
            if 'Meta_Review' in note.invitation:
                metareview_text = note.content.get('metareview', '')
                metareview_conf = note.content.get('confidence', '')
                decision = note.content.get('recommendation', '')
                break

        # Find all official reviews
        reviews = [note for note in forum_notes if 'Official_Review' in note.invitation]

        # Extract data for each review
        for review_note in reviews:
            review_id = review_note.id

            # Extract review content (V1 API - direct values)
            review_content = review_note.content
            review_text = review_content.get('review', '')

            # Handle different rating formats
            rating_field = review_content.get('rating', '')
            if isinstance(rating_field, str):
                # Format like "6: Marginally above acceptance threshold"
                rating = rating_field.split(':')[0].strip() if ':' in rating_field else rating_field
            else:
                rating = str(rating_field)

            confidence = review_content.get('confidence', '')

            # Get reviewer signature
            reviewer = review_note.signatures[0] if review_note.signatures else 'Anonymous'

            # Find rebuttal (author response to this review)
            rebuttal_text = self._find_rebuttal_for_review(forum_notes, review_id)

            row = {
                'id': forum_url,
                'paper_title': paper_title,
                'abstract': abstract,
                'reviewer': reviewer,
                'review': review_text,
                'rating': rating,
                'conf_rev': confidence,
                'metareview': metareview_text,
                'conf_meta': metareview_conf,
                'recommendation': decision,
                'rebuttal': rebuttal_text,  # NEW COLUMN
            }

            rows.append(row)

        return rows

    def _find_rebuttal_for_review(self, forum_notes: List, review_id: str) -> str:
        """
        Find author rebuttal that replies to a specific review.

        Simple approach: Look for notes that:
        1. Are posted by authors (check 'Official_Comment' or 'Author.*Comment')
        2. Reply to (replyto field = review_id)

        If multiple rebuttals found, concatenate with separator.
        If none found, return empty string.
        """
        rebuttals = []

        # Look through all forum notes for replies to this review
        for note in forum_notes:
            # Check if this note is a reply to the review
            if hasattr(note, 'replyto') and note.replyto == review_id:
                # Check if it's an author comment/rebuttal
                invitation = note.invitation if hasattr(note, 'invitation') else ''

                # Patterns that indicate author rebuttals
                author_patterns = ['Official_Comment', 'Author.*Comment', 'Comment']
                is_author_reply = any(pattern in invitation for pattern in author_patterns)

                # Also check signatures for author markers
                if hasattr(note, 'signatures'):
                    is_author_reply = is_author_reply or any('Authors' in sig for sig in note.signatures)

                if is_author_reply:
                    # Extract comment text (field name varies: 'comment', 'rebuttal', 'title')
                    rebuttal_content = (note.content.get('comment', '') or
                                       note.content.get('rebuttal', '') or
                                       note.content.get('title', ''))
                    if rebuttal_content:
                        rebuttals.append(rebuttal_content)

        # Concatenate multiple rebuttals with separator
        if len(rebuttals) > 1:
            return "\n\n--- ADDITIONAL RESPONSE ---\n\n".join(rebuttals)
        elif len(rebuttals) == 1:
            return rebuttals[0]
        else:
            return ''  # No rebuttal found

    def fetch_year(self, year: int, output_path: Optional[Path] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch all ICLR data for a given year.

        Args:
            year: Conference year
            output_path: Where to save CSV (optional)
            limit: Limit number of papers to process (for testing)

        Returns:
            DataFrame with all reviews and rebuttals
        """
        logger.info(f"Starting fetch for ICLR {year}")

        venue_id = self.get_venue_id(year)
        submissions = self.fetch_submissions(venue_id)

        # Limit submissions for testing if specified
        if limit is not None:
            logger.info(f"⚠️  TEST MODE: Limiting to {limit} submissions (out of {len(submissions)})")
            submissions = submissions[:limit]

        all_rows = []

        # Process each submission
        for submission in tqdm(submissions, desc=f"Processing {year} submissions"):
            try:
                rows = self.extract_reviews_and_rebuttals(submission)
                all_rows.extend(rows)

                # Rate limiting: sleep briefly between submissions
                time.sleep(0.1)

            except Exception as e:
                logger.warning(f"Error processing submission {submission.id}: {e}")
                continue

        # Create DataFrame
        df = pd.DataFrame(all_rows)

        logger.info(f"Extracted {len(df)} reviews for {year}")
        logger.info(f"Reviews with rebuttals: {(df['rebuttal'] != '').sum()}")

        # Save to CSV if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Use QUOTE_ALL to properly escape newlines and quotes in text fields
            import csv
            df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
            logger.info(f"Saved to {output_path}")

        return df

    def validate_dataframe(self, df: pd.DataFrame, year: int):
        """Validate fetched data quality."""
        logger.info(f"Validating data for {year}")

        # Check row count
        if len(df) < 100:
            logger.warning(f"Low review count for {year}: {len(df)} (expected 500+)")

        # Check required columns
        required_cols = ['id', 'paper_title', 'review', 'rating', 'rebuttal']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check for empty reviews
        empty_reviews = df['review'].isna().sum()
        if empty_reviews > 0:
            logger.warning(f"{empty_reviews} reviews have missing text")

        # Rebuttal statistics
        total_reviews = len(df)
        with_rebuttals = (df['rebuttal'] != '').sum()
        rebuttal_pct = (with_rebuttals / total_reviews * 100) if total_reviews > 0 else 0

        logger.info(f"Validation summary for {year}:")
        logger.info(f"  Total reviews: {total_reviews}")
        logger.info(f"  With rebuttals: {with_rebuttals} ({rebuttal_pct:.1f}%)")
        logger.info(f"  Unique papers: {df['id'].nunique()}")


def main():
    """Fetch ICLR data with rebuttals."""
    import argparse
    from datetime import datetime

    current_year = datetime.now().year

    parser = argparse.ArgumentParser(
        description='Fetch ICLR review data from OpenReview API'
    )
    parser.add_argument(
        '--year',
        type=int,
        help='Fetch single year only'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2020,
        help='Start year for batch fetch (default: 2020)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=current_year,
        help=f'End year for batch fetch (default: {current_year})'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).resolve().parent.parent / 'data',
        help='Output directory for CSV files'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of papers to process (for testing)'
    )

    args = parser.parse_args()

    # Determine years to fetch
    if args.year:
        years_to_fetch = [args.year]
    else:
        years_to_fetch = list(range(args.start_year, args.end_year + 1))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize fetcher
    fetcher = ICLRDataFetcher()

    # Fetch each year
    for year in years_to_fetch:
        logger.info(f"\n{'='*60}")
        logger.info(f"FETCHING ICLR {year}")
        logger.info('='*60)

        output_path = output_dir / f'all_reviews_{year}.csv'

        try:
            df = fetcher.fetch_year(year, output_path=output_path, limit=args.limit)
            fetcher.validate_dataframe(df, year)

        except Exception as e:
            logger.error(f"Failed to fetch {year}: {e}")
            continue

    logger.info("\n✓ Data fetching complete!")
    logger.info(f"Files saved to: {output_dir}")


if __name__ == '__main__':
    main()
