"""
Fetch ICLR 2022-2025 reviews and rebuttals from OpenReview API.
Based on DISAPERE data_prep_lib.py patterns.
"""

import openreview
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICLRDataFetcher:
    """Fetch ICLR reviews and rebuttals from OpenReview API."""

    def __init__(self, base_url='https://api2.openreview.net'):
        """
        Initialize OpenReview client.

        Note: No authentication needed for public ICLR data.
        Uses guest client by default.
        """
        logger.info(f"Connecting to OpenReview API at {base_url}")
        self.client = openreview.api.OpenReviewClient(baseurl=base_url)

    def get_venue_id(self, year: int) -> str:
        """Get OpenReview venue ID for ICLR in a given year."""
        # ICLR venue ID format changed over years
        venue_formats = {
            2022: 'ICLR.cc/2022/Conference',
            2023: 'ICLR.cc/2023/Conference',
            2024: 'ICLR.cc/2024/Conference',
            2025: 'ICLR.cc/2025/Conference',
        }
        return venue_formats.get(year, f'ICLR.cc/{year}/Conference')

    def fetch_submissions(self, venue_id: str) -> List:
        """Fetch all submissions for a venue."""
        logger.info(f"Fetching submissions for {venue_id}")

        # Get all submissions (papers)
        submissions = self.client.get_all_notes(
            invitation=f'{venue_id}/-/Blind_Submission',
            details='directReplies'  # Include reviews and rebuttals as replies
        )

        logger.info(f"Found {len(submissions)} submissions")
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

        # Get paper metadata
        paper_title = submission.content.get('title', {}).get('value', '')
        abstract = submission.content.get('abstract', {}).get('value', '')

        # Get meta-review (if exists)
        metareview_text = ''
        metareview_conf = ''
        decision = ''

        # Check for meta-review in direct replies
        if hasattr(submission, 'details') and 'directReplies' in submission.details:
            for reply in submission.details['directReplies']:
                if 'Meta_Review' in reply.get('invitation', ''):
                    metareview_text = reply.content.get('metareview', {}).get('value', '')
                    metareview_conf = reply.content.get('confidence', {}).get('value', '')
                    decision = reply.content.get('recommendation', {}).get('value', '')
                    break

        # Get official reviews
        reviews = []
        if hasattr(submission, 'details') and 'directReplies' in submission.details:
            for reply in submission.details['directReplies']:
                # Check if this is an official review
                if 'Official_Review' in reply.get('invitation', ''):
                    reviews.append(reply)

        # Extract data for each review
        for review_note in reviews:
            review_id = review_note.id

            # Extract review content
            review_content = review_note.content
            review_text = review_content.get('review', {}).get('value', '')

            # Handle different rating formats
            rating_field = review_content.get('rating', {}).get('value', '')
            if isinstance(rating_field, str):
                # Format like "6: Marginally above acceptance threshold"
                rating = rating_field.split(':')[0].strip() if ':' in rating_field else rating_field
            else:
                rating = str(rating_field)

            confidence = review_content.get('confidence', {}).get('value', '')

            # Get reviewer signature
            reviewer = review_note.signatures[0] if review_note.signatures else 'Anonymous'

            # Find rebuttal (author response to this review)
            rebuttal_text = self._find_rebuttal_for_review(submission, review_id)

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

    def _find_rebuttal_for_review(self, submission, review_id: str) -> str:
        """
        Find author rebuttal that replies to a specific review.

        Simple approach: Look for notes that:
        1. Are posted by authors
        2. Reply to (replyto field = review_id)

        If multiple rebuttals found, concatenate with separator.
        If none found, return empty string.
        """
        rebuttals = []

        if hasattr(submission, 'details') and 'directReplies' in submission.details:
            for reply in submission.details['directReplies']:
                # Check if this is an official review
                if 'Official_Review' in reply.get('invitation', ''):
                    # Check replies to this review
                    if hasattr(reply, 'details') and 'directReplies' in reply.details:
                        for subreply in reply.details['directReplies']:
                            # Check if posted by authors
                            if subreply.get('replyto') == reply.id:
                                # This is likely a rebuttal
                                rebuttal_content = subreply.content.get('comment', {}).get('value', '')
                                if rebuttal_content:
                                    rebuttals.append(rebuttal_content)

        # Concatenate multiple rebuttals with separator
        if len(rebuttals) > 1:
            return "\n\n--- ADDITIONAL RESPONSE ---\n\n".join(rebuttals)
        elif len(rebuttals) == 1:
            return rebuttals[0]
        else:
            return ''  # No rebuttal found

    def fetch_year(self, year: int, output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Fetch all ICLR data for a given year.

        Args:
            year: Conference year (2022-2025)
            output_path: Where to save CSV (optional)

        Returns:
            DataFrame with all reviews and rebuttals
        """
        logger.info(f"Starting fetch for ICLR {year}")

        venue_id = self.get_venue_id(year)
        submissions = self.fetch_submissions(venue_id)

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
            df.to_csv(output_path, index=False)
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
    """Fetch ICLR 2022-2025 data."""

    # Configuration
    years_to_fetch = [2022, 2023, 2024, 2025]
    output_dir = Path(__file__).parent / 'glimpse' / 'data'

    # Initialize fetcher
    fetcher = ICLRDataFetcher()

    # Fetch each year
    for year in years_to_fetch:
        logger.info(f"\n{'='*60}")
        logger.info(f"FETCHING ICLR {year}")
        logger.info('='*60)

        output_path = output_dir / f'all_reviews_{year}.csv'

        try:
            df = fetcher.fetch_year(year, output_path=output_path)
            fetcher.validate_dataframe(df, year)

        except Exception as e:
            logger.error(f"Failed to fetch {year}: {e}")
            continue

    logger.info("\n✓ Data fetching complete!")
    logger.info(f"Files saved to: {output_dir}")


if __name__ == '__main__':
    main()
