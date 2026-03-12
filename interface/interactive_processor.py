"""
Interactive Tab Processing Module
Aligns interactive review processing with the preprocessed pipeline.
"""

import sys
import os
from pathlib import Path
import torch
import math
import json
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import pandas as pd
import re

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from dependencies.rsa_reranker import RSAReranking
from dependencies.Glimpse_tokenizer import glimpse_tokenizer
from dependencies.sentence_filter import (
    is_section_header, is_noise_sentence, filter_and_clean_sentences,
    strip_header_prefix, HIGHLIGHT_THRESHOLD,
)

# Try to import OpenReview, but don't fail if not available
try:
    import openreview
    OPENREVIEW_AVAILABLE = True
except ImportError:
    OPENREVIEW_AVAILABLE = False


class InteractiveReviewProcessor:
    """Process reviews through the same pipeline as preprocessed data."""

    def __init__(self, device: str = "cuda"):
        """Initialize processor with all required models."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load summarization model (for RSA)
        rsa_model_name = "sshleifer/distilbart-cnn-12-3"
        self.rsa_model = AutoModelForSeq2SeqLM.from_pretrained(
            rsa_model_name,
            # Use float32 on all devices for accuracy (validation showed float16 fails on edge cases)
            # CPU optimization priority: algorithmic improvements give 40-50% speedup with perfect accuracy
            torch_dtype=torch.float32
        )
        self.rsa_tokenizer = AutoTokenizer.from_pretrained(rsa_model_name)
        self.rsa_model.to(self.device)
        self.rsa_model.eval()

        # Load polarity model
        # Option A (Feb 2026): DeBERTa-v3-base for +5.5% F1 improvement (0.764 vs 0.724 SciBERT)
        # Try local trained model first, fall back to HuggingFace
        polarity_model_local = BASE_DIR / "training" / "outputs" / "deberta_polarity" / "final_model"
        if polarity_model_local.exists() and (polarity_model_local / "config.json").exists():
            polarity_model_name = str(polarity_model_local)
            print(f"Loading polarity model from local trained model: {polarity_model_name}")
        else:
            # Fallback: will need to upload fine-tuned model or use legacy SciBERT
            polarity_model_name = "Sina1138/Scibert_polarity_Review"  # Legacy SciBERT
            print(f"Local model not found, using legacy SciBERT: {polarity_model_name}")

        self.polarity_tokenizer = AutoTokenizer.from_pretrained(polarity_model_name)
        self.polarity_model = AutoModelForSequenceClassification.from_pretrained(polarity_model_name)
        self.polarity_model.to(self.device)
        self.polarity_model.eval()

        # Load topic model
        # SciDeBERTa maintains best performance (F1=0.478)
        topic_model_local = BASE_DIR / "training" / "outputs" / "scideberta_topic" / "final_model"
        if topic_model_local.exists() and (topic_model_local / "config.json").exists():
            topic_model_name = str(topic_model_local)
            print(f"Loading topic model from local trained model: {topic_model_name}")
        else:
            topic_model_name = "Sina1138/SciDeberta_Review"  # Production HuggingFace model
            print(f"Using HuggingFace topic model: {topic_model_name}")

        self.topic_tokenizer = AutoTokenizer.from_pretrained(topic_model_name)
        self.topic_model = AutoModelForSequenceClassification.from_pretrained(topic_model_name)
        self.topic_model.to(self.device)
        self.topic_model.eval()

        # Topic ID to label mapping
        self.id2topic = {
            0: "Substance",
            1: "Clarity",
            2: "Soundness/Correctness",
            3: "Originality",
            4: "Motivation/Impact",
            5: "Meaningful Comparison",
            6: "Replicability",
            7: None  # Unclassified
        }

    def predict_polarity(self, sentences: List[str]) -> Dict[str, Optional[str]]:
        """
        Predict polarity for sentences.
        Returns: {sentence: "➕" | "➖" | None}
        """
        if not sentences:
            return {}

        inputs = self.polarity_tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self.polarity_model(**inputs).logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()

        emoji_map = {0: "➖", 1: None, 2: "➕"}
        return dict(zip(sentences, [emoji_map.get(p, None) for p in preds]))

    def predict_topic(self, sentences: List[str]) -> Dict[str, Optional[str]]:
        """
        Predict topic for sentences.
        Returns: {sentence: topic_label | None}
        """
        if not sentences:
            return {}

        inputs = self.topic_tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self.topic_model(**inputs).logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()

        return dict(zip(sentences, [self.id2topic.get(p, None) for p in preds]))

    def predict_consensuality(
        self,
        *texts: str,
        rationality: float = 1.0,
        iterations: int = 1
    ) -> Dict[str, float]:
        """
        Predict consensuality using RSA reranking.
        Accepts 2+ review texts.
        Returns: {sentence: consensuality_score}
        """
        texts = [t for t in texts if t and t.strip()]
        if len(texts) < 2:
            return {}

        # Tokenize all reviews
        all_sentence_lists = [[s for s in glimpse_tokenizer(t) if s.strip()] for t in texts]

        # Get unique sentences, filtering out noise (headers, citations, short fragments, etc.)
        unique_sentences = list(set(s for lst in all_sentence_lists for s in lst))
        sentences = filter_and_clean_sentences(unique_sentences)

        if not sentences:
            return {}

        # Run RSA reranking
        rsa_reranker = RSAReranking(
            self.rsa_model,
            self.rsa_tokenizer,
            candidates=sentences,
            source_texts=list(texts),
            device=str(self.device),
            rationality=rationality,
        )

        _, _, _, _, _, _, _, consensuality_scores = rsa_reranker.rerank(t=iterations)

        # Robust normalization: median-centered, IQR-scaled, clipped to [-1, 1]
        # This avoids outliers dominating the color scale
        import numpy as np
        scores = consensuality_scores.copy()
        vals = scores.values
        median = np.median(vals)
        q25, q75 = np.percentile(vals, 25), np.percentile(vals, 75)
        iqr = q75 - q25
        if iqr > 0:
            # Center on median, scale so IQR spans ~[-0.5, 0.5], clip to [-1, 1]
            scores = ((scores - median) / (iqr * 2)).clip(-1, 1)
        else:
            # All scores identical or near-identical
            scores = scores * 0

        return dict(scores)

    def format_highlighted_output(
        self,
        sentences: List[str],
        scores_dict: Dict[str, float],
        score_type: str = "consensuality"
    ) -> List[Tuple[str, Optional[float]]]:
        """
        Format output for HighlightedText component.

        Args:
            sentences: List of sentences in order
            scores_dict: Dictionary mapping sentences to scores
            score_type: "none", "consensuality", "polarity", or "topic"

        Returns:
            List of (sentence, score) tuples
        """
        if score_type == "none":
            # Show original text without any highlighting/scores
            return [(s, None) for s in sentences]
        elif score_type == "consensuality":
            return [
                (s, scores_dict.get(s, 0.0)
                 if isinstance(scores_dict.get(s), (int, float))
                    and abs(scores_dict.get(s, 0.0)) >= HIGHLIGHT_THRESHOLD
                 else None)
                for s in sentences
            ]
        else:  # polarity or topic
            return [
                (s, scores_dict.get(s, None))
                for s in sentences
            ]

    def process_reviews_fast(self, *reviews: str) -> Dict:
        """
        Process reviews WITHOUT RSA (fast path: ~3-5 sec on CPU).

        Returns polarity + topic scores immediately.
        RSA can be computed separately in background.

        Args:
            reviews: Review texts (at least 2 required)

        Returns:
            Dictionary with polarity + topic scores (consensuality empty)
        """
        reviews = [r for r in reviews if r and r.strip()]
        if len(reviews) < 2:
            raise ValueError("At least two non-empty reviews are required")

        # Tokenize reviews
        sentence_lists = [[s for s in glimpse_tokenizer(r) if s.strip()] for r in reviews]

        if any(len(sl) == 0 for sl in sentence_lists):
            raise ValueError("One or more reviews have no valid sentences")

        # Get unique sentences, filtering out noise (headers, citations, short fragments, etc.)
        all_sentences = filter_and_clean_sentences(
            list(set(s for sl in sentence_lists for s in sl))
        )

        # Predict scores (skip consensuality - that comes async)
        polarity_map = self.predict_polarity(all_sentences)
        topic_map = self.predict_topic(all_sentences)

        # Return with empty consensuality (will be updated async)
        result = {
            f"review{i+1}_sentences": sl for i, sl in enumerate(sentence_lists)
        }
        result.update({
            "consensuality_scores": {},
            "polarity_scores": polarity_map,
            "topic_scores": topic_map,
        })
        result["most_common"] = []
        result["most_unique"] = []

        return result

    def process_reviews(
        self,
        *reviews: str,
        focus: str = "Agreement"
    ) -> Dict:
        """
        Process 2-6 reviews and return scored output.

        Args:
            reviews: Review texts (at least 2 required)
            focus: "Agreement", "Polarity", or "Topic"

        Returns:
            Dictionary with formatted output for all reviews
        """
        reviews = [r for r in reviews if r and r.strip()]
        if len(reviews) < 2:
            raise ValueError("At least two non-empty reviews are required")

        # Tokenize reviews
        sentence_lists = [[s for s in glimpse_tokenizer(r) if s.strip()] for r in reviews]

        if any(len(sl) == 0 for sl in sentence_lists):
            raise ValueError("One or more reviews have no valid sentences")

        # Get unique sentences, filtering out noise (headers, citations, short fragments, etc.)
        all_sentences = filter_and_clean_sentences(
            list(set(s for sl in sentence_lists for s in sl))
        )

        # Predict scores
        polarity_map = self.predict_polarity(all_sentences)
        topic_map = self.predict_topic(all_sentences)
        consensuality_map = self.predict_consensuality(*reviews)

        # Prepare output based on focus
        result = {
            f"review{i+1}_sentences": sl for i, sl in enumerate(sentence_lists)
        }
        result.update({
            "consensuality_scores": consensuality_map,
            "polarity_scores": polarity_map,
            "topic_scores": topic_map,
        })

        # Calculate most common and unique sentences
        if consensuality_map:
            scores_series = pd.Series(consensuality_map)
            result["most_common"] = scores_series.nlargest(3).index.tolist()
            result["most_unique"] = scores_series.nsmallest(3).index.tolist()
        else:
            result["most_common"] = []
            result["most_unique"] = []

        return result


def fetch_reviews_from_openreview_link(link: str) -> Tuple[List[str], str]:
    """
    Fetch reviews from an OpenReview link.

    Args:
        link: OpenReview forum link (e.g., https://openreview.net/forum?id=XXX)

    Returns:
        Tuple of (list of review texts, paper title)

    Raises:
        ValueError: If link is invalid or no OpenReview access
        Exception: If fetching fails
    """
    print(f"[FETCH] Starting fetch for link: {link}")

    if not OPENREVIEW_AVAILABLE:
        print("[FETCH] ERROR: OpenReview library not available")
        raise ValueError(
            "OpenReview library not available. Install with: pip install openreview-py"
        )

    print("[FETCH] Step 1: Extracting forum ID from link...")
    # Extract forum ID from link (more permissive regex)
    match = re.search(r'id=([^&\s]+)', link)
    if not match:
        print("[FETCH] ERROR: Invalid link format")
        raise ValueError(f"Invalid OpenReview link format. Expected: https://openreview.net/forum?id=XXX")

    forum_id = match.group(1).strip()
    print(f"[FETCH] Forum ID extracted: {forum_id}")

    try:
        print("[FETCH] Step 2: Initializing OpenReview client (guest access)...")
        # Clear credentials from environment so client uses guest access
        # (same approach as fetch_iclr_data.py)
        env_backup = {}
        for key in ['OPENREVIEW_USERNAME', 'OPENREVIEW_PASSWORD']:
            if key in os.environ:
                env_backup[key] = os.environ.pop(key)
                print(f"[FETCH]   Temporarily cleared env var: {key}")

        try:
            client = openreview.Client(baseurl='https://api.openreview.net')
        finally:
            # Restore environment variables
            for key, value in env_backup.items():
                os.environ[key] = value

        print("[FETCH] Client initialized successfully (guest mode)")

        print(f"[FETCH] Step 3: Fetching forum notes for forum: {forum_id}")
        # Get forum notes with error handling
        try:
            forum_notes = client.get_notes(forum=forum_id)
            print(f"[FETCH] Successfully fetched {len(forum_notes)} notes")
        except Exception as api_error:
            print(f"[FETCH] ERROR during API call: {type(api_error).__name__}: {str(api_error)}")
            raise ValueError(f"Failed to connect to OpenReview API: {str(api_error)}")

        if not forum_notes:
            print("[FETCH] ERROR: No forum notes returned")
            raise ValueError(f"No forum found for ID: {forum_id}")

        # Get submission to extract title
        print("[FETCH] Step 4: Extracting paper title...")
        title = "Unknown Paper"

        # Find submission - try different patterns
        for i, note in enumerate(forum_notes):
            invitation = getattr(note, 'invitation', '')
            print(f"[FETCH]   Note {i}: invitation = {invitation[:80]}")

            if any(pattern in invitation for pattern in ['Blind_Submission', '/Submission']):
                print(f"[FETCH]   Found submission note!")
                # Try different content access patterns
                content = getattr(note, 'content', {})
                if isinstance(content, dict):
                    title = content.get('title', 'Unknown Paper')
                elif hasattr(content, 'get'):
                    title = content.get('title', 'Unknown Paper')
                print(f"[FETCH]   Title extracted: {title[:100]}")
                break

        # Extract reviews - try multiple patterns
        print("[FETCH] Step 5: Extracting reviews...")
        reviews = []
        review_id_to_num = {}  # Track note IDs for rebuttal linking
        review_patterns = ['Official_Review', 'Review', 'review']

        for i, note in enumerate(forum_notes):
            invitation = getattr(note, 'invitation', '')

            # Check if this is a review note
            if any(pattern in invitation for pattern in review_patterns):
                print(f"[FETCH]   Found review note {len(reviews)+1}: {invitation[:80]}")
                # Try different content access patterns
                content = getattr(note, 'content', {})
                review_text = None

                # Try different field names
                for field_name in ['review', 'Review', 'text', 'content']:
                    if isinstance(content, dict):
                        review_text = content.get(field_name, '')
                    elif hasattr(content, 'get'):
                        review_text = content.get(field_name, '')

                    if review_text and isinstance(review_text, str):
                        print(f"[FETCH]     Found review text in field: {field_name}")
                        break

                if review_text and isinstance(review_text, str) and review_text.strip():
                    reviews.append(review_text.strip())
                    review_id_to_num[note.id] = len(reviews)  # Map note ID to review number (1-indexed)
                    print(f"[FETCH]     Review added (length: {len(review_text)} chars)")

        print(f"[FETCH] Step 6: Review extraction complete - found {len(reviews)} reviews")

        # Extract rebuttals - look for author comments/responses
        print("[FETCH] Step 6b: Extracting author rebuttals...")
        rebuttals_structured = []
        for note in forum_notes:
            invitation = getattr(note, 'invitation', '')
            if any(p in invitation for p in ['Official_Comment', 'Author_Comment', 'Comment']):
                if hasattr(note, 'signatures') and any('Authors' in sig for sig in note.signatures):
                    content = getattr(note, 'content', {})
                    text = None
                    for field in ['comment', 'rebuttal', 'title']:
                        if isinstance(content, dict):
                            text = content.get(field, '')
                        if text and isinstance(text, str) and text.strip():
                            break
                    if text and text.strip():
                        # Track which review this is replying to (if any)
                        replyto = getattr(note, 'replyto', None)
                        reply_num = review_id_to_num.get(replyto, None)
                        rebuttals_structured.append({"text": text.strip(), "reply_to": reply_num})

        # Serialize as JSON for structured display
        rebuttal_text = json.dumps(rebuttals_structured) if rebuttals_structured else ""
        print(f"[FETCH] Found {len(rebuttals_structured)} rebuttal(s)")

        if not reviews:
            print(f"[FETCH] ERROR: No reviews found. Total notes: {len(forum_notes)}")
            print(f"[FETCH] Invitations in this forum:")
            invitations = {}
            for note in forum_notes:
                inv = getattr(note, 'invitation', 'unknown')
                invitations[inv] = invitations.get(inv, 0) + 1
            for inv, count in invitations.items():
                print(f"[FETCH]   - {inv}: {count} notes")

            raise ValueError(f"No official reviews found for submission {forum_id}. "
                           f"Found {len(forum_notes)} notes total. "
                           f"Make sure the link is to a paper with reviews.")

        print(f"[FETCH] SUCCESS: Returning {len(reviews)} reviews, title: {title[:50]}, rebuttal: {len(rebuttal_text)} chars")
        return reviews, title, rebuttal_text

    except ValueError as ve:
        print(f"[FETCH] ValueError raised: {str(ve)}")
        raise
    except Exception as e:
        print(f"[FETCH] Unexpected exception: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"[FETCH] Traceback: {traceback.format_exc()}")
        raise Exception(f"Failed to fetch from OpenReview: {str(e)}. "
                       f"Check your internet connection and verify the forum ID is correct.")

