"""
Interactive Tab Processing Module
Aligns interactive review processing with the preprocessed pipeline.
"""

import sys
import os
import time
from pathlib import Path
import torch
import math
import json
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import re

# Detect ZeroGPU (HuggingFace Spaces) — CUDA can only be used inside @spaces.GPU functions
try:
    import spaces
    _ZERO_GPU = True
except ImportError:
    _ZERO_GPU = False

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from dependencies.rsa_reranker import RSARerankingCached as RSAReranking
from dependencies.Glimpse_tokenizer import glimpse_tokenizer
from dependencies.sentence_filter import (
    is_noise_sentence, filter_and_clean_sentences,
    strip_header_prefix,
)

# Try to import OpenReview, but don't fail if not available
try:
    import openreview
    OPENREVIEW_AVAILABLE = True
except ImportError:
    OPENREVIEW_AVAILABLE = False


def _try_bettertransformer(model):
    """Apply BetterTransformer (fused attention) if available. ~6% CPU speedup."""
    try:
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model)
        print(f"  BetterTransformer enabled for {model.__class__.__name__}")
    except Exception:
        pass
    return model



def _set_optimal_threads():
    """Set PyTorch thread count from SLURM allocation to avoid over/under-subscription."""
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('SLURM_CPUS_ON_NODE')
    if slurm_cpus:
        num_threads = int(slurm_cpus)
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(min(num_threads, 4))
        print(f"[THREADS] Set to {num_threads} (from SLURM)")
    else:
        print(f"[THREADS] Using PyTorch default: {torch.get_num_threads()}")


class InteractiveReviewProcessor:
    """Process reviews through the same pipeline as preprocessed data."""

    def __init__(self, device: str = "cuda"):
        """Initialize processor with all required models.

        Models always load on CPU at startup. On ZeroGPU (HF Spaces),
        GPU is only available inside @spaces.GPU-decorated functions,
        so use ensure_device() to move models to GPU dynamically.
        """
        # On ZeroGPU, CUDA must not be initialized in main process — force CPU
        # GPU is only available inside @spaces.GPU decorated functions
        if _ZERO_GPU:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        t_total = time.time()

        # Set optimal thread count for SLURM environment
        _set_optimal_threads()

        # Load summarization model (for RSA)
        t0 = time.time()
        rsa_model_name = "sshleifer/distilbart-cnn-12-3"
        self.rsa_model = AutoModelForSeq2SeqLM.from_pretrained(
            rsa_model_name,
            # Use float32 on all devices for accuracy (validation showed float16 fails on edge cases)
            # CPU optimization priority: algorithmic improvements give 40-50% speedup with perfect accuracy
            torch_dtype=torch.float32
        )
        self.rsa_tokenizer = AutoTokenizer.from_pretrained(rsa_model_name)
        self.rsa_model.to(self.device)
        # BetterTransformer DISABLED for RSA — causes 2x slowdown on DistilBart CPU
        self.rsa_model.eval()
        print(f"[TIMING] RSA model loaded in {time.time() - t0:.1f}s")

        # Load polarity model
        # Option A (Feb 2026): DeBERTa-v3-base for +5.5% F1 improvement (0.764 vs 0.724 SciBERT)
        # Try local trained model first, fall back to HuggingFace
        t0 = time.time()
        polarity_model_local = BASE_DIR / "training" / "outputs" / "deberta_polarity" / "final_model"
        if polarity_model_local.exists() and (polarity_model_local / "config.json").exists():
            polarity_model_name = str(polarity_model_local)
            print(f"Loading polarity model from local trained model: {polarity_model_name}")
        else:
            polarity_model_name = "Sina1138/deberta_polarity_Review"
            print(f"Local model not found, using HuggingFace: {polarity_model_name}")

        self.polarity_tokenizer = AutoTokenizer.from_pretrained(polarity_model_name)
        self.polarity_model = AutoModelForSequenceClassification.from_pretrained(polarity_model_name)
        self.polarity_model.to(self.device)
        self.polarity_model = _try_bettertransformer(self.polarity_model)
        self.polarity_model.eval()

        print(f"[TIMING] Polarity model loaded in {time.time() - t0:.1f}s")

        # Load topic model
        # SciDeBERTa maintains best performance (F1=0.478)
        t0 = time.time()
        topic_model_local = BASE_DIR / "training" / "outputs" / "scideberta_topic" / "final_model"
        if topic_model_local.exists() and (topic_model_local / "config.json").exists():
            topic_model_name = str(topic_model_local)
            print(f"Loading topic model from local trained model: {topic_model_name}")
        else:
            topic_model_name = "Sina1138/scideberta_topic_Review"
            print(f"Local model not found, using HuggingFace: {topic_model_name}")

        self.topic_tokenizer = AutoTokenizer.from_pretrained(topic_model_name)
        self.topic_model = AutoModelForSequenceClassification.from_pretrained(topic_model_name)
        self.topic_model.to(self.device)
        self.topic_model = _try_bettertransformer(self.topic_model)
        self.topic_model.eval()

        print(f"[TIMING] Topic model loaded in {time.time() - t0:.1f}s")

        print(f"[TIMING] All models loaded in {time.time() - t_total:.1f}s")

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

    def ensure_device(self):
        """Move all models to the best available device.

        On ZeroGPU, GPU is managed transparently — skip manual device switching.
        """
        if _ZERO_GPU:
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device != self.device:
            print(f"[DEVICE] Switching models from {self.device} to {device}")
            self.rsa_model.to(device)
            self.polarity_model.to(device)
            self.topic_model.to(device)
            self.device = device

    @staticmethod
    def _normalize_uniqueness_scores(consensuality_scores):
        """IQR-based normalization: median-centered, clipped to [-1, 1]."""
        scores = consensuality_scores.copy()
        vals = scores.values
        median = np.median(vals)
        q25, q75 = np.percentile(vals, 25), np.percentile(vals, 75)
        iqr = q75 - q25
        if iqr > 0:
            scores = ((scores - median) / (iqr * 2)).clip(-1, 1)
        else:
            scores = scores * 0
        return scores

    def predict_polarity(self, sentences: List[str], batch_size: int = 32) -> Dict[str, Optional[str]]:
        """
        Predict polarity for sentences.
        Returns: {sentence: "➕" | "➖" | None}
        """
        if not sentences:
            return {}

        self.ensure_device()
        t0 = time.time()
        n_batches = (len(sentences) + batch_size - 1) // batch_size
        print(f"[TIMING] Polarity: {len(sentences)} sentences, {n_batches} batches")
        all_preds = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = self.polarity_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)

            with torch.no_grad():
                logits = self.polarity_model(**inputs).logits
                all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())

        print(f"[TIMING] Polarity done in {time.time() - t0:.1f}s")
        emoji_map = {0: "➖", 1: None, 2: "➕"}
        return dict(zip(sentences, [emoji_map.get(p, None) for p in all_preds]))

    def predict_topic(self, sentences: List[str], batch_size: int = 32) -> Dict[str, Optional[str]]:
        """
        Predict topic for sentences.
        Returns: {sentence: topic_label | None}
        """
        if not sentences:
            return {}

        self.ensure_device()
        t0 = time.time()
        n_batches = (len(sentences) + batch_size - 1) // batch_size
        print(f"[TIMING] Topic: {len(sentences)} sentences, {n_batches} batches")
        all_preds = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = self.topic_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)

            with torch.no_grad():
                logits = self.topic_model(**inputs).logits
                all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())

        print(f"[TIMING] Topic done in {time.time() - t0:.1f}s")
        return dict(zip(sentences, [self.id2topic.get(p, None) for p in all_preds]))

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

        self.ensure_device()

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
        scores = self._normalize_uniqueness_scores(consensuality_scores)
        return dict(scores)

    def predict_rsa_full(
        self,
        *texts: str,
        rationality: float = 1.0,
        iterations: int = 1,
        progress_callback=None,
    ) -> Dict:
        """
        Full RSA computation exposing all GLIMPSE math variables.

        Returns a dict with:
          uniqueness  — {sentence: normalized_score in [-1,1]}  (same as predict_consensuality)
          listener    — {sentence: {R1: prob, R2: prob, ...}}    L_t(d|s) distribution
          speaker     — {R1: {sentence: prob}, ...}              S_t(s|d) distribution
          best_rsa    — {R1: sentence, R2: sentence, ...}        most characteristic sentence per review
        """
        texts = [t for t in texts if t and t.strip()]
        if len(texts) < 2:
            return {}

        self.ensure_device()

        all_sentence_lists = [[s for s in glimpse_tokenizer(t) if s.strip()] for t in texts]
        unique_sentences = list(set(s for lst in all_sentence_lists for s in lst))
        sentences = filter_and_clean_sentences(unique_sentences)

        if not sentences:
            return {}

        rsa_reranker = RSAReranking(
            self.rsa_model,
            self.rsa_tokenizer,
            candidates=sentences,
            source_texts=list(texts),
            device=str(self.device),
            rationality=rationality,
            progress_callback=progress_callback,
        )

        best_rsa_arr, _, speaker_df, listener_df, _, _, _, consensuality_scores = \
            rsa_reranker.rerank(t=iterations)

        # --- Normalize uniqueness scores ---
        scores = self._normalize_uniqueness_scores(consensuality_scores)
        uniqueness = {s: float(v) for s, v in scores.items()}

        # --- Listener distribution L_t(d|s): exponentiate log-probs, normalize per column ---
        # listener_df shape: (N_reviews, K_sentences), values are log-probs
        listener_probs = np.exp(listener_df.values)  # (N, K)
        # Normalize columns so they sum to 1 per sentence
        col_sums = listener_probs.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums > 0, col_sums, 1.0)
        listener_probs = listener_probs / col_sums
        review_labels = [f"R{i+1}" for i in range(len(texts))]
        listener = {
            sent: {review_labels[i]: float(listener_probs[i, j]) for i in range(len(texts))}
            for j, sent in enumerate(listener_df.columns)
        }

        # --- Speaker distribution S_t(s|d): exponentiate log-probs, normalize per row ---
        # speaker_df shape: (N_reviews, K_sentences), values are log-probs
        speaker_probs = np.exp(speaker_df.values)  # (N, K)
        # Normalize rows so they sum to 1 per review
        row_sums = speaker_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        speaker_probs = speaker_probs / row_sums
        speaker = {
            review_labels[i]: {sent: float(speaker_probs[i, j]) for j, sent in enumerate(speaker_df.columns)}
            for i in range(len(texts))
        }

        # --- best_rsa: most characteristic sentence per review ---
        best_rsa = {review_labels[i]: str(best_rsa_arr[i]) for i in range(len(best_rsa_arr))}

        return {
            "uniqueness": uniqueness,
            "listener": listener,
            "speaker": speaker,
            "best_rsa": best_rsa,
        }


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

    def _get_field(content, *field_names):
        """Extract a field from v1 (plain str) or v2 ({"value": str}) content dicts."""
        for field in field_names:
            val = content.get(field, None) if isinstance(content, dict) else None
            if val is None:
                continue
            if isinstance(val, dict):
                val = val.get('value', '')
            if val and isinstance(val, str):
                return val
        return None

    def _get_invitations(note):
        """Return all invitation strings for a note (v1: str, v2: list)."""
        inv = getattr(note, 'invitation', None) or ''
        invs = getattr(note, 'invitations', None) or []
        result = []
        if isinstance(inv, str) and inv:
            result.append(inv)
        if isinstance(invs, list):
            result.extend(invs)
        return result

    def _fetch_with_client(client, forum_id):
        """Fetch and parse notes using a given openreview client. Returns (reviews, title, rebuttal_json)."""
        try:
            # v2 clients have get_all_notes; v1 has get_notes
            if hasattr(client, 'get_all_notes'):
                forum_notes = list(client.get_all_notes(forum=forum_id))
            else:
                forum_notes = client.get_notes(forum=forum_id)
        except Exception as api_error:
            print(f"[FETCH]   API call failed: {type(api_error).__name__}: {str(api_error)}")
            return None

        if not forum_notes:
            print(f"[FETCH]   No notes returned")
            return None

        print(f"[FETCH]   Got {len(forum_notes)} notes")

        # Extract title from submission note
        title = ""
        submission_patterns = ['Blind_Submission', '/Submission', 'submission', 'paper']
        for note in forum_notes:
            invitations = _get_invitations(note)
            inv_lower = [inv.lower() for inv in invitations]
            if any(p.lower() in inv for inv in inv_lower for p in submission_patterns):
                content = getattr(note, 'content', {})
                t = _get_field(content, 'title')
                if t:
                    title = t
                    print(f"[FETCH]   Title: {title[:80]}")
                break
        if not title:
            # Fallback: find the note whose forum == id (the root submission note)
            all_invitations = []
            for note in forum_notes:
                all_invitations.extend(_get_invitations(note))
                note_id = getattr(note, 'id', None)
                note_forum = getattr(note, 'forum', None)
                if note_id and note_forum and note_id == note_forum:
                    content = getattr(note, 'content', {})
                    t = _get_field(content, 'title')
                    if t:
                        title = t
                        print(f"[FETCH]   Title (root note): {title[:80]}")
                        break
            if not title:
                print(f"[FETCH]   No title found. Invitations seen: {all_invitations[:10]}")

        # Extract reviews
        reviews = []
        review_id_to_num = {}
        for note in forum_notes:
            invitations = _get_invitations(note)
            if any(p in inv for inv in invitations for p in ['Official_Review', 'Review', 'review']):
                content = getattr(note, 'content', {})
                text = _get_field(content, 'review', 'Review', 'text', 'content')
                if text and text.strip():
                    reviews.append(text.strip())
                    review_id_to_num[note.id] = len(reviews)
                    print(f"[FETCH]   Review {len(reviews)} found ({len(text)} chars)")

        if not reviews:
            print(f"[FETCH]   No reviews found — invitations present:")
            seen = {}
            for note in forum_notes:
                for inv in _get_invitations(note):
                    seen[inv] = seen.get(inv, 0) + 1
            for inv, count in seen.items():
                print(f"[FETCH]     {inv}: {count}")
            return None

        # Extract rebuttals
        rebuttals_structured = []
        for note in forum_notes:
            invitations = _get_invitations(note)
            if any(p in inv for inv in invitations for p in ['Official_Comment', 'Author_Comment', 'Comment']):
                sigs = getattr(note, 'signatures', []) or []
                if any('Authors' in sig for sig in sigs):
                    content = getattr(note, 'content', {})
                    text = _get_field(content, 'comment', 'rebuttal', 'title')
                    if text and text.strip():
                        replyto = getattr(note, 'replyto', None)
                        reply_num = review_id_to_num.get(replyto, None)
                        rebuttals_structured.append({"text": text.strip(), "reply_to": reply_num})

        rebuttal_text = json.dumps(rebuttals_structured) if rebuttals_structured else ""
        print(f"[FETCH]   {len(rebuttals_structured)} rebuttal(s) found")
        return reviews, title, rebuttal_text

    _browser_headers = {
        'Origin': 'https://openreview.net',
        'Referer': 'https://openreview.net/',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    }

    def _make_clients():
        """Yield (label, client) pairs to try, with browser headers injected."""
        # Clear stale env vars so openreview.Client doesn't auto-read them and try (bad) credentials
        _saved = {k: os.environ.pop(k) for k in ('OPENREVIEW_USERNAME', 'OPENREVIEW_PASSWORD') if k in os.environ}
        try:
            for label, baseurl, cls in [
                ("v2 guest", 'https://api2.openreview.net', openreview.api.OpenReviewClient),
                ("v1 guest", 'https://api.openreview.net', openreview.Client),
            ]:
                try:
                    client = cls(baseurl=baseurl)
                    client.headers.update(_browser_headers)
                    yield label, client
                except Exception as e:
                    print(f"[FETCH] {label} client init failed: {e}")
        finally:
            os.environ.update(_saved)

    result = None
    for label, client in _make_clients():
        print(f"[FETCH] Trying {label}...")
        result = _fetch_with_client(client, forum_id)
        if result:
            print(f"[FETCH] {label} succeeded")
            break

    if result is None:
        raise ValueError(
            f"Could not fetch reviews for '{forum_id}'. "
            f"This paper's reviews may require an OpenReview account. "
            f"Set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD in your environment and restart the app."
        )

    reviews, title, rebuttal_text = result
    print(f"[FETCH] SUCCESS: {len(reviews)} reviews, title: {title[:50]}")
    return reviews, title, rebuttal_text

