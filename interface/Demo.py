import sys, os.path
import threading
import time
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from collections import defaultdict
import hashlib
import json
import math
import re as _re
import html as _html
import torch
import numpy as np
import gradio as gr
import pandas as pd
import ast
from tqdm import tqdm

# ZeroGPU support for HuggingFace Spaces
try:
    import spaces
    _gpu = spaces.GPU
except ImportError:
    _gpu = lambda f: f  # no-op when not on HF Spaces

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

BASE_DIR = Path(__file__).resolve().parent.parent

# Controls how aggressively agreement colors are amplified.
# Lower = more vivid colors (0.2 = very strong, 1.0 = no amplification).
# Asymmetric: unique/red (positive) is amplified less than common/blue (negative)
# to avoid overwhelming red when most sentences are unique.
AGREEMENT_AMP_UNIQUE = 1.0  # exponent for positive scores (red = unique)
AGREEMENT_AMP_COMMON = 1.0  # exponent for negative scores (blue = common)
MAX_PREPROCESSED_REVIEWS = 10  # Number of review/agreement/rebuttal slots in pre-processed tab
LISTENER_CONCENTRATION_THRESHOLD = 0.70  # Above this, listener "wins" over uniqueness score
INFORMATIVENESS_MULTIPLIER = 2.0  # Multiplied by uniform baseline (1/K) for informativeness threshold


def _make_sentence_id(sentence: str) -> str:
    """Deterministic DOM ID for a sentence, used by click-to-scroll."""
    return "sent_" + hashlib.md5(sentence.encode("utf-8")).hexdigest()[:12]


def _click_to_scroll_js(sent_id: str, color: str = "#3b82f6") -> str:
    """Return inline onclick JS for smooth-scroll + outline flash."""
    return (
        f"(function(){{var el=document.getElementById('{sent_id}');"
        f"if(el){{el.scrollIntoView({{behavior:'smooth',block:'center'}});"
        f"el.style.outline='3px solid {color}';"
        f"setTimeout(function(){{el.style.outline='';}},2500);}}}})();"
    )


def _source_badges_html(sent: str, sentence_lists: list) -> str:
    """Return R# badge HTML for all reviews containing the sentence."""
    source = [r_idx + 1 for r_idx, sl in enumerate(sentence_lists) if sent in sl]
    return " ".join(
        f'<span style="background:#f3f4f6;color:#374151;padding:2px 6px;'
        f'border-radius:4px;font-size:0.72em;font-weight:600;">R{n}</span>'
        for n in source
    )


def _listener_dist_bars(sent: str, listener: dict, source_badges: str,
                        badge_fg: str = "#1e40af") -> str:
    """Render L_t(d|s) distribution bars or plain badge row."""
    if listener and sent in listener:
        dist = listener[sent]
        bar_parts = []
        for label, prob in sorted(dist.items()):
            pct = int(round(prob * 100))
            bar_w = max(2, int(prob * 80))
            bar_parts.append(
                f'<span style="display:inline-flex;align-items:center;gap:2px;margin-right:6px;">'
                f'<span style="font-size:0.7em;font-weight:600;color:{badge_fg};">{label}</span>'
                f'<span style="display:inline-block;width:{bar_w}px;height:5px;'
                f'background:#3b82f6;border-radius:3px;"></span>'
                f'<span style="font-size:0.7em;color:#6b7280;">{pct}%</span>'
                f'</span>'
            )
        return (
            f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:3px;margin-bottom:3px;">'
            f'{source_badges}'
            f'<span style="color:#d1d5db;font-size:0.75em;">\u2192</span> '
            + "".join(bar_parts) + "</div>"
        )
    return f'<div style="display:flex;gap:4px;margin-bottom:3px;">{source_badges}</div>'


def _get_context(sentence: str, sentence_lists: list):
    """Return (context_before, context_after) strings for the first review containing sentence."""
    for sl in sentence_lists:
        if sentence in sl:
            idx = sl.index(sentence)
            before = _html.escape(sl[idx - 1]) if idx > 0 else ""
            after = _html.escape(sl[idx + 1]) if idx < len(sl) - 1 else ""
            return before, after
    return "", ""


_TOGGLE_BTN_STYLE = (
    'background:none;border:1px solid #d1d5db;border-radius:6px;padding:4px 12px;'
    'font-size:0.78em;color:#6b7280;cursor:pointer;white-space:nowrap;'
    'line-height:1;height:28px;box-sizing:border-box;vertical-align:middle;'
    'display:inline-flex;align-items:center;justify-content:center;flex-shrink:0;'
)

def _toggle_html(selector: str, text_when_all_open: str,
                 text_when_not_all_open: str, initial_label: str) -> str:
    """Generate a toggle button for expanding/collapsing details elements."""
    return (
        '<button onclick="'
        f"let tab=this.closest('.tabitem')||this.closest('.gradio-container');"
        f"let details=tab.querySelectorAll('{selector}');"
        "if(!details.length)return;"
        "let allOpen=Array.from(details).every(d=>d.open);"
        "details.forEach(d=>d.open=!allOpen);"
        f"this.textContent=allOpen?'{text_when_all_open}':'{text_when_not_all_open}';"
        f'" style="{_TOGGLE_BTN_STYLE}"'
        f'>{initial_label}</button>'
    )


def _rebuttal_toggle_html() -> str:
    """Generate an Expand/Collapse All Responses toggle button with inline JS."""
    return _toggle_html("details:not(.review-collapse)",
                        "Expand All Responses", "Collapse All Responses",
                        "Expand All Responses")


def _review_toggle_html() -> str:
    """Generate a Collapse/Expand All Reviews toggle button with inline JS."""
    return _toggle_html("details.review-collapse",
                        "Expand All Reviews", "Collapse All Reviews",
                        "Collapse All Reviews")


def _jump_buttons_html(active_count: int, prefix: str = "int") -> str:
    """Generate jump-to buttons [R1] [R2] ... that scroll to each review.
    prefix: 'int' for interactive tab, 'pre' for pre-processed tab."""
    buttons = []
    for i in range(1, active_count + 1):
        anchor_id = f"{prefix}-review-anchor-{i}"
        js = (
            f"(function(){{var el=document.getElementById('{anchor_id}');"
            f"if(el)el.scrollIntoView({{behavior:'smooth',block:'start'}});"
            f"}})()"
        )
        buttons.append(
            f'<button onclick="{js}" '
            f'style="{_TOGGLE_BTN_STYLE}font-weight:600;">'
            f'R{i}</button>'
        )
    return "".join(buttons)



def _should_break_before(sent: str) -> bool:
    """Detect if a paragraph break should be inserted before this sentence."""
    s = sent.strip()
    # Numbered items: 1), 2., (1), 1:, etc.
    if _re.match(r'^[\(\[]?\d+[\)\]\.:]', s):
        return True
    # Dash/bullet items
    if len(s) > 2 and s[0] in ('-', '•', '*', '–', '—') and s[1] == ' ':
        return True
    # Markdown separators / headers
    if s.startswith('##') or s.startswith('---'):
        return True
    # Common review section headers
    if _re.match(
        r'^\*{0,2}(Rating|Strengths?|Weaknesses?|Questions?|Limitations?|Summary|'
        r'Soundness|Presentation|Contribution|Confidence|Experience|Review Assessment|'
        r'Recommendation|Overall|Minor|Major|Typos?|Suggestions?|Comments?|'
        r'Detailed\s+Comments?|Pros?|Cons?|Flag|Clarity|Significance|Originality)',
        s, _re.IGNORECASE,
    ):
        return True
    return False


def _is_review_header(sent: str) -> bool:
    """Detect if a sentence is a review metadata header (Rating:, Experience:, etc.)."""
    return bool(_re.match(
        r'^\*{0,2}(Rating|Confidence|Experience|Review Assessment|Recommendation|Flag)\b',
        sent.strip(), _re.IGNORECASE,
    ))


# ---- Polarity / Topic color maps for HTML rendering ----
_POLARITY_COLORS = {
    2: "#d4fcd6", 0: "#fcd6d6",        # integer keys (pre-processed tab)
    "➕": "#d4fcd6", "➖": "#fcd6d6",   # emoji keys (interactive tab)
}  # positive=green, negative=red
_TOPIC_HTML_COLORS = {
    "Substance": "#b3e5fc",
    "Clarity": "#c8e6c9",
    "Soundness/Correctness": "#fff9c4",
    "Originality": "#f8bbd0",
    "Motivation/Impact": "#d1c4e9",
    "Meaningful Comparison": "#ffe0b2",
    "Replicability": "#b2dfdb",
}


def _wrap_review_card(label: str, inner_html: str, collapsible: bool = True) -> str:
    """Wrap review content in a styled card with gray header. Single source of truth for review card styling."""
    escaped = _html.escape(label) if label else ""
    if collapsible:
        return (
            f'<details open class="review-collapse" style="border:1px solid #d1d5db;'
            f'border-radius:8px;padding:0;margin-bottom:10px;overflow:hidden;">'
            f'<summary style="font-weight:600;font-size:0.9em;color:#374151;cursor:pointer;'
            f'list-style:none;display:flex;align-items:center;gap:6px;'
            f'background:#f9fafb;padding:8px 14px;border-bottom:1px solid #e5e7eb;">'
            f'<span style="transition:transform 0.2s;font-size:0.7em;">▶</span> '
            f'{escaped}</summary>'
            f'<div style="padding:12px 16px;">{inner_html}</div>'
            f'</details>'
        )
    else:
        if not label:
            return inner_html
        return (
            f'<div style="border:1px solid #d1d5db;border-radius:8px;padding:0;margin-bottom:10px;overflow:hidden;">'
            f'<div style="background:#f9fafb;padding:8px 14px;border-bottom:1px solid #e5e7eb;'
            f'font-weight:600;font-size:0.85em;color:#374151;">{escaped}</div>'
            f'<div style="padding:12px 16px;">{inner_html}</div>'
            f'</div>'
        )


def render_review_html(
    review_items: list,
    mode: str = "plain",
    label: str = "Review",
    wrap: bool = False,
) -> str:
    """
    Render a review as HTML with proper paragraph formatting.

    Args:
        review_items: list of (sentence, metadata_dict) tuples
        mode: "plain", "polarity", or "topic"
        label: header label
        wrap: if False, return bare content (caller handles outer wrapper)
    """
    if not review_items:
        return ""

    parts = []
    parts.append('<div style="line-height:1.8;font-size:0.95em;margin-top:6px;">')

    for i, (sent, metadata) in enumerate(review_items):
        # Paragraph break detection
        if i > 0 and _should_break_before(sent):
            parts.append('<br>')

        # Header styling (Rating:, Experience:, etc.)
        is_header = _is_review_header(sent)

        bg = ""
        label_text = ""
        if mode == "polarity":
            polarity = metadata.get("polarity")
            if polarity in _POLARITY_COLORS:
                bg = f"background:{_POLARITY_COLORS[polarity]};"
        elif mode == "topic":
            topic = metadata.get("topic")
            if topic and topic != "NONE" and topic in _TOPIC_HTML_COLORS:
                bg = f"background:{_TOPIC_HTML_COLORS[topic]};"
                label_text = topic

        style = f"padding:1px 3px;border-radius:3px;{bg}"
        if is_header:
            style += "font-weight:600;color:#92400e;"

        sent_id = _make_sentence_id(sent)
        escaped = _html.escape(sent)

        if label_text:
            # Show topic label as a small tag
            parts.append(
                f'<span id="{sent_id}" style="{style}" title="{_html.escape(label_text)}">'
                f'{escaped} </span>'
            )
        else:
            parts.append(f'<span id="{sent_id}" style="{style}">{escaped} </span>')

    parts.append('</div>')
    content = "".join(parts)
    if wrap:
        return _wrap_review_card(label, content, collapsible=True)
    elif label:
        return _wrap_review_card(label, content, collapsible=False)
    return content


def format_summary_cards(
    sentences: list,
    scores: dict,
    sentence_lists: list,
    card_type: str = "common",
    listener: dict = None,
    speaker: dict = None,
) -> str:
    """
    Most Common Opinions hub.

    Selection: When listener/speaker data is available, re-ranks candidates by
    informativeness × (1 − normalized_uniqueness) so substantive agreements
    surface above generic filler. Falls back to the raw score order when the
    full RSA data is not available (pre-processed tab).

    Each card shows:
    - L_t(d|s) distribution bars (R1 40% · R2 45% · R3 15%) when data is available
    - Context snippet (1 sentence before / after)
    - Clickable to scroll to the sentence in the full review below
    """
    if not sentences:
        return ""

    border_color = "#93c5fd"
    badge_bg = "#dbeafe"
    badge_fg = "#1e40af"

    # Pre-compute expected listener share per reviewer from review lengths.
    # Used for bar chart normalization (bar width = deviation from expected, not raw prob).
    num_reviews = len(sentence_lists)
    total_sents = sum(len(sl) for sl in sentence_lists) or 1
    expected_share = {
        f"R{i+1}": len(sl) / total_sents
        for i, sl in enumerate(sentence_lists)
    }

    # Render in the order given — selection and filtering happens in compute_rsa_in_background.
    ctx_style = "color:#b0b0b0;font-size:0.85em;font-style:italic;"
    cards_parts = []

    for sent in sentences:
        sent_id = _make_sentence_id(sent)
        context_before, context_after = _get_context(sent, sentence_lists)

        # --- Source badge: which review(s) this sentence physically appears in ---
        source_badge = _source_badges_html(sent, sentence_lists)

        # --- L_t(d|s) distribution bars ---
        dist_html = _listener_dist_bars(sent, listener, source_badge, badge_fg=badge_fg)

        onclick = _click_to_scroll_js(sent_id)

        # Inline context: ...before SENTENCE after...  (all one line)
        before_span = f'<span style="{ctx_style}">...{_html.escape(context_before)} </span>' if context_before else ""
        after_span = f'<span style="{ctx_style}"> {_html.escape(context_after)}...</span>' if context_after else ""

        cards_parts.append(
            f'<div style="border:1px solid #e5e7eb;border-left:3px solid {border_color};'
            f'border-radius:6px;padding:8px 12px;margin-bottom:5px;cursor:pointer;" '
            f'onclick="{_html.escape(onclick)}">'
            f'{dist_html}'
            f'<div style="color:#111827;line-height:1.5;">'
            f'{before_span}'
            f'<span style="font-weight:500;">{_html.escape(sent)}</span>'
            f'{after_span}'
            f'</div>'
            f'</div>'
        )

    # Wrap in collapsible <details> (open by default)
    inner = "".join(cards_parts)
    return (
        f'<details open style="margin-bottom:8px;">'
        f'<summary style="cursor:pointer;font-weight:600;font-size:0.9em;color:#374151;'
        f'margin-bottom:6px;list-style:none;">'
        f'<span style="margin-right:4px;">▸</span>Most Common Opinions</summary>'
        f'{inner}'
        f'</details>'
    )


def _normalize_polarity(val) -> Optional[str]:
    """Normalize polarity from any format to 'positive'/'negative'/None."""
    if val == "➕" or val == 2 or val == "positive":
        return "positive"
    if val == "➖" or val == 0 or val == "negative":
        return "negative"
    return None  # neutral or unknown


def format_common_themes(
    sentence_lists: list,
    polarity_map: dict,
    topic_map: dict,
    speaker: dict = None,
    uniqueness: dict = None,
    listener: dict = None,
) -> str:
    """
    Common Themes Across Reviews — groups sentences by topic, then shows
    polarity breakdown within each topic.

    A topic is "common" if sentences from ≥2 different reviewers share it.
    Each topic card shows a polarity percentage bar and representative
    sentences per reviewer under each polarity sub-group.

    Falls back to RSA-based generic sentences if no common themes are found.
    """
    num_reviews = len(sentence_lists)
    if num_reviews < 2:
        return ""

    # --- Step 1: Build per-sentence (topic, polarity) index ---
    # topic_data[topic][polarity][r_idx] = [sentences]
    topic_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for r_idx, sl in enumerate(sentence_lists):
        for sent in sl:
            if _should_break_before(sent) or _is_review_header(sent):
                continue
            if len(sent.split()) < 5:
                continue
            topic = topic_map.get(sent)
            polarity = _normalize_polarity(polarity_map.get(sent))
            if topic is None and polarity is None:
                continue
            topic_key = topic if topic else "Other"
            polarity_key = polarity if polarity else "neutral"
            topic_data[topic_key][polarity_key][r_idx].append(sent)

    # --- Step 2: Filter to topics with ≥2 unique reviewers ---
    common_topics = []
    for topic, pol_dict in topic_data.items():
        all_reviewers = set()
        total_sents = 0
        for pol, rev_dict in pol_dict.items():
            all_reviewers.update(rev_dict.keys())
            total_sents += sum(len(s) for s in rev_dict.values())
        if len(all_reviewers) < 2:
            continue
        common_topics.append((topic, len(all_reviewers), total_sents, pol_dict))

    # Prefer topics that have non-neutral polarity entries
    has_sentiment = [t for t in common_topics
                     if any(p in t[3] for p in ("positive", "negative"))]
    if len(has_sentiment) >= 1:
        common_topics = has_sentiment

    # Rank by reviewer count (desc), then sentence count (desc); push "Other" to bottom
    common_topics.sort(key=lambda t: (0 if t[0] == "Other" else 1, t[1], t[2]), reverse=True)

    # --- Fallback: Generic Sentences ---
    if not common_topics:
        if not uniqueness:
            return ""
        scores_series = pd.Series(uniqueness)
        n_seed = min(5, len(scores_series))
        fallback = scores_series.nsmallest(n_seed).index.tolist()
        if not fallback:
            return ""

        parts = [
            '<details open style="margin-bottom:10px;border:1px solid #d1d5db;border-radius:8px;'
            'padding:0;overflow:hidden;">',
            '<summary style="cursor:pointer;font-weight:700;font-size:0.92em;color:#1f2937;'
            'padding:10px 14px;list-style:none;background:#f9fafb;border-bottom:1px solid #e5e7eb;'
            'user-select:none;display:flex;align-items:center;gap:6px;">'
            '<span class="collapse-arrow" style="display:inline-block;transition:transform 0.2s;'
            'font-size:0.75em;">&#9660;</span>'
            'Generic Sentences</summary>',
            '<div style="padding:8px 10px;">',
            '<div style="background:#fefce8;border:1px solid #fde68a;border-radius:6px;'
            'padding:8px 12px;margin-bottom:8px;font-size:0.82em;color:#92400e;">'
            'No shared themes detected across reviewers. Showing sentences with lowest '
            'RSA uniqueness score (most generic across reviews).</div>',
        ]
        for sent in fallback:
            sent_id = _make_sentence_id(sent)
            badges = _source_badges_html(sent, sentence_lists)
            dist_html = _listener_dist_bars(sent, listener, badges)

            onclick = _click_to_scroll_js(sent_id)
            parts.append(
                f'<div style="border:1px solid #e5e7eb;border-left:3px solid #d1d5db;'
                f'border-radius:6px;padding:6px 10px;margin-bottom:4px;cursor:pointer;" '
                f'onclick="{_html.escape(onclick)}">'
                f'{dist_html}'
                f'<span style="color:#111827;">{_html.escape(sent)}</span>'
                f'</div>'
            )
        parts.append('</div></details>')
        return "".join(parts)

    # --- Step 3: Render topic cards with polarity breakdown ---
    _pol_colors = {"negative": "#ef4444", "positive": "#22c55e", "neutral": "#9ca3af"}
    _pol_labels = {"negative": "Negative", "positive": "Positive", "neutral": "Neutral"}
    # Render order: negative first (concerns), then positive, then neutral
    _pol_order = ["negative", "positive", "neutral"]

    def _pick_best(sents, r_idx):
        """Pick best sentence by speaker score for this reviewer."""
        r_label = f"R{r_idx + 1}"
        if speaker and r_label in speaker:
            sp = speaker[r_label]
            scored = [(s, sp.get(s, 0.0)) for s in sents]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[0][0]
        return sents[0]

    def _sent_row(sent, r_idx):
        """Render a single sentence row with R# badge and click-to-scroll."""
        r_label = f"R{r_idx + 1}"
        sent_id = _make_sentence_id(sent)
        onclick = _click_to_scroll_js(sent_id)
        return (
            f'<div style="display:flex;align-items:baseline;gap:6px;padding:2px 0;'
            f'padding-left:8px;cursor:pointer;" '
            f'onclick="{_html.escape(onclick)}">'
            f'<span style="background:#f3f4f6;color:#374151;padding:1px 5px;'
            f'border-radius:3px;font-size:0.7em;font-weight:600;flex-shrink:0;">{r_label}</span>'
            f'<span style="color:#374151;font-size:0.85em;line-height:1.4;">{_html.escape(sent)}</span>'
            f'</div>'
        )

    cards = []
    for topic, n_reviewers, total_sents, pol_dict in common_topics:
        border_color = _TOPIC_HTML_COLORS.get(topic, "#d1d5db")
        reviewer_text = (
            f"All {num_reviews} reviewers" if n_reviewers == num_reviews
            else f"{n_reviewers} of {num_reviews} reviewers"
        )

        # --- Polarity percentage bar ---
        pol_counts = {}
        for pol in _pol_order:
            if pol in pol_dict:
                pol_counts[pol] = sum(len(s) for s in pol_dict[pol].values())
        total = sum(pol_counts.values()) or 1

        # Inline polarity bar + labels (compact, on one line)
        _pol_colors_soft = {"negative": "#f87171", "positive": "#4ade80", "neutral": "#d1d5db"}
        bar_segments = []
        bar_labels = []
        for pol in _pol_order:
            cnt = pol_counts.get(pol, 0)
            if cnt == 0:
                continue
            pct = round(cnt / total * 100)
            color = _pol_colors_soft[pol]
            bar_segments.append(
                f'<span style="display:inline-block;height:6px;width:{max(pct, 4)}%;'
                f'background:{color};opacity:0.75;"></span>'
            )
            bar_labels.append(
                f'<span style="font-size:0.7em;color:#6b7280;">'
                f'{pct}% {_pol_labels[pol]}</span>'
            )

        # Skip polarity bar for "Other" — unclassified sentences aren't necessarily related
        if topic == "Other":
            polarity_bar = ""
        else:
            polarity_bar = (
                f'<div style="display:flex;align-items:center;gap:6px;margin-left:8px;">'
                f'<div style="display:flex;width:80px;height:6px;border-radius:3px;overflow:hidden;'
                f'background:#f3f4f6;flex-shrink:0;">'
                + "".join(bar_segments)
                + '</div>'
                + " ".join(bar_labels)
                + '</div>'
            )

        # --- Header (polarity bar inline after reviewer count) ---
        header = (
            f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">'
            f'<span style="font-weight:600;font-size:0.88em;color:#1f2937;">{_html.escape(topic)}</span>'
            f'<span style="color:#9ca3af;font-size:0.75em;">·</span>'
            f'<span style="font-size:0.75em;color:#6b7280;">{reviewer_text}</span>'
            f'{polarity_bar}'
            f'</div>'
        )

        # --- Polarity sub-groups ---
        sub_groups_html = []
        for pol in _pol_order:
            if pol not in pol_dict:
                continue
            rev_dict = pol_dict[pol]
            cnt = pol_counts.get(pol, 0)
            color = _pol_colors[pol]

            sub_header = (
                f'<div style="font-size:0.78em;font-weight:600;color:{color};'
                f'margin:4px 0 2px 0;">'
                f'{_pol_labels[pol]} ({cnt} sentence{"s" if cnt != 1 else ""})</div>'
            )

            rows = []
            for r_idx in sorted(rev_dict.keys()):
                best = _pick_best(rev_dict[r_idx], r_idx)
                rows.append(_sent_row(best, r_idx))

            sub_groups_html.append(sub_header + "".join(rows))

        cards.append(
            f'<div style="border:1px solid #e5e7eb;border-left:3px solid {border_color};'
            f'border-radius:6px;padding:8px 12px;margin-bottom:5px;">'
            f'{header}{"".join(sub_groups_html)}'
            f'</div>'
        )

    inner = "".join(cards)
    return (
        f'<details open style="margin-bottom:10px;border:1px solid #d1d5db;border-radius:8px;'
        f'padding:0;overflow:hidden;">'
        f'<summary style="cursor:pointer;font-weight:700;font-size:0.92em;color:#1f2937;'
        f'padding:10px 14px;list-style:none;background:#f9fafb;border-bottom:1px solid #e5e7eb;'
        f'user-select:none;display:flex;align-items:center;gap:6px;">'
        f'<span class="collapse-arrow" style="display:inline-block;transition:transform 0.2s;'
        f'font-size:0.75em;">&#9654;</span>'
        f'Common Themes Across Reviews</summary>'
        f'<div style="padding:8px 10px;">{inner}</div>'
        f'</details>'
    )


def format_divergent_cards(
    uniqueness: dict,
    sentence_lists: list,
    listener: dict,
    speaker: dict,
) -> Dict[int, str]:
    """
    Most Divergent Opinions — returns per-review HTML dict {review_index: html}.

    For each review, finds the sentences where argmax(L_t(d|s)) points to that
    review (i.e., the listener assigns it most strongly to that reviewer) AND
    the uniqueness score is above the median. Ranks within each reviewer's set
    by their Speaker score S_t(s|d) (how characteristic of that reviewer).
    Shows the top 2 per reviewer.
    """
    if not uniqueness or not listener or not speaker:
        return {}

    num_reviews = len(speaker)
    if num_reviews == 0:
        return {}

    median_u = float(np.median(list(uniqueness.values())))
    review_labels = [f"R{i+1}" for i in range(num_reviews)]

    # Minimum speaker score to suppress generic filler.
    k = max(sum(len(v) for v in speaker.values()) // max(len(speaker), 1), 1)
    min_speaker_score = INFORMATIVENESS_MULTIPLIER / k

    # Group sentences by their argmax review
    grouped: dict = {label: [] for label in review_labels}
    for sent, u_score in uniqueness.items():
        if u_score <= median_u:
            continue
        if sent not in listener:
            continue
        dist = listener[sent]
        if not dist:
            continue
        argmax_label = max(dist, key=lambda k: dist[k])
        if argmax_label not in grouped:
            continue
        s_score = speaker.get(argmax_label, {}).get(sent, 0.0)
        if s_score < min_speaker_score:
            continue
        grouped[argmax_label].append((sent, u_score, s_score))

    # Sort each group by speaker score descending and take top 2
    for label in review_labels:
        grouped[label].sort(key=lambda x: x[2], reverse=True)
        grouped[label] = grouped[label][:2]

    border_color = "#fca5a5"
    result: Dict[int, str] = {}

    for i, label in enumerate(review_labels):
        items = grouped[label]
        if not items:
            continue

        ctx_style = "color:#b0b0b0;font-size:0.85em;font-style:italic;"
        html_parts = [
            '<div style="margin-top:10px;margin-bottom:6px;font-weight:600;font-size:0.82em;color:#7f1d1d;">'
            f'Unique Points in This Review</div>'
        ]
        for sent, u_score, s_score in items:
            sent_id = _make_sentence_id(sent)
            context_before, context_after = _get_context(sent, sentence_lists)

            dom_pct = 0
            if sent in listener:
                dom_pct = int(round(max(listener[sent].values(), default=0.0) * 100))
            uniqueness_badge = (
                f'<span style="background:#fee2e2;color:#991b1b;padding:2px 6px;'
                f'border-radius:4px;font-size:0.7em;font-weight:600;display:inline-block;margin-bottom:3px;">'
                f'{dom_pct}% listener share</span>'
            )

            onclick = _click_to_scroll_js(sent_id, "#ef4444")

            before_span = f'<span style="{ctx_style}">...{_html.escape(context_before)} </span>' if context_before else ""
            after_span = f'<span style="{ctx_style}"> {_html.escape(context_after)}...</span>' if context_after else ""

            html_parts.append(
                f'<div style="border:1px solid #e5e7eb;border-left:3px solid {border_color};'
                f'border-radius:6px;padding:8px 12px;margin-bottom:5px;cursor:pointer;" '
                f'onclick="{_html.escape(onclick)}">'
                f'{uniqueness_badge}'
                f'<div style="color:#111827;line-height:1.5;">'
                f'{before_span}'
                f'<span style="font-weight:500;">{_html.escape(sent)}</span>'
                f'{after_span}'
                f'</div>'
                f'</div>'
            )

        result[i] = "".join(html_parts)

    return result


def render_agreement_html(
    sentences: List[str],
    uniqueness: Dict[str, float],
    listener: Dict[str, Dict[str, float]],
    speaker: Dict[str, Dict[str, float]],
    num_reviews: int,
    label: str = "Agreement",
    wrap: bool = False,
) -> str:
    """
    Custom HTML renderer for Agreement mode (replaces gr.HighlightedText).

    Each sentence gets:
    - Continuous opacity from score magnitude (strongest opinions most vivid)
    - CSS hover tooltip showing L_t(d|s): "R1 (40%) · R2 (45%)"
    - Informativeness dimming for generic common filler
    - Sentence ID for click-to-scroll from summary cards
    """
    if not sentences:
        return ""

    # Color scale legend bar
    legend_html = (
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;'
        'font-size:0.75em;color:#6b7280;">'
        '<span style="background:linear-gradient(to right,rgba(59,130,246,0.7),rgba(209,213,219,0.3),rgba(239,68,68,0.7));'
        'width:120px;height:8px;border-radius:4px;display:inline-block;"></span>'
        '<span>← Common &nbsp;|&nbsp; Unique →</span>'
        '</div>'
    )

    parts = []
    parts.append(legend_html)
    parts.append('<div style="line-height:1.8;font-size:0.95em;margin-top:6px;">')

    # Compute informativeness threshold: 2 / K (twice uniform baseline)
    k = max(len(uniqueness), 1)
    info_threshold = INFORMATIVENESS_MULTIPLIER / k

    for idx, sent in enumerate(sentences):
        # Paragraph break detection
        if idx > 0 and _should_break_before(sent):
            parts.append('<br>')

        sent_id = _make_sentence_id(sent)
        score = uniqueness.get(sent)

        # Header styling (Rating:, Experience:, etc.)
        header_style = "font-weight:600;color:#92400e;" if _is_review_header(sent) else ""

        if score is None or abs(score) < HIGHLIGHT_THRESHOLD:
            # No highlight
            parts.append(f'<span id="{sent_id}" style="{header_style}">{_html.escape(sent)} </span>')
            continue

        # --- Color and opacity ---
        if score < 0:
            # Common: blue — opacity from listener ENTROPY so more balanced = more vivid.
            r, g, b = 59, 130, 246
            if listener and sent in listener:
                dist = listener[sent]
                max_prob = max(dist.values(), default=0.0)
                max_entropy = math.log(max(num_reviews, 2))
                entropy = sum(-p * math.log(p) for p in dist.values() if p > 0)

                # If listener is highly concentrated on one reviewer (>70%), the RSA
                # uniqueness score and listener disagree — trust the listener and
                # suppress blue. This prevents e.g. R2 91% sentences from appearing blue.
                if max_prob > LISTENER_CONCENTRATION_THRESHOLD:
                    opacity = 0.0
                else:
                    opacity = (entropy / max_entropy) ** AGREEMENT_AMP_COMMON if max_entropy > 0 else 0.0
            else:
                opacity = abs(score) ** AGREEMENT_AMP_COMMON

            # Informativeness dimming for generic filler
            if speaker:
                info = compute_informativeness(sent, speaker, num_reviews)
                if info < info_threshold:
                    opacity *= 0.3
        else:
            # Unique: red — opacity from score magnitude (as before)
            r, g, b = 239, 68, 68
            opacity = abs(score) ** AGREEMENT_AMP_UNIQUE

        bg_color = f"rgba({r},{g},{b},{opacity:.3f})"

        # --- Tooltip content from L_t(d|s) ---
        tooltip_text = ""
        if listener and sent in listener:
            dist = listener[sent]
            parts_tooltip = " · ".join(
                f"{lbl} {int(round(p * 100))}%"
                for lbl, p in sorted(dist.items())
            )
            tooltip_text = f"{parts_tooltip}"
        else:
            tooltip_text = f"Score: {score:+.2f}"

        # Inline JS positions the tooltip near the cursor using fixed positioning
        hover_js = (
            "var t=this.querySelector('.rsa-tooltip');var r=this.getBoundingClientRect();"
            "t.style.display='block';"
            "t.style.left=Math.min(r.left,window.innerWidth-290)+'px';"
            "t.style.top=(r.top-t.offsetHeight-6)+'px';"
        )
        leave_js = "this.querySelector('.rsa-tooltip').style.display='none';"
        parts.append(
            f'<span id="{sent_id}" class="rsa-sentence" style="background:{bg_color};" '
            f'onmouseenter="{hover_js}" onmouseleave="{leave_js}">'
            f'{_html.escape(sent)} '
            f'<span class="rsa-tooltip">{_html.escape(tooltip_text)}</span>'
            f'</span>'
        )

    parts.append("</div>")  # close sentence container
    content = "".join(parts)
    if wrap:
        return _wrap_review_card(label, content, collapsible=True)
    elif label:
        return _wrap_review_card(label, content, collapsible=False)
    return content


def build_review_card(
    label: str,
    *,
    review_items: list = None,
    mode: str = "plain",
    sentences: List[str] = None,
    uniqueness: Dict = None,
    listener: dict = None,
    speaker: dict = None,
    num_reviews: int = 0,
    divergent_html: str = "",
    rebuttal_html: str = "",
    collapsible: bool = True,
) -> str:
    """Unified review card builder — single entry point for both tabs.

    For plain/polarity/topic: pass review_items + mode.
    For agreement: pass sentences + RSA dicts (uniqueness, listener, speaker, num_reviews).
    Optional divergent_html and rebuttal_html are appended inside the card.
    """
    if sentences is not None:
        inner = render_agreement_html(
            sentences, uniqueness or {}, listener, speaker,
            num_reviews=num_reviews, label="",
        )
    elif review_items is not None:
        inner = render_review_html(review_items, mode=mode, label="")
    else:
        inner = ""
    return _wrap_review_card(label, f"{inner}{divergent_html}{rebuttal_html}", collapsible=collapsible)


# Auto-detect the preprocessed dataset CSV
def _find_preprocessed_csv() -> Path:
    """Find the most recent preprocessed_scored_reviews_*.csv in the data dir."""
    data_dir = BASE_DIR / "data"
    candidates = sorted(data_dir.glob("preprocessed_scored_reviews_*.csv"))
    if candidates:
        return candidates[-1]  # Last alphabetically = latest year range
    return data_dir / "preprocessed_scored_reviews.csv"


def load_scored_reviews_with_rebuttals(csv_path: Path = None):
    """Load dataset with rebuttal metadata. Auto-detects CSV if no path given."""
    if csv_path is None:
        csv_path = _find_preprocessed_csv()

    if not csv_path.exists():
        return [], pd.DataFrame()

    df = pd.read_csv(csv_path)
    tqdm.pandas(desc="Parsing scored_dict")
    df["scored_dict"] = df["scored_dict"].progress_apply(ast.literal_eval)

    # Parse metadata column
    if "metadata" in df.columns:
        df["metadata"] = df["metadata"].progress_apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and x != '{}' else {}
        )
    else:
        df["metadata"] = [{}] * len(df)

    years = df["year"].tolist()
    return years, df

years_new, df_new = load_scored_reviews_with_rebuttals()

if df_new.empty:
    raise FileNotFoundError(
        f"No preprocessed dataset found. Run the pipeline first (./pipeline/process_new_data.sh)."
    )

# Use new data only
years, all_scored_reviews_df = years_new, df_new

# Build a {forum_url: paper_title} lookup from raw data CSVs (processed CSVs lack paper_title)
def _load_paper_titles() -> dict:
    titles = {}
    for csv in sorted((BASE_DIR / "data").glob("all_reviews_*.csv")):
        try:
            df = pd.read_csv(csv, usecols=["id", "paper_title"])
            for _, row in df.iterrows():
                if row["id"] not in titles and pd.notna(row.get("paper_title", "")):
                    titles[row["id"]] = str(row["paper_title"])
        except Exception:
            pass
    return titles

_paper_titles = _load_paper_titles()
year_range_str = f"{min(years)}–{max(years)}" if years else "N/A"

# -----------------------------------
# Pre-processed Tab
# -----------------------------------

def get_preprocessed_scores(year):
    scored_reviews = all_scored_reviews_df[all_scored_reviews_df["year"] == year]["scored_dict"].iloc[0]
    return scored_reviews


def get_preprocessed_metadata(year):
    row = all_scored_reviews_df[all_scored_reviews_df["year"] == year]
    if "metadata" in row.columns and not row.empty:
        meta = row["metadata"].iloc[0]
        return meta if isinstance(meta, dict) else {}
    return {}


# -----------------------------------
# Interactive Tab Configuration
# -----------------------------------

# Define the manual color map for topics
topic_color_map = {
    "Substance": "#cce0ff",             # lighter blue
    "Clarity": "#e6ee9c",               # lighter yellow-green
    "Soundness/Correctness": "#ffcccc", # lighter red
    "Originality": "#d1c4e9",           # lighter purple
    "Motivation/Impact": "#b2ebf2",     # lighter teal
    "Meaningful Comparison": "#fff9c4", # lighter yellow
    "Replicability": "#c8e6c9",         # lighter green
}


# GLIMPSE Home/Description Page
glimpse_description = f"""
# ReView: A Tool for Visualizing and Analyzing Scientific Reviews
## **Overview**
ReView is a visualization tool designed to assist **area chairs** and **researchers** in efficiently analyzing scholarly reviews. The interface offers two main ways to explore scholarly reviews:
- Pre-Processed Reviews: Explore real peer reviews from ICLR ({year_range_str}) with structured visualizations of sentiment, topics, and reviewer agreement.
- Interactive Tab: Enter your own reviews and view them analyzed in real time using the same NLP-powered highlighting options.
All reviews are shown in their original, unaltered form, with visual overlays to help identify key insights such as disagreements, sentiment and common themes—reducing cognitive load and scrolling effort.
---
## **Key Features**
- *Traceability and Transparency:* The tool preserves the original text of each review and overlays highlights for key aspects (e.g., sentiment, topic, agreement), allowing area chairs to trace back every insight to its source without modifying or summarizing the content.
- *Structured Overview*: All reviews are displayed in one interface and with radio buttons, one can navigate from one highlighting option to the other.
- *Interactive*: The tool allows users to input their own reviews and, within seconds, view them annotated with highlighted aspects
---
## **Highlighting Options**
- *Agreement:* Identifies both shared and conflicting points across reviews, helping to surface consensus and disagreement.
- *Polarity:* Highlights positive and negative sentiments within the reviews to reveal tone and stance.
- *Topic:* Organizes the review sentences by their discussed topics, ensuring coverage of diverse reviewer perspectives and improving clarity.
---
### How to Use ReView
ReView offers two main ways to explore peer reviews: using pre-processed reviews or by entering your own.
#### Pre-Processed Reviews Tab
Use this tab to explore reviews from ICLR ({year_range_str}):
1. **Select a conference year** from the dropdown menu on the right.
2. **Navigate between submissions** using the *Next* and *Previous* buttons on the left.
3. **Choose a highlighting view** using the radio buttons:
   - **Original**: Displays unmodified review text.
   - **Agreement**: Highlights consensus points in **red** and disagreements in **purple**.
   - **Polarity**: Highlights **positive** sentiment in **green** and **negative** sentiment in **red**.
   - **Topic**: Highlights comments by discussion topic using color-coded labels.
#### Interactive Tab
Use this tab to analyze your own review text:
1. **Enter 2–6 reviews** in the input fields. Use the **➕ Add Review** button to add up to 6 reviews.
2. **Click "Process"** to analyze the input (average processing time: ~42 seconds).
3. **Explore the results** using the same highlighting options as above (Agreement, Polarity, Topic).
"""


EXAMPLES = [
    "The paper gives really interesting insights on the topic of transfer learning. It is well presented and the experiment are extensive. I believe the authors missed Jane and al 2021. In addition, I think, there is a mistake in the math.",
    "The paper gives really interesting insights on the topic of transfer learning. It is well presented and the experiment are extensive. Some parts remain really unclear and I would like to see a more detailed explanation of the proposed method.",
    "The paper gives really interesting insights on the topic of transfer learning. It is not well presented and lack experiments. Some parts remain really unclear and I would like to see a more detailed explanation of the proposed method.",
]

PROCESSING_TIMER_HTML = """
<div style="display:flex;align-items:center;gap:12px;padding:16px;background:#f0f4ff;border-radius:8px;border:1px solid #c7d2fe;margin:8px 0;">
  <div style="width:24px;height:24px;border:3px solid #e0e7ff;border-top:3px solid #4f46e5;border-radius:50%;animation:procspin 1s linear infinite;flex-shrink:0;"></div>
  <div>
    <div style="font-weight:600;color:#312e81;">Processing reviews...</div>
  </div>
</div>
<style>@keyframes procspin{to{transform:rotate(360deg);}}</style>
"""

FETCHING_HTML = """
<div style="display:flex;align-items:center;gap:12px;padding:16px;background:#f0f4ff;border-radius:8px;border:1px solid #c7d2fe;margin:8px 0;">
  <div style="width:24px;height:24px;border:3px solid #e0e7ff;border-top:3px solid #4f46e5;border-radius:50%;animation:procspin 1s linear infinite;flex-shrink:0;"></div>
  <div>
    <div style="font-weight:600;color:#312e81;">Fetching reviews from OpenReview...</div>
  </div>
</div>
<style>@keyframes procspin{to{transform:rotate(360deg);}}</style>
"""

POLARITY_PROGRESS_HTML = """
<div style="padding:10px 16px;background:#f0fff4;border-radius:8px;border:1px solid #bbf7d0;margin:0;">
  <div style="display:flex;align-items:center;gap:10px;">
    <div style="width:18px;height:18px;border:3px solid #dcfce7;border-top:3px solid #16a34a;border-radius:50%;animation:procspin 1s linear infinite;flex-shrink:0;"></div>
    <span style="font-weight:600;color:#14532d;font-size:0.9em;white-space:nowrap;">Computing polarity &amp; topic... </span>
    <div style="flex:1;background:#dcfce7;border-radius:4px;height:6px;overflow:hidden;">
      <div style="background:linear-gradient(90deg,#4ade80,#16a34a);height:100%;border-radius:4px;animation:agrslide 2s ease-in-out infinite;"></div>
    </div>
  </div>
</div>
"""

AGREEMENT_PROGRESS_HTML = """
<div style="padding:10px 16px;background:#f0f4ff;border-radius:8px;border:1px solid #c7d2fe;margin:0;">
  <div style="display:flex;align-items:center;gap:10px;">
    <div style="width:18px;height:18px;border:3px solid #e0e7ff;border-top:3px solid #4f46e5;border-radius:50%;animation:procspin 1s linear infinite;flex-shrink:0;"></div>
    <span style="font-weight:600;color:#312e81;font-size:0.9em;white-space:nowrap;">Computing agreement in background...</span>
    <div style="flex:1;background:#e0e7ff;border-radius:4px;height:6px;overflow:hidden;">
      <div style="background:linear-gradient(90deg,#818cf8,#4f46e5);height:100%;border-radius:4px;animation:agrslide 2s ease-in-out infinite;"></div>
    </div>
  </div>
</div>
"""


def _status_html(msg, kind="success"):
    """Generate styled HTML for status messages."""
    styles = {
        "success": ("✅", "#f0fdf4", "#16a34a", "#166534", "#bbf7d0"),
        "error":   ("❌", "#fef2f2", "#dc2626", "#991b1b", "#fecaca"),
        "warning": ("⚠️", "#fffbeb", "#d97706", "#92400e", "#fde68a"),
    }
    icon, bg, _, text_color, border_color = styles.get(kind, styles["success"])
    return f'''<div style="display:flex;align-items:center;gap:10px;padding:12px 16px;background:{bg};border-radius:8px;border:1px solid {border_color};margin:8px 0;">
  <span style="font-size:1.2em;">{icon}</span>
  <span style="color:{text_color};font-weight:500;">{msg}</span>
</div>'''

# ===== INTERACTIVE TAB: GLOBAL PROCESSOR INITIALIZATION =====
# Initialize once at module load to avoid reloading models
from interface.interactive_processor import InteractiveReviewProcessor
from dependencies.sentence_filter import (
    is_noise_sentence, filter_and_clean_sentences, strip_header_prefix,
    HIGHLIGHT_THRESHOLD, compute_informativeness,
)
_interactive_processor = None

def get_interactive_processor():
    """Lazy-load the processor to avoid duplicate model loading."""
    global _interactive_processor
    if _interactive_processor is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _interactive_processor = InteractiveReviewProcessor(device=device)
    return _interactive_processor


MAX_INTERACTIVE_REVIEWS = 6


def fetch_openreview_reviews(link: str):
    """
    Fetch reviews from OpenReview link and populate the textboxes.

    Returns:
        Tuple of (review1..6, title, status_html)
    """
    print(f"\n[DEMO] fetch_openreview_reviews called with link: {link}")

    if not link.strip():
        raise gr.Error("Please paste a valid OpenReview link before fetching.")

    try:
        from interface.interactive_processor import fetch_reviews_from_openreview_link
        reviews, title, rebuttal = fetch_reviews_from_openreview_link(link)
        print(f"[DEMO] Got {len(reviews)} reviews from fetch function")

        while len(reviews) < MAX_INTERACTIVE_REVIEWS:
            reviews.append("")
        reviews = reviews[:MAX_INTERACTIVE_REVIEWS]

        num_reviews = len([r for r in reviews if r.strip()])
        status = _status_html(f"Fetched {num_reviews} reviews for: {title}", "success")
        return (*reviews, title, rebuttal, status)

    except gr.Error:
        raise
    except ValueError as e:
        raise gr.Error(str(e))
    except Exception as e:
        error_msg = str(e)
        print(f"[DEMO] Exception caught: {type(e).__name__}: {error_msg}")

        if "openreview" in error_msg.lower():
            suggestion = " Try: pip install openreview-py"
        elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            suggestion = " Check your internet connection."
        else:
            suggestion = ""

        raise gr.Error(f"{error_msg}{suggestion}")


def _parse_rebuttal_json(rebuttal: str) -> Optional[list]:
    """Parse rebuttal JSON string, returning list of items or None."""
    if not rebuttal or not rebuttal.strip():
        return None
    try:
        items = json.loads(rebuttal)
        return items if items else None
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None


_REBUTTAL_PER_REVIEW_STYLE = (
    "margin-top:8px;margin-bottom:12px;border-radius:6px;overflow:hidden;"
    "border:1px solid #fde68a;background:#fffef5;"
)
_REBUTTAL_GENERAL_STYLE = (
    "margin-top:16px;border-radius:8px;overflow:hidden;"
    "border:1px solid #fde68a;"
)


def format_rebuttal_plain(text: str) -> str:
    """Format a plain text rebuttal as collapsible HTML.
    For pre-processed data where each review has its own rebuttal string."""
    if not text or not text.strip():
        return ""
    return (
        f'<details style="{_REBUTTAL_PER_REVIEW_STYLE}">'
        '<summary style="padding:10px 14px;cursor:pointer;font-size:0.75em;color:#92400e;'
        'font-weight:600;list-style:none;display:flex;align-items:center;gap:6px;">'
        '<span style="transition:transform 0.2s;">▶</span> Author Response</summary>'
        '<div style="padding:10px 14px;">'
        f'<div style="white-space:pre-wrap;color:#1f2937;font-size:0.85em;line-height:1.5;">{text}</div>'
        '</div></details>'
    )


def format_rebuttal_for_review(rebuttal: str, review_num: int) -> str:
    """Format rebuttals that reply to a specific review number."""
    if not rebuttal or not rebuttal.strip():
        return ""

    items = _parse_rebuttal_json(rebuttal)
    if items is not None:
        # Filter to only rebuttals for this review
        relevant = [item for item in items if item.get("reply_to") == review_num]
        if not relevant:
            return ""

        response_parts = []
        for i, item in enumerate(relevant):
            text = item.get("text", "").strip()
            if not text:
                continue

            response_parts.append(
                f'<div style="padding:10px 14px;">'
                f'<div style="white-space:pre-wrap;color:#1f2937;font-size:0.85em;line-height:1.5;">{text}</div>'
                f'</div>'
            )

        if not response_parts:
            return ""

        return (
            f'<details style="{_REBUTTAL_PER_REVIEW_STYLE}">'
            f'<summary style="padding:10px 14px;cursor:pointer;font-size:0.75em;color:#92400e;'
            f'font-weight:600;list-style:none;display:flex;align-items:center;gap:6px;">'
            f'<span style="transition:transform 0.2s;">▶</span> Author Response</summary>'
            + "".join(response_parts)
            + "</details>"
        )

    # Plain text - show under first review only
    if review_num == 1:
        text = rebuttal.strip()
        return (
            f'<details style="{_REBUTTAL_PER_REVIEW_STYLE}">'
            f'<summary style="padding:10px 14px;cursor:pointer;font-size:0.75em;color:#92400e;'
            f'font-weight:600;list-style:none;display:flex;align-items:center;gap:6px;">'
            f'<span style="transition:transform 0.2s;">▶</span> Author Response</summary>'
            f'<div style="padding:10px 14px;">'
            f'<div style="white-space:pre-wrap;color:#1f2937;font-size:0.85em;line-height:1.5;">{text}</div>'
            f'</div></details>'
        )
    return ""


def format_general_rebuttals(rebuttal: str) -> str:
    """Format general rebuttals (those not replying to a specific review)."""
    if not rebuttal or not rebuttal.strip():
        return ""

    HEADER_STYLE = "background:#fffbeb;padding:10px 16px;border-bottom:1px solid #fde68a;display:flex;align-items:center;gap:8px;"
    TITLE_STYLE = "font-weight:600;color:#92400e;"

    items = _parse_rebuttal_json(rebuttal)
    if items is not None:
        # Filter to only general rebuttals (no specific reply_to)
        general = [item for item in items if item.get("reply_to") is None]
        if not general:
            return ""

        response_parts = []
        for i, item in enumerate(general):
            text = item.get("text", "").strip()
            if not text:
                continue

            bg = "white" if i % 2 == 0 else "#fafafa"
            sep = "border-top:1px solid #fde68a;" if i > 0 else ""
            response_parts.append(
                f'<div style="padding:14px 16px;background:{bg};{sep}">'
                f'<div style="white-space:pre-wrap;color:#1f2937;font-size:0.9em;line-height:1.6;">{text}</div>'
                f'</div>'
            )

        if not response_parts:
            return ""

        count_label = f"{len(response_parts)} general response{'s' if len(response_parts) > 1 else ''}"
        return (
            f'<details style="{_REBUTTAL_GENERAL_STYLE}">'
            f'<summary style="{HEADER_STYLE}cursor:pointer;list-style:none;">'
            f'<span style="font-size:1.1em;">💬</span>'
            f'<span style="{TITLE_STYLE}">General Author Response</span>'
            f'<span style="margin-left:auto;font-size:0.8em;color:#78716c;">{count_label}</span>'
            f'</summary>'
            + "".join(response_parts) +
            '</details>'
        )

    # Plain text - treat as general response
    text = rebuttal.strip()
    return (
        f'<details style="{_REBUTTAL_GENERAL_STYLE}">'
        f'<summary style="{HEADER_STYLE}cursor:pointer;list-style:none;">'
        f'<span style="font-size:1.1em;">💬</span>'
        f'<span style="{TITLE_STYLE}">General Author Response</span></summary>'
        f'<div style="padding:14px 16px;background:white;">'
        f'<div style="white-space:pre-wrap;color:#1f2937;font-size:0.9em;line-height:1.6;">{text}</div>'
        f'</div></details>'
    )


@_gpu
def process_interactive_reviews_fast(text1: str, text2: str, text3: str, text4: str, text5: str, text6: str, focus: str, rebuttal_str: str = "", thread_state=None, progress=gr.Progress()) -> Tuple:
    """
    Fast processing: Polarity + Topic only (~3-5 sec on CPU).
    RSA (agreement) runs in background.
    If thread_state is provided, polarity+topic was already started during page transition —
    just wait for it instead of re-computing.
    """
    import time as _time
    from dependencies.Glimpse_tokenizer import glimpse_tokenizer

    t_start = _time.time()

    # Check if polarity+topic was already started in background by _show_raw_and_switch
    if thread_state and isinstance(thread_state, dict) and thread_state.get("thread"):
        bg_thread = thread_state["thread"]
        _result = thread_state["result"]
        sentence_lists = thread_state["sentence_lists"]
        active_texts = thread_state["active_texts"]
        all_sentences = thread_state["all_sentences"]

        progress(0.30, desc="Predicting polarity and topics...")

        # Wait for the background thread (may already be done!)
        bg_thread.join()

        if _result.get("error"):
            raise _result["error"]

        polarity_map = _result["polarity"]
        topic_map = _result["topic"]
        print(f"[TIMING] Polarity+Topic (from early-start thread): {_time.time() - t_start:.1f}s wait")
    else:
        # Fallback: compute from scratch (e.g. if thread_state was not passed)
        all_texts = [text1, text2, text3, text4, text5, text6]
        active_texts = [t for t in all_texts if t and t.strip()]

        if len(active_texts) < 2:
            raise ValueError("Please enter at least two reviews")

        progress(0.0, desc="Loading models...")
        t0 = _time.time()
        processor = get_interactive_processor()
        print(f"[TIMING] get_interactive_processor: {_time.time() - t0:.1f}s")

        progress(0.10, desc="Tokenizing reviews...")
        t0 = _time.time()
        sentence_lists = [[s for s in glimpse_tokenizer(t) if s.strip()] for t in active_texts]
        sentence_lists = [sl for sl in sentence_lists if sl]
        print(f"[TIMING] Tokenization: {_time.time() - t0:.1f}s ({sum(len(sl) for sl in sentence_lists)} total sentences)")

        if len(sentence_lists) < 2:
            raise ValueError("At least two reviews must have valid sentences")

        t0 = _time.time()
        all_sentences = filter_and_clean_sentences(
            list(set(s for sl in sentence_lists for s in sl))
        )
        print(f"[TIMING] filter_and_clean: {_time.time() - t0:.1f}s ({len(all_sentences)} unique sentences)")

        progress(0.30, desc="Predicting polarity and topics...")
        t0 = _time.time()
        polarity_map = processor.predict_polarity(all_sentences)
        topic_map = processor.predict_topic(all_sentences)
        print(f"[TIMING] Polarity+Topic (sequential): {_time.time() - t0:.1f}s")

    print(f"[TIMING] Fast processing total: {_time.time() - t_start:.1f}s")

    # Step 5: Format results as HTML with collapsible review cards
    progress(0.90, desc="Formatting results...")

    # Pre-compute per-review rebuttal HTML (embedded inside each card, like pre-processed tab)
    rebuttal_htmls = [format_rebuttal_for_review(rebuttal_str or "", i + 1) for i in range(MAX_INTERACTIVE_REVIEWS)]

    # Build per-review outputs as HTML (same format as pre-processed tab)
    none_out, agree_out, polar_out, topic_out = [], [], [], []
    for i in range(MAX_INTERACTIVE_REVIEWS):
        if i < len(sentence_lists):
            review_label = f"Review {i + 1}"
            reb = rebuttal_htmls[i]
            # Build (sentence, metadata) tuples for render_review_html
            plain_items = [(s, {}) for s in sentence_lists[i]]
            polar_items = [(s, {"polarity": polarity_map.get(s)}) for s in sentence_lists[i]]
            topic_items = [(s, {"topic": topic_map.get(s)}) for s in sentence_lists[i]]

            none_html  = build_review_card(review_label, review_items=plain_items, mode="plain", rebuttal_html=reb)
            polar_html = build_review_card(review_label, review_items=polar_items, mode="polarity", rebuttal_html=reb)
            topic_html = build_review_card(review_label, review_items=topic_items, mode="topic", rebuttal_html=reb)

            none_out.append(gr.update(visible=True, value=none_html))
            agree_out.append(gr.update(visible=False, value=""))
            polar_out.append(gr.update(visible=False, value=polar_html))
            topic_out.append(gr.update(visible=False, value=topic_html))
        else:
            none_out.append(gr.update(visible=False, value=""))
            agree_out.append(gr.update(visible=False, value=""))
            polar_out.append(gr.update(visible=False, value=""))
            topic_out.append(gr.update(visible=False, value=""))

    progress(1.0, desc="Done! Computing agreement in background...")

    # Store sentence lists and texts in state for async RSA
    rsa_state = {
        "sentence_lists": sentence_lists,
        "active_texts": active_texts,
        "polarity_map": polarity_map,
        "topic_map": topic_map,
        "rebuttal_str": rebuttal_str or "",
    }

    return (
        *none_out,
        *agree_out,
        "", "",  # most_common, most_unique (will be filled when RSA done)
        *polar_out,
        *topic_out,
        len(sentence_lists),  # active review count
        rsa_state,  # state for async RSA
    )


def _fmt_time(sec: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS like tqdm."""
    if sec is None:
        return "?"
    sec = int(sec)
    if sec < 3600:
        return f"{sec // 60:02d}:{sec % 60:02d}"
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def _agreement_progress_html(pct: int, done: int, total: int,
                              eta_sec: float = None, elapsed: float = None,
                              rate: float = None) -> str:
    """Progress bar HTML for the agreement computation, tqdm-style [elapsed<eta, rate s/it]."""
    if done > 0 and elapsed is not None:
        info = f"{done}/{total} [{_fmt_time(elapsed)}<{_fmt_time(eta_sec)}, {rate:.1f}s/it]"
    else:
        info = f"{done}/{total}"
    return f"""
<div style="padding:10px 16px;background:#f0f4ff;border-radius:8px;border:1px solid #c7d2fe;margin:0;">
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-weight:600;color:#312e81;font-size:0.9em;white-space:nowrap;">Computing agreement: {pct}%</span>
    <div style="flex:1;background:#e0e7ff;border-radius:4px;height:6px;overflow:hidden;">
      <div style="background:linear-gradient(90deg,#818cf8,#4f46e5);height:100%;width:{pct}%;border-radius:4px;transition:width 0.4s ease;"></div>
    </div>
    <span style="font-size:0.78em;color:#6b7280;white-space:nowrap;">{info}</span>
  </div>
</div>
"""


def compute_rsa_in_background(rsa_state: Dict, current_focus: str):
    """
    Generator: streams real progress to agreement_progress_html while RSA runs in a thread,
    then yields final agreement HTML + summary cards.
    Outputs: agreement_text1..6, most_common, most_divergent, agreement_progress_html, focus_radio
    """
    _empty = tuple([gr.update(visible=False, value="")] * (MAX_INTERACTIVE_REVIEWS + 2)
                   + [gr.update(visible=False, value=""),
                      gr.update(choices=["No Highlighting", "Polarity", "Topic", "Agreement"], interactive=True)])

    if not rsa_state or not rsa_state.get("sentence_lists"):
        yield _empty
        return

    processor = get_interactive_processor()
    sentence_lists = rsa_state["sentence_lists"]
    active_texts = rsa_state["active_texts"]

    # Shared progress state updated by progress_callback from the RSA thread
    _prog = {"done": 0, "total": 1, "result": None, "error": None, "start_time": None}

    def _progress_callback(done, total):
        if _prog["start_time"] is None:
            _prog["start_time"] = time.time()
        _prog["done"] = done
        _prog["total"] = total

    def _run():
        try:
            _prog["result"] = processor.predict_rsa_full(*active_texts, progress_callback=_progress_callback)
        except Exception as e:
            _prog["error"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Yield progress updates while RSA thread is running
    _no_op_8 = [gr.update()] * (MAX_INTERACTIVE_REVIEWS + 2)
    while t.is_alive():
        done, total = _prog["done"], _prog["total"]
        pct = int(done / total * 100) if total > 0 else 0
        # tqdm-style ETA: elapsed/done * remaining
        eta_sec = None
        elapsed = None
        rate = None
        t0 = _prog.get("start_time")
        if t0 and done > 0:
            elapsed = time.time() - t0
            rate = elapsed / done  # seconds per batch
            eta_sec = rate * (total - done)
        yield (*_no_op_8, gr.update(visible=True, value=_agreement_progress_html(pct, done, total, eta_sec, elapsed, rate)), gr.update())
        time.sleep(0.4)

    t.join()

    if _prog["error"]:
        print(f"[RSA ERROR] {type(_prog['error']).__name__}: {_prog['error']}")
        error_msg = f"❌ Agreement computation failed: {str(_prog['error'])[:100]}"
        yield (
            *[gr.update(visible=False, value="")] * MAX_INTERACTIVE_REVIEWS,
            error_msg, "",
            gr.update(visible=False, value=""),
            gr.update(choices=["No Highlighting", "Polarity", "Topic", "Agreement"], interactive=True),
        )
        return

    rsa_result = _prog["result"] or {}
    uniqueness = rsa_result.get("uniqueness", {})
    listener = rsa_result.get("listener", {})
    speaker = rsa_result.get("speaker", {})

    polarity_map = rsa_state.get("polarity_map", {})
    topic_map = rsa_state.get("topic_map", {})
    most_common_text = format_common_themes(
        sentence_lists, polarity_map, topic_map,
        speaker=speaker, uniqueness=uniqueness, listener=listener,
    )

    if uniqueness:
        divergent_per_review = format_divergent_cards(uniqueness, sentence_lists, listener, speaker)
    else:
        divergent_per_review = {}

    show_agreement = current_focus in ("Agreement", "Agreement (Processing)")
    num_reviews = len(active_texts)

    # Pre-compute per-review rebuttal HTML (embedded inside agreement cards)
    rebuttal_str = rsa_state.get("rebuttal_str", "")
    rebuttal_htmls = [format_rebuttal_for_review(rebuttal_str, i + 1) for i in range(MAX_INTERACTIVE_REVIEWS)]

    agree_out = []
    for i in range(MAX_INTERACTIVE_REVIEWS):
        if i < len(sentence_lists):
            html_val = build_review_card(
                f"Agreement in Review {i + 1}",
                sentences=sentence_lists[i],
                uniqueness=uniqueness, listener=listener, speaker=speaker,
                num_reviews=num_reviews,
                divergent_html=divergent_per_review.get(i, ""),
                rebuttal_html=rebuttal_htmls[i],
            )
            agree_out.append(gr.update(visible=show_agreement, value=html_val))
        else:
            agree_out.append(gr.update(visible=False, value=""))

    yield (
        *agree_out,
        most_common_text, "",
        gr.update(visible=False, value=""),
        gr.update(choices=["No Highlighting", "Polarity", "Topic", "Agreement"], interactive=True),
    )




CUSTOM_CSS = """
.review-section-header h3 {
    color: #1e40af;
    border-left: 4px solid #3b82f6;
    padding-left: 10px;
    margin-top: 16px;
}
.rebuttal-section-header h3 {
    color: #92400e;
    border-left: 4px solid #f59e0b;
    padding-left: 10px;
    margin-top: 16px;
}

/* RSA sentence tooltip styles — uses JS positioning via onmouseenter */
.rsa-sentence {
    position: relative;
    cursor: help;
    padding: 1px 3px;
    border-radius: 2px;
    display: inline;
}
.rsa-tooltip {
    display: none;
    position: fixed;
    background: #1f2937;
    color: white;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 0.78em;
    z-index: 10000;
    pointer-events: none;
    max-width: 280px;
    width: max-content;
    white-space: normal;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}

/* Collapsible author response toggle */
details summary::-webkit-details-marker { display: none; }
details[open] summary span:first-child { display: inline-block; transform: rotate(90deg); }

/* Smooth scrolling everywhere */
html, body, .gradio-container, main, .contain { scroll-behavior: smooth !important; }

/* Back to top button */
#back-to-top-btn {
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 9999;
    padding: 8px 14px;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    background: white;
    color: #374151;
    font-size: 0.82em;
    font-weight: 500;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    display: none;
    transition: opacity 0.2s;
}
#back-to-top-btn:hover { background: #f3f4f6; }

/* Paper title heading style for interactive tab */
.paper-title-heading textarea {
    font-size: 1.17em !important;
    font-weight: 700 !important;
    color: #1f2937 !important;
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    line-height: 1.3 !important;
    min-height: 0 !important;
}
.paper-title-heading { padding: 0 !important; margin: 0 !important; min-height: 0 !important; }

/* Zero-height review anchor elements for jump navigation */
.review-anchor { height: 0 !important; overflow: hidden !important; margin: 0 !important; padding: 0 !important; min-height: 0 !important; border: none !important; }
.review-anchor > * { height: 0 !important; margin: 0 !important; padding: 0 !important; }

/* Tighter vertical spacing in results section */
.results-compact { gap: 4px !important; }


/* Progress bar group — zero internal spacing, collapses when both children hidden */
.progress-group, .progress-group > * {
    gap: 0 !important; padding: 0 !important; margin: 0 !important;
    border: none !important; box-shadow: none !important;
    border-radius: 0 !important; min-height: 0 !important;
}
/* Remove Gradio wrapper spacing around individual progress bars */
.progress-compact { margin: 0 !important; padding: 0 !important; width: 100% !important; min-height: 0 !important; }
/* Suppress Gradio's loading/progress indicator on progress bar components */
.progress-compact .progress-bar, .progress-compact .eta-bar,
.progress-compact > .wrap, .progress-compact .generating { display: none !important; }

/* Suppress Gradio's orange "pending update" pulsing border */
.generating { animation: none !important; border-color: #e5e7eb !important; box-shadow: none !important; }

/* Remove the border/separator line around the display mode radio row */
.no-border-row { border: none !important; box-shadow: none !important; padding: 0 !important; margin-bottom: 0 !important; }

/* Progress bar animations — global so they survive HTML replacement in generators */
@keyframes procspin { to { transform: rotate(360deg); } }
@keyframes agrslide { 0% { width:15%; margin-left:0; } 50% { width:35%; margin-left:50%; } 100% { width:15%; margin-left:85%; } }
"""

with gr.Blocks(
    title="ReView",
    css=CUSTOM_CSS,
    theme=gr.themes.Default(),
    js="""() => {
        document.querySelector('body').classList.remove('dark');
        var btn = document.createElement('button');
        btn.id = 'back-to-top-btn';
        btn.textContent = '\\u2191 Top';
        btn.onclick = function() {
            window.scrollTo({top: 0, behavior: 'smooth'});
            document.documentElement.scrollTop = 0;
            document.body.scrollTop = 0;
            var els = document.querySelectorAll('.gradio-container, main, .contain, .main');
            for (var i = 0; i < els.length; i++) els[i].scrollTop = 0;
        };
        document.body.appendChild(btn);
        function getMaxScroll() {
            var y = window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0;
            var els = document.querySelectorAll('.gradio-container, main, .contain, .main');
            for (var i = 0; i < els.length; i++) {
                if (els[i].scrollTop > y) y = els[i].scrollTop;
            }
            return y;
        }
        document.addEventListener('scroll', function() {
            btn.style.display = getMaxScroll() > 400 ? '' : 'none';
        }, true);
        setInterval(function() {
            btn.style.display = getMaxScroll() > 400 ? '' : 'none';
        }, 500);

        /* Gray-out and prevent selection of radio choices containing ⏳ */
        var _prevRadio = 'No Highlighting';
        setInterval(function() {
            var labels = document.querySelectorAll('.no-border-row label input[type=radio]');
            labels.forEach(function(inp) {
                var lbl = inp.closest('label') || inp.parentElement;
                var txt = (lbl && lbl.textContent) || '';
                if (txt.indexOf('⏳') !== -1) {
                    lbl.style.opacity = '0.4';
                    lbl.style.pointerEvents = 'none';
                    lbl.title = 'Still computing…';
                } else {
                    lbl.style.opacity = '';
                    lbl.style.pointerEvents = '';
                    lbl.title = '';
                }
            });
        }, 300);
    }""",
) as demo:
    # gr.Markdown("# ReView Interface")
    
    # TODO: Uncomment this for home/description tab once finished with testing.
    # with gr.Tab("Introduction"):
    #     gr.Markdown(glimpse_description)
        
    # -----------------------------------
    # Pre-processed Tab
    # -----------------------------------
    with gr.Tab("Pre-processed Reviews", elem_classes=["results-compact"]):
        # Initialize state for this session.
        if not years:
            raise ValueError("No years available in new dataset")
        initial_year = years[0]
        initial_scored_reviews = get_preprocessed_scores(initial_year)
        initial_review_ids = list(initial_scored_reviews.keys())
        initial_review = initial_scored_reviews[initial_review_ids[0]]
        number_of_displayed_reviews = len(initial_scored_reviews[initial_review_ids[0]])
        initial_metadata = get_preprocessed_metadata(initial_year)
        initial_state = {
            "year_choice": initial_year,
            "scored_reviews_for_year": initial_scored_reviews,
            "review_ids": initial_review_ids,
            "current_review_index": 0,
            "current_review": initial_review,
            "number_of_displayed_reviews": number_of_displayed_reviews,
            "metadata_for_year": initial_metadata,
        }
        state = gr.State(initial_state)

        def update_review_display(state, score_type):

            review_ids = state["review_ids"]
            current_index = state["current_review_index"]
            current_review = state["scored_reviews_for_year"][review_ids[current_index]]

            show_polarity = score_type == "Polarity"
            show_consensuality = score_type == "Agreement"
            show_topic = score_type == "Topic"
            
            
            if show_polarity:
                color_map = {"➕": "#d4fcd6", "➖": "#fcd6d6"}
                legend = False
            elif show_topic:
                color_map = topic_color_map  # No color map for topics
                legend = False
            elif show_consensuality:
                color_map = None  # Continuous scale, no predefined colors
                legend = True
            else:
                color_map = {}  # Default to empty map
                legend = False

            current_id = review_ids[current_index]
            # Primary source: raw CSV lookup (processed CSVs lack paper_title)
            paper_title = _paper_titles.get(current_id, "")
            # Fallback: metadata column in preprocessed CSV
            if not paper_title:
                paper_meta = state.get("metadata_for_year", {}).get(current_id, {})
                paper_title = paper_meta.get("paper_title", "") if isinstance(paper_meta, dict) else ""
            if paper_title:
                new_review_id = (
                    f"### {paper_title}\n\n"
                    f"[View on OpenReview]({current_id}) &nbsp;·&nbsp; "
                    f"({current_index + 1} of {len(state['review_ids'])} submissions)"
                )
            else:
                new_review_id = (
                    f"### [View on OpenReview]({current_id})\n\n"
                    f"({current_index + 1} of {len(state['review_ids'])} submissions)"
                )

            number_of_displayed_reviews = len(current_review)
            review_updates = []
            rebuttal_updates = []
            consensuality_dict = {}

            # Pre-compute robust normalization stats (median + IQR) for raw KL scores
            _kl_median, _kl_iqr = 0.0, 0.0
            if show_consensuality:
                all_raw_scores = []
                for review_data in current_review:
                    if isinstance(review_data, dict) and "sentences" in review_data:
                        items = review_data["sentences"].items()
                    else:
                        items = review_data.items() if isinstance(review_data, dict) else []
                    for _, metadata in items:
                        all_raw_scores.append(metadata.get("consensuality", 0.0))
                if all_raw_scores:
                    arr = np.array(all_raw_scores)
                    _kl_median = float(np.median(arr))
                    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
                    _kl_iqr = q75 - q25

            # Build per-review sentence lists, cache, and polarity/topic maps in a single pass
            review_sentence_lists = []
            review_items_cache = []  # Cache (review_item, rebuttal_html) per review
            prep_polarity_map = {}
            prep_topic_map = {}
            for idx in range(number_of_displayed_reviews):
                review_data = current_review[idx]
                rebuttal_html = ""
                if isinstance(review_data, dict) and "sentences" in review_data:
                    review_item = list(review_data["sentences"].items())
                    rebuttal_html = format_rebuttal_plain(review_data.get("rebuttal", ""))
                else:
                    review_item = list(review_data.items())
                review_sentence_lists.append([s for s, _ in review_item])
                review_items_cache.append((review_item, rebuttal_html))

                # Build polarity/topic maps from pre-processed metadata
                for sent, meta in review_item:
                    if not isinstance(meta, dict):
                        continue
                    pol_val = meta.get("polarity")
                    if pol_val == 0:
                        prep_polarity_map[sent] = "➖"
                    elif pol_val == 2:
                        prep_polarity_map[sent] = "➕"
                    topic = meta.get("topic")
                    if topic and topic != "NONE":
                        prep_topic_map[sent] = topic

            # For agreement mode, build uniqueness dict and extract RSA distributions
            # RSA listener/speaker come from metadata (if pipeline saved them)
            prep_listener = None
            prep_speaker = None
            if show_consensuality:
                for idx in range(number_of_displayed_reviews):
                    review_item, _ = review_items_cache[idx]
                    for sentence, metadata in review_item:
                        raw = metadata.get("consensuality", 0.0)
                        if _kl_iqr > 0:
                            score = max(-1.0, min(1.0, (raw - _kl_median) / (_kl_iqr * 2)))
                        else:
                            score = 0.0
                        if not is_noise_sentence(sentence) and abs(score) >= HIGHLIGHT_THRESHOLD:
                            consensuality_dict[sentence] = score

                # Extract listener/speaker from metadata (saved by pipeline)
                meta_for_year = state.get("metadata_for_year", {})
                submission_meta = meta_for_year.get(current_id, {})
                if isinstance(submission_meta, dict):
                    rsa_data = submission_meta.get("rsa", {})
                    if rsa_data:
                        prep_listener = rsa_data.get("listener")
                        prep_speaker = rsa_data.get("speaker")

            agreement_updates = []
            divergent_per_review = {}
            # Pre-compute per-review divergent cards if we have RSA data
            if show_consensuality and prep_listener and prep_speaker and consensuality_dict:
                divergent_per_review = format_divergent_cards(
                    consensuality_dict, review_sentence_lists, prep_listener, prep_speaker,
                )

            for i in range(MAX_PREPROCESSED_REVIEWS):
                if i < number_of_displayed_reviews:
                    review_item, rebuttal_html = review_items_cache[i]

                    # All modes now use HTML rendering for proper paragraph formatting.
                    # HighlightedText is always hidden; prep_agreement HTML is always shown.
                    review_updates.append(
                        gr.update(
                            visible=False,
                            value=[],
                            show_legend=False,
                            color_map=color_map,
                            key=f"updated_{score_type}_{i}"
                        )
                    )

                    review_label = f"Review {i + 1}"
                    if show_consensuality:
                        html_content = build_review_card(
                            review_label,
                            sentences=[s for s, _ in review_item],
                            uniqueness=consensuality_dict,
                            listener=prep_listener, speaker=prep_speaker,
                            num_reviews=number_of_displayed_reviews,
                            divergent_html=divergent_per_review.get(i, ""),
                            rebuttal_html=rebuttal_html,
                        )
                    else:
                        m = "polarity" if show_polarity else ("topic" if show_topic else "plain")
                        html_content = build_review_card(
                            review_label, review_items=review_item, mode=m,
                            rebuttal_html=rebuttal_html,
                        )

                    agreement_updates.append(gr.update(visible=True, value=html_content))
                    # Rebuttal is now embedded in the review card, so hide the separate component
                    rebuttal_updates.append(gr.update(visible=False, value=""))
                else:
                    review_updates.append(
                        gr.update(
                            visible=False,
                            value=[],
                            show_legend=False,
                            color_map=color_map,
                            key=f"updated_{score_type}_{i}"
                        )
                    )
                    agreement_updates.append(gr.update(visible=False, value=""))
                    rebuttal_updates.append(gr.update(visible=False, value=""))

            # General rebuttal display (currently unused in new format, kept for backward compat)
            general_rebuttal_update = gr.update(visible=False, value="")

            # Common Themes (topic+polarity grouping) — consistent with interactive tab
            if show_consensuality:
                most_common_html = format_common_themes(
                    review_sentence_lists, prep_polarity_map, prep_topic_map,
                    speaker=prep_speaker, uniqueness=consensuality_dict if consensuality_dict else None,
                    listener=prep_listener,
                )

                most_common_visibility = gr.update(visible=True, value=most_common_html)
                most_unique_visibility = gr.update(visible=False, value="")
            else:
                most_common_visibility = gr.update(visible=False, value="")
                most_unique_visibility = gr.update(visible=False, value="")
                
            # update color legend (topic or polarity)
            if show_polarity:
                polarity_legend = (
                    '<div style="display:flex;gap:12px;align-items:center;padding:8px 0;font-size:0.8em;">'
                    '<span style="background:#d4fcd6;padding:2px 8px;border-radius:4px;">Positive</span>'
                    '<span style="background:#fcd6d6;padding:2px 8px;border-radius:4px;">Negative</span>'
                    '<span style="color:#9ca3af;">Neutral (no highlight)</span>'
                    '</div>'
                )
                topic_color_map_visibility = gr.update(visible=True, value=polarity_legend)
            elif show_topic:
                legend_items = " ".join(
                    f'<span style="background:{color};padding:2px 8px;border-radius:4px;'
                    f'font-size:0.8em;margin-right:4px;">{_html.escape(name)}</span>'
                    for name, color in _TOPIC_HTML_COLORS.items()
                )
                topic_legend_html = (
                    f'<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center;'
                    f'padding:8px 0;">{legend_items}</div>'
                )
                topic_color_map_visibility = gr.update(visible=True, value=topic_legend_html)
            else:
                topic_color_map_visibility = gr.update(visible=False, value="")

            # Toggle bar: review collapse + rebuttal expand buttons
            has_any_rebuttal = any(
                rebuttal_html
                for _, rebuttal_html in review_items_cache
            )
            toggle_buttons = [_review_toggle_html()]
            if has_any_rebuttal:
                toggle_buttons.append(_rebuttal_toggle_html())
            jump_html = _jump_buttons_html(number_of_displayed_reviews, prefix="pre")
            toggle_bar_html = (
                '<div style="display:flex;align-items:center;gap:8px;">'
                f'<span style="font-size:0.78em;color:#6b7280;white-space:nowrap;">Jump to:</span>'
                + jump_html
                + '<span style="flex:1;"></span>'
                + "".join(toggle_buttons) + '</div>'
            )
            toggle_bar_update = gr.update(visible=True, value=toggle_bar_html)

            return (
                new_review_id,
                *review_updates,
                *agreement_updates,  # 10 agreement HTML sections
                most_common_visibility,
                most_unique_visibility,
                topic_color_map_visibility,
                toggle_bar_update,  # Review collapse + rebuttal expand buttons
                *rebuttal_updates,  # 10 per-review rebuttals
                general_rebuttal_update,  # General rebuttal section
                state
            )



        # Precompute the initial outputs so something is shown on load.
        init_display = update_review_display(initial_state, score_type="Original")
        # init_display returns: (review_id, review1..10, agreement1..10, most_common, most_unique, topic_box, prep_rebuttal1..10, prep_general_rebuttal, state)

        with gr.Row():
            
            with gr.Column(scale=1):
                review_id = gr.Markdown(value=init_display[0], container=True)
                with gr.Row():
                    previous_button = gr.Button("Previous", variant="secondary", interactive=True)
                    next_button = gr.Button("Next", variant="primary", interactive=True)
                    
                    
            with gr.Column(scale=1):
                # Input controls.
                year = gr.Dropdown(choices=years, label="Select Year", interactive=True, value=initial_year)
                score_type = gr.Radio(
                    choices=["No Highlighting", "Polarity", "Topic", "Agreement"],
                    label="Display Mode:",
                    value="No Highlighting",
                    interactive=True
                )

        # Output display.
        with gr.Row():
            most_common_sentences = gr.HTML(
                visible=False,
                value="",
                label="Most Common Opinions",
            )
            most_unique_sentences = gr.HTML(
                visible=False,
                value="",
                label="Most Divergent Opinions",
            )
        
        # Topic color legend (HTML version)
        topic_text_box = gr.HTML(visible=False, value="")
        
        prep_toggle_bar = gr.HTML(visible=False, value="")
        gr.HTML(value='<div id="pre-review-anchor-1"></div>', elem_classes=["review-anchor"])
        review1 = gr.HighlightedText(show_legend=False, label="📝 Review 1", visible=number_of_displayed_reviews >= 1, key="initial_review1")
        prep_agreement1 = gr.HTML(visible=False, value="")
        prep_rebuttal1 = gr.HTML(visible=False, value="")
        gr.HTML(value='<div id="pre-review-anchor-2"></div>', elem_classes=["review-anchor"])
        review2 = gr.HighlightedText(show_legend=False, label="📝 Review 2", visible=number_of_displayed_reviews >= 2, key="initial_review2")
        prep_agreement2 = gr.HTML(visible=False, value="")
        prep_rebuttal2 = gr.HTML(visible=False, value="")
        gr.HTML(value='<div id="pre-review-anchor-3"></div>', elem_classes=["review-anchor"])
        review3 = gr.HighlightedText(show_legend=False, label="📝 Review 3", visible=number_of_displayed_reviews >= 3, key="initial_review3")
        prep_agreement3 = gr.HTML(visible=False, value="")
        prep_rebuttal3 = gr.HTML(visible=False, value="")
        gr.HTML(value='<div id="pre-review-anchor-4"></div>', elem_classes=["review-anchor"])
        review4 = gr.HighlightedText(show_legend=False, label="📝 Review 4", visible=number_of_displayed_reviews >= 4, key="initial_review4")
        prep_agreement4 = gr.HTML(visible=False, value="")
        prep_rebuttal4 = gr.HTML(visible=False, value="")
        gr.HTML(value='<div id="pre-review-anchor-5"></div>', elem_classes=["review-anchor"])
        review5 = gr.HighlightedText(show_legend=False, label="📝 Review 5", visible=number_of_displayed_reviews >= 5, key="initial_review5")
        prep_agreement5 = gr.HTML(visible=False, value="")
        prep_rebuttal5 = gr.HTML(visible=False, value="")
        gr.HTML(value='<div id="pre-review-anchor-6"></div>', elem_classes=["review-anchor"])
        review6 = gr.HighlightedText(show_legend=False, label="📝 Review 6", visible=number_of_displayed_reviews >= 6, key="initial_review6")
        prep_agreement6 = gr.HTML(visible=False, value="")
        prep_rebuttal6 = gr.HTML(visible=False, value="")
        gr.HTML(value='<div id="pre-review-anchor-7"></div>', elem_classes=["review-anchor"])
        review7 = gr.HighlightedText(show_legend=False, label="📝 Review 7", visible=number_of_displayed_reviews >= 7, key="initial_review7")
        prep_agreement7 = gr.HTML(visible=False, value="")
        prep_rebuttal7 = gr.HTML(visible=False, value="")
        gr.HTML(value='<div id="pre-review-anchor-8"></div>', elem_classes=["review-anchor"])
        review8 = gr.HighlightedText(show_legend=False, label="📝 Review 8", visible=number_of_displayed_reviews >= 8, key="initial_review8")
        prep_agreement8 = gr.HTML(visible=False, value="")
        prep_rebuttal8 = gr.HTML(visible=False, value="")
        gr.HTML(value='<div id="pre-review-anchor-9"></div>', elem_classes=["review-anchor"])
        review9 = gr.HighlightedText(show_legend=False, label="📝 Review 9", visible=number_of_displayed_reviews >= 9, key="initial_review9")
        prep_agreement9 = gr.HTML(visible=False, value="")
        prep_rebuttal9 = gr.HTML(visible=False, value="")
        gr.HTML(value='<div id="pre-review-anchor-10"></div>', elem_classes=["review-anchor"])
        review10 = gr.HighlightedText(show_legend=False, label="📝 Review 10", visible=number_of_displayed_reviews >= 10, key="initial_review10")
        prep_agreement10 = gr.HTML(visible=False, value="")
        prep_rebuttal10 = gr.HTML(visible=False, value="")

        # General rebuttal section (for rebuttals not tied to specific reviews)
        prep_general_rebuttal = gr.HTML(visible=False, value="")

        # Callback functions that update state.
        def year_change(year, state, score_type):
            state["year_choice"] = year
            state["scored_reviews_for_year"] = get_preprocessed_scores(year)
            state["metadata_for_year"] = get_preprocessed_metadata(year)
            state["review_ids"] = list(state["scored_reviews_for_year"].keys())
            state["current_review_index"] = 0
            state["current_review"] = state["scored_reviews_for_year"][state["review_ids"][0]]
            return update_review_display(state, score_type)

        def next_review(state, score_type):
            state["current_review_index"] = (state["current_review_index"] + 1) % len(state["review_ids"])
            state["current_review"] = state["scored_reviews_for_year"][state["review_ids"][state["current_review_index"]]]
            return update_review_display(state, score_type)

        def previous_review(state, score_type):
            state["current_review_index"] = (state["current_review_index"] - 1) % len(state["review_ids"])
            state["current_review"] = state["scored_reviews_for_year"][state["review_ids"][state["current_review_index"]]]
            return update_review_display(state, score_type)

        # Hook up the callbacks with the session state.
        _review_outputs = [review_id, review1, review2, review3, review4, review5, review6, review7, review8, review9, review10, prep_agreement1, prep_agreement2, prep_agreement3, prep_agreement4, prep_agreement5, prep_agreement6, prep_agreement7, prep_agreement8, prep_agreement9, prep_agreement10, most_common_sentences, most_unique_sentences, topic_text_box, prep_toggle_bar, prep_rebuttal1, prep_rebuttal2, prep_rebuttal3, prep_rebuttal4, prep_rebuttal5, prep_rebuttal6, prep_rebuttal7, prep_rebuttal8, prep_rebuttal9, prep_rebuttal10, prep_general_rebuttal, state]
        year.change(fn=year_change, inputs=[year, state, score_type], outputs=_review_outputs)
        score_type.change(fn=update_review_display, inputs=[state, score_type], outputs=_review_outputs)
        next_button.click(fn=next_review, inputs=[state, score_type], outputs=_review_outputs)
        previous_button.click(fn=previous_review, inputs=[state, score_type], outputs=_review_outputs)   
        
        
        
        
    # -----------------------------------
    # Interactive Tab
    # -----------------------------------
    with gr.Tab("Interactive", interactive=True):

        # ---- TOP TOGGLE BAR (always visible) ----
        with gr.Row():
            paper_title_html = gr.Textbox("", visible=False, interactive=False, show_label=False, container=False, elem_classes=["paper-title-heading"])
            back_to_input_btn = gr.Button("✏️ Edit Reviews / New Input", visible=False, variant="secondary")
            view_results_btn = gr.Button("📊 View Results", visible=False, variant="secondary")

        # ---- INPUT SECTION (full-width, visible initially) ----
        with gr.Column(visible=True) as input_section:

            with gr.Tabs():
                with gr.Tab("OpenReview Link"):
                    gr.Markdown("""
                    Paste an OpenReview forum link to automatically fetch and process its reviews.

                    **Example link:**
                    https://openreview.net/forum?id=...
                    """)
                    openreview_link_input = gr.Textbox(
                        label="OpenReview Forum Link",
                        placeholder="https://openreview.net/forum?id=...",
                        interactive=True
                    )
                    fetch_reviews_button = gr.Button("Fetch & Process", variant="primary", interactive=True)
                    openreview_title = gr.Textbox(label="Paper Title", interactive=False, visible=False, value="")
                    openreview_rebuttal = gr.Textbox(label="💬 Author Rebuttal", interactive=False, visible=False, value="", lines=3)

                with gr.Tab("Paste Reviews Manually"):
                    review1_textbox = gr.Textbox(lines=5, value=EXAMPLES[0], label="📝 Review 1", interactive=True)
                    review2_textbox = gr.Textbox(lines=5, value=EXAMPLES[1], label="📝 Review 2", interactive=True)
                    review3_textbox = gr.Textbox(lines=5, value=EXAMPLES[2], label="📝 Review 3", interactive=True)
                    review4_textbox = gr.Textbox(lines=5, value="", label="📝 Review 4", interactive=True, visible=False)
                    review5_textbox = gr.Textbox(lines=5, value="", label="📝 Review 5", interactive=True, visible=False)
                    review6_textbox = gr.Textbox(lines=5, value="", label="📝 Review 6", interactive=True, visible=False)
                    paste_rebuttal = gr.Textbox(label="💬 Author Rebuttal (optional)", interactive=True, lines=3, placeholder="Paste the author rebuttal here (optional)...")
                    interactive_review_count = gr.State(3)
                    with gr.Row():
                        add_review_btn = gr.Button("➕ Add Review", variant="secondary", interactive=True)
                        submit_button = gr.Button("Process", variant="primary", interactive=True)
                        clear_button = gr.Button("Clear", variant="secondary", interactive=True)

            status_html = gr.HTML("", visible=False)

        # ---- RESULTS SECTION (full-width, hidden initially) ----
        with gr.Column(visible=False, elem_classes=["results-compact"]) as results_section:
            with gr.Row(elem_classes=["no-border-row"]):
                focus_radio = gr.Radio(
                    choices=["No Highlighting", "Polarity", "Topic", "Agreement (Processing)"],
                    value="No Highlighting",
                    label="Display Mode:",
                    interactive=True
                )

            # Progress bars in a zero-gap group so they sit flush against each other
            with gr.Group(elem_classes=["progress-group"]):
                polarity_progress_html = gr.HTML("", visible=False, elem_classes=["progress-compact"])
                agreement_progress_html = gr.HTML("", visible=False, elem_classes=["progress-compact"])

            # Color legend for polarity/topic modes (hidden by default, shown by toggle_display_mode)
            interactive_legend_html = gr.HTML("", visible=False)

            with gr.Row():
                most_divergent = gr.HTML(
                    visible=False, value="", label="Most Divergent Opinions",
                )
                most_common = gr.HTML(
                    visible=False, value="", label="Most Common Opinions",
                )

            interactive_rebuttal_toggle = gr.HTML(visible=False, value="")

            # Review 1 (all display modes as HTML + rebuttal)
            gr.HTML(value='<div id="int-review-anchor-1"></div>', elem_classes=["review-anchor"])
            none_text1 = gr.HTML(visible=True, value="", elem_id="int-review-1")
            agreement_text1 = gr.HTML(visible=False, value="")
            polarity_text1 = gr.HTML(visible=False, value="")
            topic_text1 = gr.HTML(visible=False, value="")
            rebuttal_for_review1 = gr.HTML(visible=False, value="")

            # Review 2
            gr.HTML(value='<div id="int-review-anchor-2"></div>', elem_classes=["review-anchor"])
            none_text2 = gr.HTML(visible=False, value="", elem_id="int-review-2")
            agreement_text2 = gr.HTML(visible=False, value="")
            polarity_text2 = gr.HTML(visible=False, value="")
            topic_text2 = gr.HTML(visible=False, value="")
            rebuttal_for_review2 = gr.HTML(visible=False, value="")

            # Review 3
            gr.HTML(value='<div id="int-review-anchor-3"></div>', elem_classes=["review-anchor"])
            none_text3 = gr.HTML(visible=False, value="", elem_id="int-review-3")
            agreement_text3 = gr.HTML(visible=False, value="")
            polarity_text3 = gr.HTML(visible=False, value="")
            topic_text3 = gr.HTML(visible=False, value="")
            rebuttal_for_review3 = gr.HTML(visible=False, value="")

            # Review 4
            gr.HTML(value='<div id="int-review-anchor-4"></div>', elem_classes=["review-anchor"])
            none_text4 = gr.HTML(visible=False, value="", elem_id="int-review-4")
            agreement_text4 = gr.HTML(visible=False, value="")
            polarity_text4 = gr.HTML(visible=False, value="")
            topic_text4 = gr.HTML(visible=False, value="")
            rebuttal_for_review4 = gr.HTML(visible=False, value="")

            # Review 5
            gr.HTML(value='<div id="int-review-anchor-5"></div>', elem_classes=["review-anchor"])
            none_text5 = gr.HTML(visible=False, value="", elem_id="int-review-5")
            agreement_text5 = gr.HTML(visible=False, value="")
            polarity_text5 = gr.HTML(visible=False, value="")
            topic_text5 = gr.HTML(visible=False, value="")
            rebuttal_for_review5 = gr.HTML(visible=False, value="")

            # Review 6
            gr.HTML(value='<div id="int-review-anchor-6"></div>', elem_classes=["review-anchor"])
            none_text6 = gr.HTML(visible=False, value="", elem_id="int-review-6")
            agreement_text6 = gr.HTML(visible=False, value="")
            polarity_text6 = gr.HTML(visible=False, value="")
            topic_text6 = gr.HTML(visible=False, value="")
            rebuttal_for_review6 = gr.HTML(visible=False, value="")

            # General rebuttal display (for rebuttals not tied to specific reviews)
            interactive_rebuttal_display = gr.HTML(visible=False, value="")

        # ---- CALLBACKS ----

        # State to hold raw rebuttal string (set by _show_raw_and_switch, consumed by process_interactive_reviews_fast)
        interactive_rebuttal_state = gr.State("")

        # State to hold background processing thread (polarity+topic starts during page transition)
        processing_thread_state = gr.State(None)

        _interactive_inputs = [review1_textbox, review2_textbox, review3_textbox, review4_textbox, review5_textbox, review6_textbox, focus_radio, interactive_rebuttal_state, processing_thread_state]

        # State to hold RSA computation results for async updates
        rsa_computation_state = gr.State({})

        # Sink states: absorb none_textN outputs from process_interactive_reviews_fast
        # (none_texts are already set by _show_raw_and_switch; routing them to states
        #  prevents Gradio from showing a "pending" loading spinner on the review cards)
        _none_sinks = [gr.State(None) for _ in range(6)]

        _interactive_outputs = [
            *_none_sinks,  # absorb none_text1..6 — already populated, no spinner wanted
            agreement_text1, agreement_text2, agreement_text3, agreement_text4, agreement_text5, agreement_text6,
            most_common, most_divergent,
            polarity_text1, polarity_text2, polarity_text3, polarity_text4, polarity_text5, polarity_text6,
            topic_text1, topic_text2, topic_text3, topic_text4, topic_text5, topic_text6,
            interactive_review_count,
            rsa_computation_state,
        ]

        # Outputs for RSA async computation (updates agreement sections + most common/unique)
        _rsa_outputs = [
            agreement_text1, agreement_text2, agreement_text3, agreement_text4, agreement_text5, agreement_text6,
            most_common, most_divergent,
            agreement_progress_html,
            focus_radio,
        ]

        # Fetch OpenReview reviews → show timer → fast process → swap to results → async RSA
        def _validate_and_start_fetch(link):
            if not link or not link.strip():
                raise gr.Error("Please paste a valid OpenReview link before fetching.")
            return gr.update(value=FETCHING_HTML, visible=True), gr.update(interactive=False)

        no_title_state = gr.State("")

        def _show_raw_and_switch(r1, r2, r3, r4, r5, r6, rebuttal, title=""):
            """Immediately switch to results view with raw tokenized reviews.
            Also kicks off polarity+topic in a background thread so processing
            overlaps with page transition rendering."""
            import time as _time
            from dependencies.Glimpse_tokenizer import glimpse_tokenizer
            texts = [r1, r2, r3, r4, r5, r6]
            active_count = sum(1 for t in texts if t and t.strip())

            # Pre-compute per-review rebuttal HTML (to embed inside review cards)
            rebuttal_htmls = [format_rebuttal_for_review(rebuttal or "", i + 1) for i in range(6)]
            has_per_review = any(rebuttal_htmls)

            # Tokenize each review and render as HTML with collapsible cards (fast, no ML)
            # Rebuttal is embedded INSIDE the card (like the pre-processed tab)
            none_out = []
            for idx, t in enumerate(texts):
                if t and t.strip():
                    sentences = [s for s in glimpse_tokenizer(t) if s.strip()]
                    plain_items = [(s, {}) for s in sentences]
                    html = build_review_card(f"Review {idx + 1}", review_items=plain_items, mode="plain", rebuttal_html=rebuttal_htmls[idx])
                    none_out.append(gr.update(visible=True, value=html))
                else:
                    none_out.append(gr.update(visible=False, value=""))

            # Per-review rebuttal components are now hidden (content embedded in cards above)
            per_review = [gr.update(visible=False, value="") for _ in range(6)]

            general_formatted = format_general_rebuttals(rebuttal or "")
            has_any = has_per_review or bool(general_formatted)

            # Toggle bar
            right_buttons = [_review_toggle_html()]
            if has_any:
                right_buttons.append(_rebuttal_toggle_html())
            toggle_bar = (
                '<div style="display:flex;align-items:center;gap:8px;">'
                '<span style="font-size:0.78em;color:#6b7280;white-space:nowrap;">Jump to:</span>'
                + _jump_buttons_html(active_count)
                + '<span style="flex:1;"></span>'
                + "".join(right_buttons) + '</div>'
            )

            title_text = title.strip() if title and title.strip() else ""

            # Start polarity+topic in background thread NOW, so processing
            # overlaps with Gradio rendering the page transition to the user.
            # By the time process_interactive_reviews_fast runs, work is already underway.
            active_texts = [t for t in texts if t and t.strip()]
            sentence_lists = []
            for t in active_texts:
                sents = [s for s in glimpse_tokenizer(t) if s.strip()]
                if sents:
                    sentence_lists.append(sents)
            all_sentences = filter_and_clean_sentences(
                list(set(s for sl in sentence_lists for s in sl))
            )

            processor = get_interactive_processor()
            _thread_result = {"polarity": None, "topic": None, "error": None}

            def _run_polarity_topic():
                try:
                    t0 = _time.time()
                    _thread_result["polarity"] = processor.predict_polarity(all_sentences)
                    _thread_result["topic"] = processor.predict_topic(all_sentences)
                    print(f"[TIMING] Early polarity+topic thread done in {_time.time() - t0:.1f}s")
                except Exception as e:
                    _thread_result["error"] = e

            bg_thread = threading.Thread(target=_run_polarity_topic, daemon=True)
            bg_thread.start()
            print(f"[TIMING] Background polarity+topic thread started (page transitioning...)")

            thread_state = {
                "thread": bg_thread,
                "result": _thread_result,
                "sentence_lists": sentence_lists,
                "active_texts": active_texts,
                "all_sentences": all_sentences,
            }

            return (
                *none_out,                                              # none_text1..6
                gr.update(visible=False),                              # input_section
                gr.update(visible=True),                               # results_section
                gr.update(visible=True),                               # back_to_input_btn
                gr.update(visible=bool(title_text), value=title_text), # paper_title_html
                gr.update(visible=False),                              # view_results_btn
                gr.update(choices=["No Highlighting", "Polarity ⏳", "Topic ⏳", "Agreement ⏳"],
                           value="No Highlighting", interactive=True), # focus_radio
                gr.update(visible=True, value=POLARITY_PROGRESS_HTML), # polarity_progress_html
                gr.update(visible=True, value=AGREEMENT_PROGRESS_HTML),# agreement_progress_html
                gr.update(visible=True, value=toggle_bar),             # interactive_rebuttal_toggle
                *per_review,                                           # rebuttal_for_review1..6 (hidden, embedded in cards)
                gr.update(visible=bool(general_formatted), value=general_formatted),  # interactive_rebuttal_display
                active_count,                                          # interactive_review_count
                rebuttal or "",                                        # interactive_rebuttal_state
                gr.update(visible=False, value=""),                     # interactive_legend_html (reset on new submission)
                thread_state,                                           # processing_thread_state
            )

        def _show_results_with_rebuttal(rebuttal, active_count):
            # Generate per-review rebuttals
            per_review = []
            has_per_review = False
            for i in range(1, 7):
                formatted = format_rebuttal_for_review(rebuttal or "", i)
                if formatted:
                    has_per_review = True
                per_review.append(gr.update(visible=bool(formatted), value=formatted))

            # Generate general rebuttal section (only for rebuttals not tied to specific reviews)
            general_formatted = format_general_rebuttals(rebuttal or "")
            has_general = bool(general_formatted)

            # Toggle bar: jump buttons (left) + collapse toggles (right)
            has_any = has_per_review or has_general
            right_buttons = [_review_toggle_html()]
            if has_any:
                right_buttons.append(_rebuttal_toggle_html())
            toggle_bar = (
                '<div style="display:flex;align-items:center;gap:8px;">'
                '<span style="font-size:0.78em;color:#6b7280;white-space:nowrap;">Jump to:</span>'
                + _jump_buttons_html(active_count)
                + '<span style="flex:1;"></span>'
                + "".join(right_buttons) + '</div>'
            )
            toggle_update = gr.update(visible=True, value=toggle_bar)

            return (
                gr.update(visible=False),   # input_section
                gr.update(visible=True),    # results_section
                gr.update(value=AGREEMENT_PROGRESS_HTML, visible=True),  # agreement_progress_html (in results_section)
                gr.update(visible=True),    # back_to_input_btn
                gr.update(visible=False),   # view_results_btn
                gr.update(choices=["No Highlighting", "Polarity", "Topic", "Agreement (Processing)"], value="No Highlighting"),
                toggle_update,  # rebuttal toggle button
                *per_review,  # 6 per-review rebuttal components
                gr.update(visible=has_general, value=general_formatted),  # general rebuttal display (only if exists)
            )

        fetch_reviews_button.click(
            fn=_validate_and_start_fetch,
            inputs=[openreview_link_input],
            outputs=[status_html, fetch_reviews_button]
        ).success(
            fn=fetch_openreview_reviews,
            inputs=[openreview_link_input],
            outputs=[review1_textbox, review2_textbox, review3_textbox, review4_textbox, review5_textbox, review6_textbox,
                     openreview_title, openreview_rebuttal, status_html]
        ).success(
            fn=lambda r4, r5, r6: (
                gr.update(visible=bool(r4.strip())),
                gr.update(visible=bool(r5.strip())),
                gr.update(visible=bool(r6.strip())),
            ),
            inputs=[review4_textbox, review5_textbox, review6_textbox],
            outputs=[review4_textbox, review5_textbox, review6_textbox]
        ).success(
            fn=_show_raw_and_switch,
            inputs=[review1_textbox, review2_textbox, review3_textbox, review4_textbox, review5_textbox, review6_textbox,
                    openreview_rebuttal, openreview_title],
            outputs=[none_text1, none_text2, none_text3, none_text4, none_text5, none_text6,
                     input_section, results_section, back_to_input_btn, paper_title_html, view_results_btn, focus_radio,
                     polarity_progress_html, agreement_progress_html,
                     interactive_rebuttal_toggle,
                     rebuttal_for_review1, rebuttal_for_review2, rebuttal_for_review3,
                     rebuttal_for_review4, rebuttal_for_review5, rebuttal_for_review6,
                     interactive_rebuttal_display, interactive_review_count,
                     interactive_rebuttal_state, interactive_legend_html,
                     processing_thread_state]
        ).success(
            fn=process_interactive_reviews_fast,
            inputs=_interactive_inputs,
            outputs=_interactive_outputs
        ).success(
            fn=lambda: (
                gr.update(visible=False, value=""),
                gr.update(choices=["No Highlighting", "Polarity", "Topic", "Agreement ⏳"], interactive=True),
            ),
            inputs=[],
            outputs=[polarity_progress_html, focus_radio]
        ).success(
            fn=lambda: gr.update(interactive=True),
            inputs=[],
            outputs=[fetch_reviews_button]
        ).success(
            fn=compute_rsa_in_background,
            inputs=[rsa_computation_state, focus_radio],
            outputs=_rsa_outputs
        )

        # Process (Paste Reviews): show raw reviews immediately → fast scoring → async RSA
        submit_button.click(
            fn=lambda: gr.update(interactive=False),
            inputs=[],
            outputs=[submit_button]
        ).success(
            fn=_show_raw_and_switch,
            inputs=[review1_textbox, review2_textbox, review3_textbox, review4_textbox, review5_textbox, review6_textbox,
                    paste_rebuttal, no_title_state],
            outputs=[none_text1, none_text2, none_text3, none_text4, none_text5, none_text6,
                     input_section, results_section, back_to_input_btn, paper_title_html, view_results_btn, focus_radio,
                     polarity_progress_html, agreement_progress_html,
                     interactive_rebuttal_toggle,
                     rebuttal_for_review1, rebuttal_for_review2, rebuttal_for_review3,
                     rebuttal_for_review4, rebuttal_for_review5, rebuttal_for_review6,
                     interactive_rebuttal_display, interactive_review_count,
                     interactive_rebuttal_state, interactive_legend_html,
                     processing_thread_state]
        ).success(
            fn=process_interactive_reviews_fast,
            inputs=_interactive_inputs,
            outputs=_interactive_outputs
        ).success(
            fn=lambda: (
                gr.update(visible=False, value=""),
                gr.update(choices=["No Highlighting", "Polarity", "Topic", "Agreement ⏳"], interactive=True),
            ),
            inputs=[],
            outputs=[polarity_progress_html, focus_radio]
        ).success(
            fn=lambda: gr.update(interactive=True),
            inputs=[],
            outputs=[submit_button]
        ).success(
            fn=compute_rsa_in_background,
            inputs=[rsa_computation_state, focus_radio],
            outputs=_rsa_outputs
        )

        # Top bar: Back to input
        back_to_input_btn.click(
            fn=lambda: (
                gr.update(visible=True),    # show input
                gr.update(visible=False),   # hide results
                gr.update(visible=False),   # hide "back to input"
                gr.update(visible=False),   # hide paper title
                gr.update(visible=True),    # show "view results"
            ),
            inputs=[],
            outputs=[input_section, results_section, back_to_input_btn, paper_title_html, view_results_btn]
        )

        # Top bar: View Results (toggle back without re-processing)
        view_results_btn.click(
            fn=lambda: (
                gr.update(visible=False),   # hide input
                gr.update(visible=True),    # show results
                gr.update(visible=True),    # show "back to input"
                gr.update(visible=True),    # show paper title
                gr.update(visible=False),   # hide "view results"
            ),
            inputs=[],
            outputs=[input_section, results_section, back_to_input_btn, paper_title_html, view_results_btn]
        )

        # Clear button
        clear_button.click(
            fn=lambda: (
                "", "", "", "", "", "",                          # clear all textboxes
                "",                                              # clear paste_rebuttal
                3,                                               # reset count
                "", "",                                          # clear most common/divergent
                *([""] * 6), *([""] * 6), *([""] * 6), *([""] * 6),   # clear all output panels (none, agree, polar, topic)
                {},                                              # reset rsa_computation_state
            ),
            inputs=[],
            outputs=[
                review1_textbox, review2_textbox, review3_textbox, review4_textbox, review5_textbox, review6_textbox,
                paste_rebuttal,
                interactive_review_count,
                most_common, most_divergent,
                none_text1, none_text2, none_text3, none_text4, none_text5, none_text6,
                agreement_text1, agreement_text2, agreement_text3, agreement_text4, agreement_text5, agreement_text6,
                polarity_text1, polarity_text2, polarity_text3, polarity_text4, polarity_text5, polarity_text6,
                topic_text1, topic_text2, topic_text3, topic_text4, topic_text5, topic_text6,
                rsa_computation_state,
            ]
        ).then(
            fn=lambda: (
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),  # review4-6
                gr.update(visible=False, value=""),  # rebuttal toggle
                gr.update(visible=False, value=""), gr.update(visible=False, value=""), gr.update(visible=False, value=""),  # per-review rebuttals 1-3
                gr.update(visible=False, value=""), gr.update(visible=False, value=""), gr.update(visible=False, value=""),  # per-review rebuttals 4-6
                gr.update(visible=False, value=""),  # consolidated rebuttal
                gr.update(visible=False, value=""),  # paper_title_html
                gr.update(visible=False, value=""),  # polarity_progress_html
                gr.update(visible=False, value=""),  # agreement_progress_html
            ),
            inputs=[],
            outputs=[review4_textbox, review5_textbox, review6_textbox,
                     interactive_rebuttal_toggle,
                     rebuttal_for_review1, rebuttal_for_review2, rebuttal_for_review3,
                     rebuttal_for_review4, rebuttal_for_review5, rebuttal_for_review6,
                     interactive_rebuttal_display, paper_title_html,
                     polarity_progress_html, agreement_progress_html]
        )

        # Color legend HTML snippets for polarity/topic modes
        _POLARITY_LEGEND = (
            '<div style="display:flex;gap:12px;align-items:center;padding:8px 0;font-size:0.8em;">'
            '<span style="background:#d4fcd6;padding:2px 8px;border-radius:4px;">Positive</span>'
            '<span style="background:#fcd6d6;padding:2px 8px;border-radius:4px;">Negative</span>'
            '<span style="color:#9ca3af;">Neutral (no highlight)</span>'
            '</div>'
        )
        _TOPIC_LEGEND = (
            '<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center;padding:8px 0;">'
            + " ".join(
                f'<span style="background:{color};padding:2px 8px;border-radius:4px;'
                f'font-size:0.8em;margin-right:4px;">{_html.escape(name)}</span>'
                for name, color in _TOPIC_HTML_COLORS.items()
            )
            + '</div>'
        )

        # Toggle display mode (No Highlighting / Polarity / Topic / Agreement[Processing])
        def toggle_display_mode(focus, active_count):
            # Strip ⏳ loading suffix — treat "Polarity ⏳" as "Polarity", etc.
            effective_focus = focus.split(" ⏳")[0] if " ⏳" in focus else focus
            # Legacy compat
            if effective_focus == "Agreement (Processing)":
                effective_focus = "Agreement"

            updates = []
            for mode in ["No Highlighting", "Polarity", "Topic", "Agreement"]:
                for i in range(MAX_INTERACTIVE_REVIEWS):
                    updates.append(gr.update(visible=(mode == effective_focus and i < active_count)))

            # Most common shows in Agreement mode; most_divergent is now per-review (always hidden here)
            show_opinions = effective_focus == "Agreement" and focus != "Agreement (Processing)"
            updates.append(gr.update(visible=False))  # most_divergent (per-review now, always hidden)
            updates.append(gr.update(visible=show_opinions))  # most_common

            # Color legend
            if effective_focus == "Polarity":
                updates.append(gr.update(visible=True, value=_POLARITY_LEGEND))
            elif effective_focus == "Topic":
                updates.append(gr.update(visible=True, value=_TOPIC_LEGEND))
            else:
                updates.append(gr.update(visible=False, value=""))

            return tuple(updates)

        focus_radio.change(
            fn=toggle_display_mode,
            inputs=[focus_radio, interactive_review_count],
            outputs=[
                none_text1, none_text2, none_text3, none_text4, none_text5, none_text6,
                polarity_text1, polarity_text2, polarity_text3, polarity_text4, polarity_text5, polarity_text6,
                topic_text1, topic_text2, topic_text3, topic_text4, topic_text5, topic_text6,
                agreement_text1, agreement_text2, agreement_text3, agreement_text4, agreement_text5, agreement_text6,
                most_divergent, most_common,
                interactive_legend_html,
            ]
        )

        # Add Review button: show next hidden textbox (reviews 4-6)
        def add_review(count):
            new_count = min(count + 1, MAX_INTERACTIVE_REVIEWS)
            vis = [gr.update(visible=(i + 4 <= new_count)) for i in range(MAX_INTERACTIVE_REVIEWS - 3)]
            return (new_count, *vis)

        add_review_btn.click(
            fn=add_review,
            inputs=[interactive_review_count],
            outputs=[interactive_review_count, review4_textbox, review5_textbox, review6_textbox]
        )

    # Populate pre-processed tab with initial data on page load
    demo.load(
        fn=lambda s: update_review_display(s, "No Highlighting"),
        inputs=[state],
        outputs=_review_outputs,
    )

# Pre-load interactive processor models at startup so first request isn't slow
get_interactive_processor()

demo.launch(share=False)