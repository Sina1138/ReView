import sys, os.path
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import hashlib
import json
import math
import torch
import gradio as gr
import pandas as pd
import ast
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

BASE_DIR = Path(__file__).resolve().parent.parent

# Controls how aggressively agreement colors are amplified.
# Lower = more vivid colors (0.2 = very strong, 1.0 = no amplification).
# Asymmetric: unique/red (positive) is amplified less than common/blue (negative)
# to avoid overwhelming red when most sentences are unique.
AGREEMENT_AMP_UNIQUE = 0.95  # exponent for positive scores (red = unique)
AGREEMENT_AMP_COMMON = 0.65  # exponent for negative scores (blue = common)

import html as _html

def _make_sentence_id(sentence: str) -> str:
    """Deterministic DOM ID for a sentence, used by click-to-scroll."""
    return "sent_" + hashlib.md5(sentence.encode("utf-8")).hexdigest()[:12]



def _get_context(sentence: str, sentence_lists: list):
    """Return (context_before, context_after) strings for the first review containing sentence."""
    for sl in sentence_lists:
        if sentence in sl:
            idx = sl.index(sentence)
            before = _html.escape(sl[idx - 1]) if idx > 0 else ""
            after = _html.escape(sl[idx + 1]) if idx < len(sl) - 1 else ""
            return before, after
    return "", ""


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
        source_reviews = [r_idx + 1 for r_idx, sl in enumerate(sentence_lists) if sent in sl]
        source_badge = " ".join(
            f'<span style="background:#f3f4f6;color:#374151;padding:2px 6px;'
            f'border-radius:4px;font-size:0.72em;font-weight:600;">R{n}</span>'
            for n in source_reviews
        )

        # --- L_t(d|s) distribution bars ---
        dist_html = ""
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
            dist_html = (
                f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:3px;margin-bottom:3px;">'
                f'{source_badge}'
                f'<span style="color:#d1d5db;font-size:0.75em;">→</span> '
                + "".join(bar_parts)
                + "</div>"
            )
        else:
            dist_html = f'<div style="display:flex;gap:4px;margin-bottom:3px;">{source_badge}</div>'

        # Click-to-scroll via inline JS
        onclick = (
            f"(function(){{var el=document.getElementById('{sent_id}');"
            f"if(el){{el.scrollIntoView({{behavior:'smooth',block:'center'}});"
            f"el.style.outline='3px solid #3b82f6';"
            f"setTimeout(function(){{el.style.outline='';}},2500);}}}})();"
        )

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

    import numpy as np
    num_reviews = len(speaker)
    if num_reviews == 0:
        return {}

    median_u = float(np.median(list(uniqueness.values())))
    review_labels = [f"R{i+1}" for i in range(num_reviews)]

    # Minimum speaker score to suppress generic filler.
    k = max(sum(len(v) for v in speaker.values()) // max(len(speaker), 1), 1)
    min_speaker_score = 2.0 / k

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

            onclick = (
                f"(function(){{var el=document.getElementById('{sent_id}');"
                f"if(el){{el.scrollIntoView({{behavior:'smooth',block:'center'}});"
                f"el.style.outline='3px solid #ef4444';"
                f"setTimeout(function(){{el.style.outline='';}},2500);}}}})();"
            )

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

    parts = [f'<div style="font-weight:600;font-size:0.85em;color:#374151;margin-bottom:4px;">{_html.escape(label)}</div>']
    parts.append(legend_html)
    parts.append('<div style="line-height:1.8;font-size:0.95em;">')

    # Compute informativeness threshold: 2 / K (twice uniform baseline)
    k = max(len(uniqueness), 1)
    info_threshold = 2.0 / k

    for sent in sentences:
        sent_id = _make_sentence_id(sent)
        score = uniqueness.get(sent)

        if score is None or abs(score) < HIGHLIGHT_THRESHOLD:
            # No highlight
            parts.append(f'<span id="{sent_id}">{_html.escape(sent)} </span>')
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
                if max_prob > 0.70:
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

    parts.append("</div>")
    return "".join(parts)


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


def format_rebuttal_for_review(rebuttal: str, review_num: int) -> str:
    """Format rebuttals that reply to a specific review number."""
    if not rebuttal or not rebuttal.strip():
        return ""

    CARD_STYLE = "margin-top:8px;margin-bottom:12px;border-radius:6px;overflow:hidden;border:1px solid #fde68a;background:#fffef5;"

    try:
        items = json.loads(rebuttal)
        if not items:
            return ""

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
            f'<details style="{CARD_STYLE}">'
            f'<summary style="padding:10px 14px;cursor:pointer;font-size:0.75em;color:#92400e;'
            f'font-weight:600;list-style:none;display:flex;align-items:center;gap:6px;">'
            f'<span style="transition:transform 0.2s;">▶</span> Author Response</summary>'
            + "".join(response_parts)
            + "</details>"
        )

    except (json.JSONDecodeError, TypeError, AttributeError):
        # Plain text - show under first review only
        if review_num == 1:
            text = rebuttal.strip()
            return (
                f'<details style="{CARD_STYLE}">'
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

    CARD_STYLE = "margin-top:16px;border-radius:8px;overflow:hidden;border:1px solid #fde68a;"
    HEADER_STYLE = "background:#fffbeb;padding:10px 16px;border-bottom:1px solid #fde68a;display:flex;align-items:center;gap:8px;"
    TITLE_STYLE = "font-weight:600;color:#92400e;"

    try:
        items = json.loads(rebuttal)
        if not items:
            return ""

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
            f'<details style="{CARD_STYLE}">'
            f'<summary style="{HEADER_STYLE}cursor:pointer;list-style:none;">'
            f'<span style="font-size:1.1em;">💬</span>'
            f'<span style="{TITLE_STYLE}">General Author Response</span>'
            f'<span style="margin-left:auto;font-size:0.8em;color:#78716c;">{count_label}</span>'
            f'</summary>'
            + "".join(response_parts) +
            '</details>'
        )
    except (json.JSONDecodeError, TypeError, AttributeError):
        # Plain text - treat as general response
        text = rebuttal.strip()
        return (
            f'<details style="{CARD_STYLE}">'
            f'<summary style="{HEADER_STYLE}cursor:pointer;list-style:none;">'
            f'<span style="font-size:1.1em;">💬</span>'
            f'<span style="{TITLE_STYLE}">General Author Response</span></summary>'
            f'<div style="padding:14px 16px;background:white;">'
            f'<div style="white-space:pre-wrap;color:#1f2937;font-size:0.9em;line-height:1.6;">{text}</div>'
            f'</div></details>'
        )


def process_interactive_reviews_fast(text1: str, text2: str, text3: str, text4: str, text5: str, text6: str, focus: str, progress=gr.Progress()) -> Tuple:
    """
    Fast processing: Polarity + Topic only (~3-5 sec on CPU).
    RSA (agreement) runs in background.
    Returns immediately with placeholder agreement sections that update when ready.
    """
    from dependencies.Glimpse_tokenizer import glimpse_tokenizer

    all_texts = [text1, text2, text3, text4, text5, text6]
    active_texts = [t for t in all_texts if t and t.strip()]

    if len(active_texts) < 2:
        raise ValueError("Please enter at least two reviews")

    # Step 1: Load models
    progress(0.0, desc="Loading models...")
    processor = get_interactive_processor()

    # Step 2: Tokenize
    progress(0.10, desc="Tokenizing reviews...")
    sentence_lists = [[s for s in glimpse_tokenizer(t) if s.strip()] for t in active_texts]
    sentence_lists = [sl for sl in sentence_lists if sl]

    if len(sentence_lists) < 2:
        raise ValueError("At least two reviews must have valid sentences")

    all_sentences = filter_and_clean_sentences(
        list(set(s for sl in sentence_lists for s in sl))
    )

    # Step 3-4: Polarity + Topic (parallelize both models)
    progress(0.30, desc="Predicting polarity and topics (parallel)...")
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as executor:
        polarity_future = executor.submit(processor.predict_polarity, all_sentences)
        topic_future = executor.submit(processor.predict_topic, all_sentences)
        polarity_map = polarity_future.result()
        topic_map = topic_future.result()

    # Step 5: Format results (no consensuality yet - it's computing in background)
    progress(0.90, desc="Formatting results...")

    consensuality_map = {}  # Empty - will be filled by async RSA

    fmt = processor.format_highlighted_output
    # Build per-review outputs (pad inactive reviews with empty/hidden)
    none_out, agree_out, polar_out, topic_out = [], [], [], []
    for i in range(MAX_INTERACTIVE_REVIEWS):
        if i < len(sentence_lists):
            # No Highlighting mode: show original text without any highlighting
            none_out.append(gr.update(visible=True, value=fmt(sentence_lists[i], {}, "none")))
            # Agreement section shows empty placeholder (RSA fills in async)
            agree_out.append(gr.update(visible=False, value=""))
            polar_out.append(gr.update(visible=False, value=fmt(sentence_lists[i], polarity_map, "polarity")))
            topic_out.append(gr.update(visible=False, value=fmt(sentence_lists[i], topic_map, "topic")))
        else:
            none_out.append(gr.update(visible=False, value=None))
            agree_out.append(gr.update(visible=False, value=""))
            polar_out.append(gr.update(visible=False, value=None))
            topic_out.append(gr.update(visible=False, value=None))

    progress(1.0, desc="Done! Computing agreement in background...")

    # Store sentence lists and texts in state for async RSA
    rsa_state = {
        "sentence_lists": sentence_lists,
        "active_texts": active_texts,
        "polarity_map": polarity_map,
        "topic_map": topic_map,
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


def compute_rsa_in_background(rsa_state: Dict, current_focus: str, progress=gr.Progress()) -> Tuple:
    """
    Compute RSA (agreement) in background.
    Returns updates for agreement HTML sections + summary cards.
    """
    if not rsa_state or not rsa_state.get("sentence_lists"):
        return tuple([gr.update(visible=False, value="") ] * (MAX_INTERACTIVE_REVIEWS + 2))

    progress(0.0, desc="Computing agreement (this may take a minute)...")

    processor = get_interactive_processor()
    sentence_lists = rsa_state["sentence_lists"]
    active_texts = rsa_state["active_texts"]

    try:
        # Full RSA computation — exposes listener, speaker, best_rsa alongside uniqueness
        progress(0.50, desc="Running RSA reranking...")
        rsa_result = processor.predict_rsa_full(*active_texts)

        uniqueness = rsa_result.get("uniqueness", {}) if rsa_result else {}
        listener = rsa_result.get("listener", {}) if rsa_result else {}
        speaker = rsa_result.get("speaker", {}) if rsa_result else {}

        # --- Most Common hub ---
        if uniqueness:
            import pandas as _pd
            scores_series = _pd.Series(uniqueness)

            # Step 1: Take the 15 most common sentences by raw uniqueness score.
            # The RSA math is authoritative here — trust the ranking it produces.
            n_seed = min(15, len(scores_series))
            seed = scores_series.nsmallest(n_seed).index.tolist()

            # Step 2: Re-rank by listener ENTROPY — sentences with balanced
            # distributions across reviewers (high entropy) are genuinely "common".
            # This naturally surfaces R1 63% / R2 33% / R3 4% above R1 86% / R2 7% / R3 7%
            # because entropy rewards spread, not concentration.
            # Informativeness (max speaker) is wrong here — it rewards single-reviewer
            # dominance, which is the opposite of "common".
            if listener:
                def _listener_entropy(sent):
                    dist = listener.get(sent, {})
                    ent = 0.0
                    for p in dist.values():
                        if p > 0:
                            ent -= p * math.log(p)
                    return ent
                seed.sort(key=_listener_entropy, reverse=True)

            # Step 3: Show top 5 (highest entropy = most balanced = best common).
            top_common = seed[:5]
            most_common_text = format_summary_cards(
                top_common, uniqueness, sentence_lists, "common",
                listener=listener, speaker=speaker,
            )
            # --- Most Divergent hub (per-review) ---
            divergent_per_review = format_divergent_cards(
                uniqueness, sentence_lists, listener, speaker,
            )
        else:
            most_common_text = ""
            divergent_per_review = {}

        progress(0.90, desc="Formatting agreement results...")

        show_agreement = current_focus in ("Agreement", "Agreement (Processing)")
        num_reviews = len(active_texts)
        agree_out = []
        for i in range(MAX_INTERACTIVE_REVIEWS):
            if i < len(sentence_lists):
                html_val = render_agreement_html(
                    sentence_lists[i], uniqueness, listener, speaker,
                    num_reviews=num_reviews,
                    label=f"Agreement in Review {i + 1}",
                )
                # Append this review's divergent cards below the agreement text
                if i in divergent_per_review:
                    html_val += divergent_per_review[i]
                agree_out.append(gr.update(visible=show_agreement, value=html_val))
            else:
                agree_out.append(gr.update(visible=False, value=""))

        progress(1.0, desc="Agreement computation complete!")

        # most_divergent gets empty string — divergent cards are now per-review
        return (*agree_out, most_common_text, "")

    except Exception as e:
        print(f"[RSA ERROR] {type(e).__name__}: {str(e)}")
        error_msg = f"❌ Agreement computation failed: {str(e)[:100]}"
        return tuple([gr.update(visible=False, value="")] * MAX_INTERACTIVE_REVIEWS + [error_msg, ""])




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
"""

with gr.Blocks(title="ReView", css=CUSTOM_CSS) as demo:
    # gr.Markdown("# ReView Interface")
    
    # TODO: Uncomment this for home/description tab once finished with testing.
    # with gr.Tab("Introduction"):
    #     gr.Markdown(glimpse_description)
        
    # -----------------------------------
    # Pre-processed Tab
    # -----------------------------------
    with gr.Tab("Pre-processed Reviews"):
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
            import numpy as _np
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
                    arr = _np.array(all_raw_scores)
                    _kl_median = float(_np.median(arr))
                    q25, q75 = float(_np.percentile(arr, 25)), float(_np.percentile(arr, 75))
                    _kl_iqr = q75 - q25

            # Build per-review sentence lists (needed for agreement HTML + summary cards)
            review_sentence_lists = []
            review_items_cache = []  # Cache (review_item, rebuttal_html) per review
            for idx in range(number_of_displayed_reviews):
                review_data = current_review[idx]
                rebuttal_html = ""
                if isinstance(review_data, dict) and "sentences" in review_data:
                    review_item = list(review_data["sentences"].items())
                    rebuttal = review_data.get("rebuttal", "")
                    if rebuttal and rebuttal.strip():
                        rebuttal_html = (
                            '<details style="margin-top:8px;margin-bottom:12px;border-radius:6px;overflow:hidden;border:1px solid #fde68a;background:#fffef5;">'
                            '<summary style="padding:10px 14px;cursor:pointer;font-size:0.75em;color:#92400e;'
                            'font-weight:600;list-style:none;display:flex;align-items:center;gap:6px;">'
                            '<span style="transition:transform 0.2s;">▶</span> Author Response</summary>'
                            '<div style="padding:10px 14px;">'
                            f'<div style="white-space:pre-wrap;color:#1f2937;font-size:0.85em;line-height:1.5;">{rebuttal}</div>'
                            '</div></details>'
                        )
                else:
                    review_item = list(review_data.items())
                review_sentence_lists.append([s for s, _ in review_item])
                review_items_cache.append((review_item, rebuttal_html))

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

            for i in range(10):
                if i < number_of_displayed_reviews:
                    review_item, rebuttal_html = review_items_cache[i]

                    if show_polarity:
                        highlighted = []
                        for sentence, metadata in review_item:
                            polarity = metadata.get("polarity", None)
                            if polarity == 2:
                                label = "➕"
                            elif polarity == 0:
                                label = "➖"
                            else:
                                label = None
                            highlighted.append((sentence, label))
                    elif show_consensuality:
                        highlighted = [(sentence, None) for sentence, _ in review_item]
                    elif show_topic:
                        highlighted = []
                        for sentence, metadata in review_item:
                            topic = metadata.get("topic", None)
                            if topic != "NONE":
                                highlighted.append((sentence, topic))
                            else:
                                highlighted.append((sentence, None))
                    else:
                        highlighted = [
                            (sentence, None)
                            for sentence, _ in review_item
                        ]

                    # HighlightedText: visible for all modes except agreement
                    review_updates.append(
                        gr.update(
                            visible=not show_consensuality,
                            value=highlighted,
                            color_map=color_map,
                            show_legend=legend,
                            key=f"updated_{score_type}_{i}"
                        )
                    )

                    # Agreement HTML: visible only in agreement mode
                    if show_consensuality:
                        sentences_for_review = [s for s, _ in review_item]
                        agreement_html = render_agreement_html(
                            sentences_for_review, consensuality_dict,
                            listener=prep_listener, speaker=prep_speaker,
                            num_reviews=number_of_displayed_reviews,
                            label=f"Agreement in Review {i + 1}",
                        )
                        # Append per-review divergent cards (if RSA data available)
                        if i in divergent_per_review:
                            agreement_html += divergent_per_review[i]
                        agreement_updates.append(gr.update(visible=True, value=agreement_html))
                    else:
                        agreement_updates.append(gr.update(visible=False, value=""))

                    rebuttal_updates.append(gr.update(visible=bool(rebuttal_html), value=rebuttal_html))
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

            # Set most common opinions (as HTML cards with context)
            # Uses entropy-based ranking when listener data is available (like interactive tab)
            if show_consensuality and consensuality_dict:
                scores = pd.Series(consensuality_dict)

                if prep_listener:
                    # Entropy-based ranking: same logic as interactive tab
                    n_seed = min(15, len(scores))
                    seed = scores.nsmallest(n_seed).index.tolist()

                    def _listener_entropy(sent):
                        dist = prep_listener.get(sent, {})
                        ent = 0.0
                        for p in dist.values():
                            if p > 0:
                                ent -= p * math.log(p)
                        return ent
                    seed.sort(key=_listener_entropy, reverse=True)
                    most_common = seed[:5]
                else:
                    most_common = scores.sort_values(ascending=True).head(5).index.tolist()

                most_common_html = format_summary_cards(
                    most_common, consensuality_dict, review_sentence_lists, "common",
                    listener=prep_listener, speaker=prep_speaker,
                )

                most_common_visibility = gr.update(visible=True, value=most_common_html)
                most_unique_visibility = gr.update(visible=False, value="")
            else:
                most_common_visibility = gr.update(visible=False, value="")
                most_unique_visibility = gr.update(visible=False, value="")
                
            # update topic color map
            if show_topic:
                topic_color_map_visibility = gr.update(
                    visible=True,
                    color_map=topic_color_map,
                    value=[
                        ("", "Substance"),
                        ("", "Clarity"),
                        ("", "Soundness/Correctness"),
                        ("", "Originality"),
                        ("", "Motivation/Impact"),
                        ("", "Meaningful Comparison"),
                        ("", "Replicability"),
                    ]
                )
            else:
                topic_color_map_visibility = gr.update(visible=False, value=[])

            return (
                new_review_id,
                *review_updates,
                *agreement_updates,  # 10 agreement HTML sections
                most_common_visibility,
                most_unique_visibility,
                topic_color_map_visibility,
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
        
        # Add a new textbox for topic labels and colors
        topic_text_box = gr.HighlightedText(
            label="Topic Labels (Color-Coded)",
            visible=False,
            value=[],
            show_legend=True,
        )
        
        gr.Markdown("### 📝 Reviews", elem_classes=["review-section-header"])
        review1 = gr.HighlightedText(show_legend=False, label="📝 Review 1", visible=number_of_displayed_reviews >= 1, key="initial_review1")
        prep_agreement1 = gr.HTML(visible=False, value="")
        prep_rebuttal1 = gr.HTML(visible=False, value="")
        review2 = gr.HighlightedText(show_legend=False, label="📝 Review 2", visible=number_of_displayed_reviews >= 2, key="initial_review2")
        prep_agreement2 = gr.HTML(visible=False, value="")
        prep_rebuttal2 = gr.HTML(visible=False, value="")
        review3 = gr.HighlightedText(show_legend=False, label="📝 Review 3", visible=number_of_displayed_reviews >= 3, key="initial_review3")
        prep_agreement3 = gr.HTML(visible=False, value="")
        prep_rebuttal3 = gr.HTML(visible=False, value="")
        review4 = gr.HighlightedText(show_legend=False, label="📝 Review 4", visible=number_of_displayed_reviews >= 4, key="initial_review4")
        prep_agreement4 = gr.HTML(visible=False, value="")
        prep_rebuttal4 = gr.HTML(visible=False, value="")
        review5 = gr.HighlightedText(show_legend=False, label="📝 Review 5", visible=number_of_displayed_reviews >= 5, key="initial_review5")
        prep_agreement5 = gr.HTML(visible=False, value="")
        prep_rebuttal5 = gr.HTML(visible=False, value="")
        review6 = gr.HighlightedText(show_legend=False, label="📝 Review 6", visible=number_of_displayed_reviews >= 6, key="initial_review6")
        prep_agreement6 = gr.HTML(visible=False, value="")
        prep_rebuttal6 = gr.HTML(visible=False, value="")
        review7 = gr.HighlightedText(show_legend=False, label="📝 Review 7", visible=number_of_displayed_reviews >= 7, key="initial_review7")
        prep_agreement7 = gr.HTML(visible=False, value="")
        prep_rebuttal7 = gr.HTML(visible=False, value="")
        review8 = gr.HighlightedText(show_legend=False, label="📝 Review 8", visible=number_of_displayed_reviews >= 8, key="initial_review8")
        prep_agreement8 = gr.HTML(visible=False, value="")
        prep_rebuttal8 = gr.HTML(visible=False, value="")
        review9 = gr.HighlightedText(show_legend=False, label="📝 Review 9", visible=number_of_displayed_reviews >= 9, key="initial_review9")
        prep_agreement9 = gr.HTML(visible=False, value="")
        prep_rebuttal9 = gr.HTML(visible=False, value="")
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
        _review_outputs = [review_id, review1, review2, review3, review4, review5, review6, review7, review8, review9, review10, prep_agreement1, prep_agreement2, prep_agreement3, prep_agreement4, prep_agreement5, prep_agreement6, prep_agreement7, prep_agreement8, prep_agreement9, prep_agreement10, most_common_sentences, most_unique_sentences, topic_text_box, prep_rebuttal1, prep_rebuttal2, prep_rebuttal3, prep_rebuttal4, prep_rebuttal5, prep_rebuttal6, prep_rebuttal7, prep_rebuttal8, prep_rebuttal9, prep_rebuttal10, prep_general_rebuttal, state]
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
        with gr.Column(visible=False) as results_section:
            with gr.Row():
                focus_radio = gr.Radio(
                    choices=["No Highlighting", "Polarity", "Topic", "Agreement (Processing)"],
                    value="No Highlighting",
                    label="Display Mode:",
                    interactive=True
                )

            with gr.Row():
                most_divergent = gr.HTML(
                    visible=False, value="", label="Most Divergent Opinions",
                )
                most_common = gr.HTML(
                    visible=False, value="", label="Most Common Opinions",
                )

            # Review 1 (all display modes + rebuttal)
            none_text1 = gr.HighlightedText(show_legend=False, label="📝 Review 1", visible=True, value=None)
            agreement_text1 = gr.HTML(visible=False, value="")
            polarity_text1 = gr.HighlightedText(show_legend=True, label="Polarity in 📝 Review 1", visible=False, value=None, color_map={"➕": "#d4fcd6", "➖": "#fcd6d6"})
            topic_text1 = gr.HighlightedText(show_legend=False, label="Topic in 📝 Review 1", visible=False, value=None, color_map=topic_color_map)
            rebuttal_for_review1 = gr.HTML(visible=False, value="")

            # Review 2 (all display modes + rebuttal)
            none_text2 = gr.HighlightedText(show_legend=False, label="📝 Review 2", visible=False, value=None)
            agreement_text2 = gr.HTML(visible=False, value="")
            polarity_text2 = gr.HighlightedText(show_legend=True, label="Polarity in 📝 Review 2", visible=False, value=None, color_map={"➕": "#d4fcd6", "➖": "#fcd6d6"})
            topic_text2 = gr.HighlightedText(show_legend=False, label="Topic in 📝 Review 2", visible=False, value=None, color_map=topic_color_map)
            rebuttal_for_review2 = gr.HTML(visible=False, value="")

            # Review 3 (all display modes + rebuttal)
            none_text3 = gr.HighlightedText(show_legend=False, label="📝 Review 3", visible=False, value=None)
            agreement_text3 = gr.HTML(visible=False, value="")
            polarity_text3 = gr.HighlightedText(show_legend=True, label="Polarity in 📝 Review 3", visible=False, value=None, color_map={"➕": "#d4fcd6", "➖": "#fcd6d6"})
            topic_text3 = gr.HighlightedText(show_legend=False, label="Topic in 📝 Review 3", visible=False, value=None, color_map=topic_color_map)
            rebuttal_for_review3 = gr.HTML(visible=False, value="")

            # Review 4 (all display modes + rebuttal)
            none_text4 = gr.HighlightedText(show_legend=False, label="📝 Review 4", visible=False, value=None)
            agreement_text4 = gr.HTML(visible=False, value="")
            polarity_text4 = gr.HighlightedText(show_legend=True, label="Polarity in 📝 Review 4", visible=False, value=None, color_map={"➕": "#d4fcd6", "➖": "#fcd6d6"})
            topic_text4 = gr.HighlightedText(show_legend=False, label="Topic in 📝 Review 4", visible=False, value=None, color_map=topic_color_map)
            rebuttal_for_review4 = gr.HTML(visible=False, value="")

            # Review 5 (all display modes + rebuttal)
            none_text5 = gr.HighlightedText(show_legend=False, label="📝 Review 5", visible=False, value=None)
            agreement_text5 = gr.HTML(visible=False, value="")
            polarity_text5 = gr.HighlightedText(show_legend=True, label="Polarity in 📝 Review 5", visible=False, value=None, color_map={"➕": "#d4fcd6", "➖": "#fcd6d6"})
            topic_text5 = gr.HighlightedText(show_legend=False, label="Topic in 📝 Review 5", visible=False, value=None, color_map=topic_color_map)
            rebuttal_for_review5 = gr.HTML(visible=False, value="")

            # Review 6 (all display modes + rebuttal)
            none_text6 = gr.HighlightedText(show_legend=False, label="📝 Review 6", visible=False, value=None)
            agreement_text6 = gr.HTML(visible=False, value="")
            polarity_text6 = gr.HighlightedText(show_legend=True, label="Polarity in 📝 Review 6", visible=False, value=None, color_map={"➕": "#d4fcd6", "➖": "#fcd6d6"})
            topic_text6 = gr.HighlightedText(show_legend=False, label="Topic in 📝 Review 6", visible=False, value=None, color_map=topic_color_map)
            rebuttal_for_review6 = gr.HTML(visible=False, value="")

            # General rebuttal display (for rebuttals not tied to specific reviews)
            interactive_rebuttal_display = gr.HTML(visible=False, value="")

        # ---- CALLBACKS ----

        _interactive_inputs = [review1_textbox, review2_textbox, review3_textbox, review4_textbox, review5_textbox, review6_textbox, focus_radio]

        # State to hold RSA computation results for async updates
        rsa_computation_state = gr.State({})

        _interactive_outputs = [
            none_text1, none_text2, none_text3, none_text4, none_text5, none_text6,
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
        ]

        # Fetch OpenReview reviews → show timer → fast process → swap to results → async RSA
        def _validate_and_start_fetch(link):
            if not link or not link.strip():
                raise gr.Error("Please paste a valid OpenReview link before fetching.")
            return gr.update(value=FETCHING_HTML, visible=True), gr.update(interactive=False)

        def _show_results_with_rebuttal(rebuttal):
            # Generate per-review rebuttals
            per_review = []
            for i in range(1, 7):
                formatted = format_rebuttal_for_review(rebuttal or "", i)
                per_review.append(gr.update(visible=bool(formatted), value=formatted))

            # Generate general rebuttal section (only for rebuttals not tied to specific reviews)
            general_formatted = format_general_rebuttals(rebuttal or "")
            has_general = bool(general_formatted)

            return (
                gr.update(visible=False),   # input_section
                gr.update(visible=True),    # results_section
                gr.update(value="✅ Polarity & Topic ready! Computing agreement in background...", visible=True),
                gr.update(visible=True),    # back_to_input_btn
                gr.update(visible=False),   # view_results_btn
                gr.update(choices=["No Highlighting", "Polarity", "Topic", "Agreement (Processing)"]),
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
            fn=lambda r4, r5, r6, title: (
                gr.update(visible=bool(title.strip())),
                gr.update(value=PROCESSING_TIMER_HTML, visible=True),
                gr.update(visible=bool(r4.strip())),
                gr.update(visible=bool(r5.strip())),
                gr.update(visible=bool(r6.strip())),
            ),
            inputs=[review4_textbox, review5_textbox, review6_textbox, openreview_title],
            outputs=[openreview_title, status_html, review4_textbox, review5_textbox, review6_textbox]
        ).success(
            fn=process_interactive_reviews_fast,
            inputs=_interactive_inputs,
            outputs=_interactive_outputs
        ).success(
            fn=_show_results_with_rebuttal,
            inputs=[openreview_rebuttal],
            outputs=[input_section, results_section, status_html, back_to_input_btn, view_results_btn, focus_radio,
                     rebuttal_for_review1, rebuttal_for_review2, rebuttal_for_review3,
                     rebuttal_for_review4, rebuttal_for_review5, rebuttal_for_review6,
                     interactive_rebuttal_display]
        ).success(
            fn=lambda: gr.update(interactive=True),
            inputs=[],
            outputs=[fetch_reviews_button]
        ).success(
            fn=compute_rsa_in_background,
            inputs=[rsa_computation_state, focus_radio],
            outputs=_rsa_outputs
        ).success(
            fn=lambda: (
                gr.update(value="✅ Agreement computation complete!", visible=True),
                gr.update(choices=["No Highlighting", "Polarity", "Topic", "Agreement"], interactive=True),
            ),
            inputs=[],
            outputs=[status_html, focus_radio]
        )

        # Process (Paste Reviews): show timer → fast scoring → swap to results → async RSA in background
        submit_button.click(
            fn=lambda: (
                gr.update(value=PROCESSING_TIMER_HTML, visible=True),
                gr.update(interactive=False),
            ),
            inputs=[],
            outputs=[status_html, submit_button]
        ).success(
            fn=process_interactive_reviews_fast,
            inputs=_interactive_inputs,
            outputs=_interactive_outputs
        ).success(
            fn=_show_results_with_rebuttal,
            inputs=[paste_rebuttal],
            outputs=[input_section, results_section, status_html, back_to_input_btn, view_results_btn, focus_radio,
                     rebuttal_for_review1, rebuttal_for_review2, rebuttal_for_review3,
                     rebuttal_for_review4, rebuttal_for_review5, rebuttal_for_review6,
                     interactive_rebuttal_display]
        ).success(
            fn=lambda: gr.update(interactive=True),
            inputs=[],
            outputs=[submit_button]
        ).success(
            fn=compute_rsa_in_background,
            inputs=[rsa_computation_state, focus_radio],
            outputs=_rsa_outputs
        ).success(
            fn=lambda: (
                gr.update(value="✅ Agreement computation complete!", visible=True),
                gr.update(choices=["No Highlighting", "Polarity", "Topic", "Agreement"], interactive=True),
            ),
            inputs=[],
            outputs=[status_html, focus_radio]
        )

        # Top bar: Back to input
        back_to_input_btn.click(
            fn=lambda: (
                gr.update(visible=True),                     # show input
                gr.update(visible=False),                    # hide results
                gr.update(visible=False),                    # hide "back to input"
                gr.update(visible=True),                     # show "view results"
            ),
            inputs=[],
            outputs=[input_section, results_section, back_to_input_btn, view_results_btn]
        )

        # Top bar: View Results (toggle back without re-processing)
        view_results_btn.click(
            fn=lambda: (
                gr.update(visible=False),                    # hide input
                gr.update(visible=True),                     # show results
                gr.update(visible=True),                     # show "back to input"
                gr.update(visible=False),                    # hide "view results"
            ),
            inputs=[],
            outputs=[input_section, results_section, back_to_input_btn, view_results_btn]
        )

        # Clear button
        clear_button.click(
            fn=lambda: (
                "", "", "", "", "", "",                          # clear all textboxes
                "",                                              # clear paste_rebuttal
                3,                                               # reset count
                "", "",                                          # clear most common/divergent
                *([None] * 6), *([""] * 6), *([None] * 6), *([None] * 6),   # clear all output panels (none, agree, polar, topic)
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
                gr.update(visible=False, value=""), gr.update(visible=False, value=""), gr.update(visible=False, value=""),  # per-review rebuttals 1-3
                gr.update(visible=False, value=""), gr.update(visible=False, value=""), gr.update(visible=False, value=""),  # per-review rebuttals 4-6
                gr.update(visible=False, value="")  # consolidated rebuttal
            ),
            inputs=[],
            outputs=[review4_textbox, review5_textbox, review6_textbox,
                     rebuttal_for_review1, rebuttal_for_review2, rebuttal_for_review3,
                     rebuttal_for_review4, rebuttal_for_review5, rebuttal_for_review6,
                     interactive_rebuttal_display]
        )

        # Toggle display mode (No Highlighting / Polarity / Topic / Agreement[Processing])
        def toggle_display_mode(focus, active_count):
            # Treat "Agreement (Processing)" as "Agreement" for section visibility
            effective_focus = "Agreement" if focus == "Agreement (Processing)" else focus

            updates = []
            for mode in ["No Highlighting", "Polarity", "Topic", "Agreement"]:
                for i in range(MAX_INTERACTIVE_REVIEWS):
                    updates.append(gr.update(visible=(mode == effective_focus and i < active_count)))

            # Most common shows in Agreement mode; most_divergent is now per-review (always hidden here)
            show_opinions = effective_focus == "Agreement" and focus != "Agreement (Processing)"
            updates.append(gr.update(visible=False))  # most_divergent (per-review now, always hidden)
            updates.append(gr.update(visible=show_opinions))  # most_common

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
        
demo.launch(share=False)