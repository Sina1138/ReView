import sys, os.path
from pathlib import Path
from typing import Tuple, Dict
import json

import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

BASE_DIR = Path(__file__).resolve().parent.parent

import gradio as gr
import pandas as pd
import ast
from tqdm import tqdm

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
from interface.interactive_processor import InteractiveReviewProcessor, is_section_header
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
                f'<div style="font-size:0.75em;color:#92400e;font-weight:600;margin-bottom:4px;">💬 Author Response:</div>'
                f'<div style="white-space:pre-wrap;color:#1f2937;font-size:0.85em;line-height:1.5;">{text}</div>'
                f'</div>'
            )

        if not response_parts:
            return ""

        return f'<div style="{CARD_STYLE}">{"".join(response_parts)}</div>'

    except (json.JSONDecodeError, TypeError, AttributeError):
        # Plain text - show under first review only
        if review_num == 1:
            text = rebuttal.strip()
            return (
                f'<div style="{CARD_STYLE}">'
                f'<div style="padding:10px 14px;">'
                f'<div style="font-size:0.75em;color:#92400e;font-weight:600;margin-bottom:4px;">💬 Author Response:</div>'
                f'<div style="white-space:pre-wrap;color:#1f2937;font-size:0.85em;line-height:1.5;">{text}</div>'
                f'</div></div>'
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
            f'<div style="{CARD_STYLE}">'
            f'<div style="{HEADER_STYLE}"><span style="font-size:1.1em;">💬</span>'
            f'<span style="{TITLE_STYLE}">General Author Response</span>'
            f'<span style="margin-left:auto;font-size:0.8em;color:#78716c;">{count_label}</span></div>'
            + "".join(response_parts) +
            '</div>'
        )
    except (json.JSONDecodeError, TypeError, AttributeError):
        # Plain text - treat as general response
        text = rebuttal.strip()
        return (
            f'<div style="{CARD_STYLE}">'
            f'<div style="{HEADER_STYLE}"><span style="font-size:1.1em;">💬</span>'
            f'<span style="{TITLE_STYLE}">General Author Response</span></div>'
            f'<div style="padding:14px 16px;background:white;">'
            f'<div style="white-space:pre-wrap;color:#1f2937;font-size:0.9em;line-height:1.6;">{text}</div>'
            f'</div></div>'
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

    all_sentences = [s for s in set(s for sl in sentence_lists for s in sl) if not is_section_header(s)]

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
            # Agreement section shows "Computing..." placeholder
            agree_out.append(gr.update(visible=False, value=[("⏳ Computing agreement...", None)]))
            polar_out.append(gr.update(visible=False, value=fmt(sentence_lists[i], polarity_map, "polarity")))
            topic_out.append(gr.update(visible=False, value=fmt(sentence_lists[i], topic_map, "topic")))
        else:
            none_out.append(gr.update(visible=False, value=None))
            agree_out.append(gr.update(visible=False, value=None))
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
    Returns updates for agreement sections only.
    """
    if not rsa_state or not rsa_state.get("sentence_lists"):
        return tuple([gr.update(visible=False, value=None)] * (MAX_INTERACTIVE_REVIEWS + 2))

    progress(0.0, desc="Computing agreement (this may take a minute)...")

    processor = get_interactive_processor()
    sentence_lists = rsa_state["sentence_lists"]
    active_texts = rsa_state["active_texts"]

    try:
        # Compute consensuality
        progress(0.50, desc="Running RSA reranking...")
        consensuality_map = processor.predict_consensuality(*active_texts)

        # Calculate most common and unique
        if consensuality_map:
            import pandas as _pd
            scores_series = _pd.Series(consensuality_map)
            most_common_text = "\n".join(scores_series.nlargest(3).index.tolist())
            most_unique_text = "\n".join(scores_series.nsmallest(3).index.tolist())
        else:
            most_common_text = ""
            most_unique_text = ""

        progress(0.90, desc="Formatting agreement results...")

        fmt = processor.format_highlighted_output
        show_agreement = current_focus in ("Agreement", "Agreement (Processing)")
        agree_out = []
        for i in range(MAX_INTERACTIVE_REVIEWS):
            if i < len(sentence_lists):
                agree_out.append(gr.update(visible=show_agreement, value=fmt(sentence_lists[i], consensuality_map, "consensuality")))
            else:
                agree_out.append(gr.update(visible=False, value=None))

        progress(1.0, desc="Agreement computation complete!")

        return (*agree_out, most_common_text, most_unique_text)

    except Exception as e:
        print(f"[RSA ERROR] {type(e).__name__}: {str(e)}")
        error_msg = f"❌ Agreement computation failed: {str(e)[:100]}"
        return tuple([gr.update(visible=False, value=None)] * MAX_INTERACTIVE_REVIEWS + [error_msg, ""])




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
"""

with gr.Blocks(title="ReView", css=CUSTOM_CSS) as demo:
    # gr.Markdown("# ReView Interface")
    
    with gr.Tab("Introduction"):
        gr.Markdown(glimpse_description)
        
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

            new_review_id = (
                f"### Submission Link:\n\n{review_ids[current_index]}<br>"
                f"(Showing {current_index + 1} of {len(state['review_ids'])} reviews)"
            )

            number_of_displayed_reviews = len(current_review)
            review_updates = []
            rebuttal_updates = []
            consensuality_dict = {}

            for i in range(10):
                if i < number_of_displayed_reviews:
                    # Handle new structure: current_review[i] can be dict with "sentences" and "rebuttal"
                    # OR old structure: just a dict of sentences
                    review_data = current_review[i]
                    rebuttal_html = ""

                    if isinstance(review_data, dict) and "sentences" in review_data:
                        review_item = list(review_data["sentences"].items())
                        rebuttal = review_data.get("rebuttal", "")
                        if rebuttal and rebuttal.strip():
                            # Format rebuttal as HTML card
                            rebuttal_html = (
                                '<div style="margin-top:8px;margin-bottom:12px;border-radius:6px;overflow:hidden;border:1px solid #fde68a;background:#fffef5;">'
                                '<div style="padding:10px 14px;">'
                                '<div style="font-size:0.75em;color:#92400e;font-weight:600;margin-bottom:4px;">💬 Author Response:</div>'
                                f'<div style="white-space:pre-wrap;color:#1f2937;font-size:0.85em;line-height:1.5;">{rebuttal}</div>'
                                '</div></div>'
                            )
                    else:
                        # Backward compatibility with old format
                        review_item = list(review_data.items())

                    if show_polarity:
                        highlighted = []
                        for sentence, metadata in review_item:
                            polarity = metadata.get("polarity", None)
                            if polarity >= 0.995:
                                label = "➕"  # positive
                            elif polarity <= -0.99:
                                label = "➖"  # negative
                            else:
                                label = None  # ignore neutral (1)
                            highlighted.append((sentence, label))
                    elif show_consensuality:
                        highlighted = []
                        for sentence, metadata in review_item:
                            score = metadata.get("consensuality", 0.0)
                            score = score * 2 - 1  # Normalize to [-1, 1]
                            score = score/2.5 if score > 0 else score  # Amplify unique scores for better visibility
                            score *= -1  # Invert the score for highlighting
                            
                            consensuality_dict[sentence] = score
                            highlighted.append((sentence, score))
                        
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

                    review_updates.append(
                        gr.update(
                            visible=True,
                            value=highlighted,
                            color_map=color_map,
                            show_legend=legend,
                            key=f"updated_{score_type}_{i}"
                        )
                    )
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
                    rebuttal_updates.append(gr.update(visible=False, value=""))

            # General rebuttal display (currently unused in new format, kept for backward compat)
            general_rebuttal_update = gr.update(visible=False, value="")

            # Set most consensual / unique sentences
            if show_consensuality and consensuality_dict:
                scores = pd.Series(consensuality_dict)
                most_unique = scores.sort_values(ascending=True).head(3).index.tolist()
                most_common = scores.sort_values(ascending=False).head(3).index.tolist()
                most_common_text = "\n".join(most_common)
                most_unique_text = "\n".join(most_unique)

                most_common_visibility = gr.update(visible=True, value=most_common_text)
                most_unique_visibility = gr.update(visible=True, value=most_unique_text)
            else:
                # Debugging statements to check visibility settings
                # print("Hiding most common and unique sentences")

                most_common_visibility = gr.update(visible=False, value=[])
                most_unique_visibility = gr.update(visible=False, value=[])
                
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
                most_common_visibility,
                most_unique_visibility,
                topic_color_map_visibility,
                *rebuttal_updates,  # 10 per-review rebuttals
                general_rebuttal_update,  # General rebuttal section
                state
            )



        # Precompute the initial outputs so something is shown on load.
        init_display = update_review_display(initial_state, score_type="Original")
        # init_display returns: (review_id, review1..10, most_common, most_unique, topic_box, prep_rebuttal1..10, prep_general_rebuttal, state)

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
                    label="Score Type to Display",
                    value="No Highlighting",
                    interactive=True
                )

        # Output display.
        with gr.Row():
            most_common_sentences = gr.Textbox(
            lines=8,
            label="Most Common Opinions",
            visible=False,
            value=[]
        )
            most_unique_sentences = gr.Textbox(
            lines=8,
            label="Most Divergent Opinions",
            visible=False,
            value=[]
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
        prep_rebuttal1 = gr.HTML(visible=False, value="")
        review2 = gr.HighlightedText(show_legend=False, label="📝 Review 2", visible=number_of_displayed_reviews >= 2, key="initial_review2")
        prep_rebuttal2 = gr.HTML(visible=False, value="")
        review3 = gr.HighlightedText(show_legend=False, label="📝 Review 3", visible=number_of_displayed_reviews >= 3, key="initial_review3")
        prep_rebuttal3 = gr.HTML(visible=False, value="")
        review4 = gr.HighlightedText(show_legend=False, label="📝 Review 4", visible=number_of_displayed_reviews >= 4, key="initial_review4")
        prep_rebuttal4 = gr.HTML(visible=False, value="")
        review5 = gr.HighlightedText(show_legend=False, label="📝 Review 5", visible=number_of_displayed_reviews >= 5, key="initial_review5")
        prep_rebuttal5 = gr.HTML(visible=False, value="")
        review6 = gr.HighlightedText(show_legend=False, label="📝 Review 6", visible=number_of_displayed_reviews >= 6, key="initial_review6")
        prep_rebuttal6 = gr.HTML(visible=False, value="")
        review7 = gr.HighlightedText(show_legend=False, label="📝 Review 7", visible=number_of_displayed_reviews >= 7, key="initial_review7")
        prep_rebuttal7 = gr.HTML(visible=False, value="")
        review8 = gr.HighlightedText(show_legend=False, label="📝 Review 8", visible=number_of_displayed_reviews >= 8, key="initial_review8")
        prep_rebuttal8 = gr.HTML(visible=False, value="")
        review9 = gr.HighlightedText(show_legend=False, label="📝 Review 9", visible=number_of_displayed_reviews >= 9, key="initial_review9")
        prep_rebuttal9 = gr.HTML(visible=False, value="")
        review10 = gr.HighlightedText(show_legend=False, label="📝 Review 10", visible=number_of_displayed_reviews >= 10, key="initial_review10")
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
        _review_outputs = [review_id, review1, review2, review3, review4, review5, review6, review7, review8, review9, review10, most_common_sentences, most_unique_sentences, topic_text_box, prep_rebuttal1, prep_rebuttal2, prep_rebuttal3, prep_rebuttal4, prep_rebuttal5, prep_rebuttal6, prep_rebuttal7, prep_rebuttal8, prep_rebuttal9, prep_rebuttal10, prep_general_rebuttal, state]
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
                most_divergent = gr.Textbox(
                    lines=4, label="Most Divergent Opinions", visible=False, value="", container=True
                )
                most_common = gr.Textbox(
                    lines=4, label="Most Common Opinions", visible=False, value="", container=True
                )

            # Review 1 (all display modes + rebuttal)
            none_text1 = gr.HighlightedText(show_legend=False, label="📝 Review 1", visible=True, value=None)
            agreement_text1 = gr.HighlightedText(show_legend=True, label="Agreement in 📝 Review 1", visible=False, value=None)
            polarity_text1 = gr.HighlightedText(show_legend=True, label="Polarity in 📝 Review 1", visible=False, value=None, color_map={"➕": "#d4fcd6", "➖": "#fcd6d6"})
            topic_text1 = gr.HighlightedText(show_legend=False, label="Topic in 📝 Review 1", visible=False, value=None, color_map=topic_color_map)
            rebuttal_for_review1 = gr.HTML(visible=False, value="")

            # Review 2 (all display modes + rebuttal)
            none_text2 = gr.HighlightedText(show_legend=False, label="📝 Review 2", visible=False, value=None)
            agreement_text2 = gr.HighlightedText(show_legend=True, label="Agreement in 📝 Review 2", visible=False, value=None)
            polarity_text2 = gr.HighlightedText(show_legend=True, label="Polarity in 📝 Review 2", visible=False, value=None, color_map={"➕": "#d4fcd6", "➖": "#fcd6d6"})
            topic_text2 = gr.HighlightedText(show_legend=False, label="Topic in 📝 Review 2", visible=False, value=None, color_map=topic_color_map)
            rebuttal_for_review2 = gr.HTML(visible=False, value="")

            # Review 3 (all display modes + rebuttal)
            none_text3 = gr.HighlightedText(show_legend=False, label="📝 Review 3", visible=False, value=None)
            agreement_text3 = gr.HighlightedText(show_legend=True, label="Agreement in 📝 Review 3", visible=False, value=None)
            polarity_text3 = gr.HighlightedText(show_legend=True, label="Polarity in 📝 Review 3", visible=False, value=None, color_map={"➕": "#d4fcd6", "➖": "#fcd6d6"})
            topic_text3 = gr.HighlightedText(show_legend=False, label="Topic in 📝 Review 3", visible=False, value=None, color_map=topic_color_map)
            rebuttal_for_review3 = gr.HTML(visible=False, value="")

            # Review 4 (all display modes + rebuttal)
            none_text4 = gr.HighlightedText(show_legend=False, label="📝 Review 4", visible=False, value=None)
            agreement_text4 = gr.HighlightedText(show_legend=True, label="Agreement in 📝 Review 4", visible=False, value=None)
            polarity_text4 = gr.HighlightedText(show_legend=True, label="Polarity in 📝 Review 4", visible=False, value=None, color_map={"➕": "#d4fcd6", "➖": "#fcd6d6"})
            topic_text4 = gr.HighlightedText(show_legend=False, label="Topic in 📝 Review 4", visible=False, value=None, color_map=topic_color_map)
            rebuttal_for_review4 = gr.HTML(visible=False, value="")

            # Review 5 (all display modes + rebuttal)
            none_text5 = gr.HighlightedText(show_legend=False, label="📝 Review 5", visible=False, value=None)
            agreement_text5 = gr.HighlightedText(show_legend=True, label="Agreement in 📝 Review 5", visible=False, value=None)
            polarity_text5 = gr.HighlightedText(show_legend=True, label="Polarity in 📝 Review 5", visible=False, value=None, color_map={"➕": "#d4fcd6", "➖": "#fcd6d6"})
            topic_text5 = gr.HighlightedText(show_legend=False, label="Topic in 📝 Review 5", visible=False, value=None, color_map=topic_color_map)
            rebuttal_for_review5 = gr.HTML(visible=False, value="")

            # Review 6 (all display modes + rebuttal)
            none_text6 = gr.HighlightedText(show_legend=False, label="📝 Review 6", visible=False, value=None)
            agreement_text6 = gr.HighlightedText(show_legend=True, label="Agreement in 📝 Review 6", visible=False, value=None)
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
                *([None] * 6), *([None] * 6), *([None] * 6), *([None] * 6),   # clear all output panels (none, agree, polar, topic)
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

            # Most common/divergent only show in Agreement mode (and not while processing)
            show_opinions = effective_focus == "Agreement" and focus != "Agreement (Processing)"
            updates.append(gr.update(visible=show_opinions))  # most_divergent
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