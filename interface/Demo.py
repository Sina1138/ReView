import sys, os.path
from pathlib import Path
from typing import Tuple

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
1. **Enter up to three reviews** in the input fields labeled *Review 1*, *Review 2*, and *Review 3*.
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
_interactive_processor = None

def get_interactive_processor():
    """Lazy-load the processor to avoid duplicate model loading."""
    global _interactive_processor
    if _interactive_processor is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _interactive_processor = InteractiveReviewProcessor(device=device)
    return _interactive_processor


def fetch_openreview_reviews(link: str) -> Tuple[str, str, str, str, str]:
    """
    Fetch reviews from OpenReview link and populate the textboxes.

    Returns:
        Tuple of (review1, review2, review3, title, status_html)
    """
    print(f"\n[DEMO] fetch_openreview_reviews called with link: {link}")

    if not link.strip():
        return ("", "", "", "", _status_html("Please paste a valid OpenReview link", "error"))

    try:
        from interface.interactive_processor import fetch_reviews_from_openreview_link
        reviews, title = fetch_reviews_from_openreview_link(link)
        print(f"[DEMO] Got {len(reviews)} reviews from fetch function")

        while len(reviews) < 3:
            reviews.append("")
        reviews = reviews[:3]

        num_reviews = len([r for r in reviews if r.strip()])
        status = _status_html(f"Fetched {num_reviews} reviews for: {title}", "success")
        return (reviews[0], reviews[1], reviews[2], title, status)

    except ValueError as e:
        error_msg = str(e)
        print(f"[DEMO] ValueError caught: {error_msg}")
        return ("", "", "", "", _status_html(error_msg, "warning"))
    except Exception as e:
        error_msg = str(e)
        print(f"[DEMO] Exception caught: {type(e).__name__}: {error_msg}")

        if "openreview" in error_msg.lower():
            suggestion = " Try: pip install openreview-py"
        elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            suggestion = " Check your internet connection."
        else:
            suggestion = ""

        return ("", "", "", "", _status_html(f"{error_msg}{suggestion}", "error"))


def process_interactive_reviews(text1: str, text2: str, text3: str, focus: str, progress=gr.Progress()) -> Tuple:
    """
    Process reviews through the interactive pipeline with progress tracking.
    """
    from dependencies.Glimpse_tokenizer import glimpse_tokenizer

    # Validate input
    if not text1.strip() and not text2.strip() and not text3.strip():
        raise ValueError("Please enter at least one review")

    # Step 1: Load models
    progress(0.0, desc="Loading models...")
    processor = get_interactive_processor()

    # Step 2: Tokenize
    progress(0.10, desc="Tokenizing reviews...")
    text1_sentences = [s for s in glimpse_tokenizer(text1) if s.strip()]
    text2_sentences = [s for s in glimpse_tokenizer(text2) if s.strip()]
    text3_sentences = [s for s in glimpse_tokenizer(text3) if s.strip()]

    if not text1_sentences or not text2_sentences or not text3_sentences:
        raise ValueError("One or more reviews are empty or have no valid sentences")

    all_sentences = list(set(text1_sentences + text2_sentences + text3_sentences))

    # Step 3: Polarity
    progress(0.20, desc="Predicting polarity...")
    polarity_map = processor.predict_polarity(all_sentences)

    # Step 4: Topic
    progress(0.40, desc="Predicting topics...")
    topic_map = processor.predict_topic(all_sentences)

    # Step 5: Consensuality (RSA) - the slow one
    progress(0.55, desc="Computing agreement (RSA reranking)...")
    consensuality_map = processor.predict_consensuality(text1, text2, text3)

    # Step 6: Format results
    progress(0.90, desc="Formatting results...")

    # Most common / unique
    if consensuality_map:
        import pandas as _pd
        scores_series = _pd.Series(consensuality_map)
        most_common_text = "\n".join(scores_series.nlargest(3).index.tolist())
        most_unique_text = "\n".join(scores_series.nsmallest(3).index.tolist())
    else:
        most_common_text = ""
        most_unique_text = ""

    # Format highlighted outputs
    fmt = processor.format_highlighted_output
    r1_agree = fmt(text1_sentences, consensuality_map, "consensuality")
    r2_agree = fmt(text2_sentences, consensuality_map, "consensuality")
    r3_agree = fmt(text3_sentences, consensuality_map, "consensuality")
    r1_polar = fmt(text1_sentences, polarity_map, "polarity")
    r2_polar = fmt(text2_sentences, polarity_map, "polarity")
    r3_polar = fmt(text3_sentences, polarity_map, "polarity")
    r1_topic = fmt(text1_sentences, topic_map, "topic")
    r2_topic = fmt(text2_sentences, topic_map, "topic")
    r3_topic = fmt(text3_sentences, topic_map, "topic")

    progress(1.0, desc="Done!")

    return (
        r1_agree, r2_agree, r3_agree,
        most_common_text, most_unique_text,
        r1_polar, r2_polar, r3_polar,
        r1_topic, r2_topic, r3_topic,
    )




with gr.Blocks(title="ReView") as demo:
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
        initial_state = {
            "year_choice": initial_year,
            "scored_reviews_for_year": initial_scored_reviews,
            "review_ids": initial_review_ids,
            "current_review_index": 0,
            "current_review": initial_review,
            "number_of_displayed_reviews": number_of_displayed_reviews,
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
            consensuality_dict = {}

            for i in range(8):
                if i < number_of_displayed_reviews:
                    review_item = list(current_review[i].items())

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
                            for sentence, metadata in review_item
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
                state
            )



        # Precompute the initial outputs so something is shown on load.
        init_display = update_review_display(initial_state, score_type="Original")
        # init_display returns: (review_id, review1, review2, review3, review4, review5, review6, review7, review8, state)

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
                    choices=["Original", "Agreement", "Polarity", "Topic"],
                    label="Score Type to Display",
                    value="Original",
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
        
        review1 = gr.HighlightedText(
            show_legend=False,
            label="Review 1",
            visible= number_of_displayed_reviews >= 1,
            key="initial_review1",
            # color_map={"Positive": "#d4fcd6", "Negative": "#fcd6d6"}
        )
        review2 = gr.HighlightedText(
            show_legend=False,
            label="Review 2",
            visible= number_of_displayed_reviews >= 2,
            key="initial_review2"
            # color_map={"Positive": "#d4fcd6", "Negative": "#fcd6d6"}
        )
        review3 = gr.HighlightedText(
            show_legend=False,
            label="Review 3",
            visible= number_of_displayed_reviews >= 3,
            key="initial_review3"
            # color_map={"Positive": "#d4fcd6", "Negative": "#fcd6d6"}
        )
        review4 = gr.HighlightedText(
            show_legend=False,
            label="Review 4",
            visible= number_of_displayed_reviews >= 4,
            key="initial_review4"
            # color_map={"Positive": "#d4fcd6", "Negative": "#fcd6d6"}
        )
        review5 = gr.HighlightedText(
            show_legend=False,
            label="Review 5",
            visible= number_of_displayed_reviews >= 5,
            key="initial_review5"
            # color_map={"Positive": "#d4fcd6", "Negative": "#fcd6d6"}
        )
        review6 = gr.HighlightedText(
            show_legend=False,
            label="Review 6",
            visible= number_of_displayed_reviews >= 6,
            key="initial_review6"
            # color_map={"Positive": "#d4fcd6", "Negative": "#fcd6d6"}
        )
        review7 = gr.HighlightedText(
            show_legend=False,
            label="Review 7",
            visible= number_of_displayed_reviews >= 7,
            key="initial_review7"
            # color_map={"Positive": "#d4fcd6", "Negative": "#fcd6d6"}
        )
        review8 = gr.HighlightedText(
            show_legend=False,
            label="Review 8",
            visible= number_of_displayed_reviews >= 8,
            key="initial_review8"
            # color_map={"Positive": "#d4fcd6", "Negative": "#fcd6d6"}
        )

        # Callback functions that update state.
        def year_change(year, state, score_type):
            state["year_choice"] = year
            state["scored_reviews_for_year"] = get_preprocessed_scores(year)
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
        year.change(
            fn=year_change,
            inputs=[year, state, score_type],
            outputs=[review_id, review1, review2, review3, review4, review5, review6, review7, review8, most_common_sentences, most_unique_sentences, topic_text_box, state]
        )
        score_type.change(
            fn=update_review_display,
            inputs=[state, score_type],
            outputs=[review_id, review1, review2, review3, review4, review5, review6, review7, review8, most_common_sentences, most_unique_sentences, topic_text_box, state]
        )
        next_button.click(
            fn=next_review,
            inputs=[state, score_type],
            outputs=[review_id, review1, review2, review3, review4, review5, review6, review7, review8, most_common_sentences, most_unique_sentences, topic_text_box, state]
        )
        previous_button.click(
            fn=previous_review,
            inputs=[state, score_type],
            outputs=[review_id, review1, review2, review3, review4, review5, review6, review7, review8, most_common_sentences, most_unique_sentences, topic_text_box, state]
        )   
        
        
        
        
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
            gr.Markdown("## Input Reviews")

            with gr.Tabs():
                with gr.Tab("Paste Reviews"):
                    review1_textbox = gr.Textbox(lines=5, value=EXAMPLES[0], label="Review 1", interactive=True)
                    review2_textbox = gr.Textbox(lines=5, value=EXAMPLES[1], label="Review 2", interactive=True)
                    review3_textbox = gr.Textbox(lines=5, value=EXAMPLES[2], label="Review 3", interactive=True)
                    with gr.Row():
                        submit_button = gr.Button("Process", variant="primary", interactive=True)
                        clear_button = gr.Button("Clear", variant="secondary", interactive=True)

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

            status_html = gr.HTML("", visible=False)

        # ---- RESULTS SECTION (full-width, hidden initially) ----
        with gr.Column(visible=False) as results_section:
            focus_radio = gr.Radio(
                choices=["Agreement", "Polarity", "Topic"],
                value="Agreement",
                label="Display Mode:",
                interactive=True
            )

            with gr.Row():
                most_divergent = gr.Textbox(
                    lines=4, label="Most Divergent Opinions", visible=True, value="", container=True
                )
                most_common = gr.Textbox(
                    lines=4, label="Most Common Opinions", visible=True, value="", container=True
                )

            agreement_text1 = gr.HighlightedText(
                show_legend=True, label="Agreement in Review 1", visible=True, value=None,
            )
            agreement_text2 = gr.HighlightedText(
                show_legend=True, label="Agreement in Review 2", visible=True, value=None,
            )
            agreement_text3 = gr.HighlightedText(
                show_legend=True, label="Agreement in Review 3", visible=True, value=None,
            )

            polarity_text1 = gr.HighlightedText(
                show_legend=True, label="Polarity in Review 1", visible=False, value=None,
                color_map={"➕": "#d4fcd6", "➖": "#fcd6d6"}
            )
            polarity_text2 = gr.HighlightedText(
                show_legend=True, label="Polarity in Review 2", visible=False, value=None,
                color_map={"➕": "#d4fcd6", "➖": "#fcd6d6"}
            )
            polarity_text3 = gr.HighlightedText(
                show_legend=True, label="Polarity in Review 3", visible=False, value=None,
                color_map={"➕": "#d4fcd6", "➖": "#fcd6d6"}
            )

            topic_text1 = gr.HighlightedText(
                show_legend=False, label="Topic in Review 1", visible=False, value=None,
                color_map=topic_color_map
            )
            topic_text2 = gr.HighlightedText(
                show_legend=False, label="Topic in Review 2", visible=False, value=None,
                color_map=topic_color_map
            )
            topic_text3 = gr.HighlightedText(
                show_legend=False, label="Topic in Review 3", visible=False, value=None,
                color_map=topic_color_map
            )

        # ---- CALLBACKS ----

        # Fetch OpenReview reviews → show timer → process → swap to results
        fetch_reviews_button.click(
            fn=lambda: (
                gr.update(value=FETCHING_HTML, visible=True),
                gr.update(interactive=False),
            ),
            inputs=[],
            outputs=[status_html, fetch_reviews_button]
        ).then(
            fn=fetch_openreview_reviews,
            inputs=[openreview_link_input],
            outputs=[review1_textbox, review2_textbox, review3_textbox, openreview_title, status_html]
        ).then(
            fn=lambda title: (
                gr.update(visible=bool(title.strip())),
                gr.update(value=PROCESSING_TIMER_HTML, visible=True),
            ),
            inputs=[openreview_title],
            outputs=[openreview_title, status_html]
        ).then(
            fn=process_interactive_reviews,
            inputs=[review1_textbox, review2_textbox, review3_textbox, focus_radio],
            outputs=[
                agreement_text1, agreement_text2, agreement_text3,
                most_common, most_divergent,
                polarity_text1, polarity_text2, polarity_text3,
                topic_text1, topic_text2, topic_text3
            ]
        ).then(
            fn=lambda: (
                gr.update(visible=False),                    # hide input
                gr.update(visible=True),                     # show results
                gr.update(value="", visible=False),          # clear status
                gr.update(interactive=True),                 # re-enable fetch button
                gr.update(visible=True),                     # show "back to input" toggle
                gr.update(visible=False),                    # hide "view results" toggle
            ),
            inputs=[],
            outputs=[input_section, results_section, status_html, fetch_reviews_button, back_to_input_btn, view_results_btn]
        )

        # Process (Paste Reviews): show timer → run scoring → swap to results
        submit_button.click(
            fn=lambda: (
                gr.update(value=PROCESSING_TIMER_HTML, visible=True),
                gr.update(interactive=False),
            ),
            inputs=[],
            outputs=[status_html, submit_button]
        ).then(
            fn=process_interactive_reviews,
            inputs=[review1_textbox, review2_textbox, review3_textbox, focus_radio],
            outputs=[
                agreement_text1, agreement_text2, agreement_text3,
                most_common, most_divergent,
                polarity_text1, polarity_text2, polarity_text3,
                topic_text1, topic_text2, topic_text3
            ]
        ).then(
            fn=lambda: (
                gr.update(visible=False),                    # hide input
                gr.update(visible=True),                     # show results
                gr.update(value="", visible=False),          # clear status
                gr.update(interactive=True),                 # re-enable submit button
                gr.update(visible=True),                     # show "back to input" toggle
                gr.update(visible=False),                    # hide "view results" toggle
            ),
            inputs=[],
            outputs=[input_section, results_section, status_html, submit_button, back_to_input_btn, view_results_btn]
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
            fn=lambda: (None, None, None, "", "", None, None, None, None, None, None),
            inputs=[],
            outputs=[
                review1_textbox, review2_textbox, review3_textbox,
                most_common, most_divergent,
                agreement_text1, agreement_text2, agreement_text3
            ]
        )

        # Toggle display mode (Agreement / Polarity / Topic)
        def toggle_display_mode(focus):
            agreement_visible = (focus == "Agreement")
            polarity_visible = (focus == "Polarity")
            topic_visible = (focus == "Topic")
            return (
                gr.update(visible=agreement_visible), gr.update(visible=agreement_visible), gr.update(visible=agreement_visible),
                gr.update(visible=polarity_visible), gr.update(visible=polarity_visible), gr.update(visible=polarity_visible),
                gr.update(visible=topic_visible), gr.update(visible=topic_visible), gr.update(visible=topic_visible),
            )

        focus_radio.change(
            fn=toggle_display_mode,
            inputs=[focus_radio],
            outputs=[
                agreement_text1, agreement_text2, agreement_text3,
                polarity_text1, polarity_text2, polarity_text3,
                topic_text1, topic_text2, topic_text3,
            ]
        )
        
demo.launch(share=False)