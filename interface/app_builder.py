"""
Shared app builder for ReView interface.

Extracts the Gradio Blocks construction from Demo.py into a configurable
factory function. Supports both full-highlight and no-highlight study
conditions, with optional JSONL interaction logging.
"""

import sys
import os.path
import threading
import time
import uuid
import json as _json
from pathlib import Path
from typing import Tuple, Dict, List, Optional

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

from interface.constants import (
    MAX_PREPROCESSED_REVIEWS, MAX_INTERACTIVE_REVIEWS,
    DISPLAY_MODES, TOPIC_COLOR_MAP, TOPIC_HTML_COLORS,
    CUSTOM_CSS, EXAMPLES,
    FETCHING_HTML, POLARITY_PROGRESS_HTML, AGREEMENT_PROGRESS_HTML,
    POLARITY_LEGEND, TOPIC_LEGEND,
)
from interface.renderers import (
    build_review_card, format_common_themes, format_divergent_cards,
    format_rebuttal_plain, format_rebuttal_for_review, format_general_rebuttals,
    render_status, render_agreement_progress,
    review_toggle_html, rebuttal_toggle_html, jump_buttons_html,
    is_review_header,
)
from interface.study_config import StudyConfig
from interface.interaction_logger import InteractionLogger, JS_EVENT_BRIDGE

# Module-level storage for background thread state (avoids pickling issues with ZeroGPU)
_thread_states: Dict = {}


# ---------------------------------------------------------------------------
# Dataset loading (shared across all conditions)
# ---------------------------------------------------------------------------

def _find_preprocessed_csv() -> Path:
    """Find the most recent preprocessed_scored_reviews_*.csv in the data dir."""
    data_dir = BASE_DIR / "data"
    candidates = sorted(data_dir.glob("preprocessed_scored_reviews_*.csv"))
    if candidates:
        return candidates[-1]
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

    if "metadata" in df.columns:
        df["metadata"] = df["metadata"].progress_apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and x != '{}' else {}
        )
    else:
        df["metadata"] = [{}] * len(df)

    years = df["year"].tolist()
    return years, df


def _load_paper_titles() -> dict:
    titles = {}
    for csv in sorted((BASE_DIR / "data").glob("all_reviews_*.csv")):
        try:
            df_raw = pd.read_csv(csv, usecols=["id", "paper_title"])
            for _, row in df_raw.iterrows():
                if row["id"] not in titles and pd.notna(row.get("paper_title", "")):
                    titles[row["id"]] = str(row["paper_title"])
        except Exception:
            pass
    return titles


# Load dataset once at module level
_years_new, _df_new = load_scored_reviews_with_rebuttals()
if _df_new.empty:
    raise FileNotFoundError(
        "No preprocessed dataset found. Run the pipeline first (./pipeline/process_new_data.sh)."
    )
years, all_scored_reviews_df = _years_new, _df_new
_paper_titles = _load_paper_titles()


# ---------------------------------------------------------------------------
# Pre-processed tab helpers
# ---------------------------------------------------------------------------

def get_preprocessed_scores(year):
    scored_reviews = all_scored_reviews_df[all_scored_reviews_df["year"] == year]["scored_dict"].iloc[0]
    return scored_reviews


def get_preprocessed_metadata(year):
    row = all_scored_reviews_df[all_scored_reviews_df["year"] == year]
    if "metadata" in row.columns and not row.empty:
        meta = row["metadata"].iloc[0]
        return meta if isinstance(meta, dict) else {}
    return {}


# ---------------------------------------------------------------------------
# Interactive tab: model initialization (lazy)
# ---------------------------------------------------------------------------

from interface.interactive_processor import InteractiveReviewProcessor
from dependencies.sentence_filter import (
    is_noise_sentence, filter_and_clean_sentences,
    HIGHLIGHT_THRESHOLD,
)

_interactive_processor = None


def get_interactive_processor():
    """Lazy-load the processor to avoid duplicate model loading."""
    global _interactive_processor
    if _interactive_processor is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _interactive_processor = InteractiveReviewProcessor(device=device)
    return _interactive_processor


@_gpu
def _gpu_predict_polarity_topic(sentences: List[str]) -> Tuple[Dict, Dict]:
    """Run polarity + topic inference on GPU. Decorated with @spaces.GPU for ZeroGPU."""
    processor = get_interactive_processor()
    processor.ensure_device()
    polarity_map = processor.predict_polarity(sentences)
    topic_map = processor.predict_topic(sentences)
    return polarity_map, topic_map


# ---------------------------------------------------------------------------
# OpenReview fetching
# ---------------------------------------------------------------------------

def fetch_openreview_reviews(link: str):
    """Fetch reviews from OpenReview link and populate the textboxes."""
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
        status = render_status(f"Fetched {num_reviews} reviews for: {title}", "success")
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


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_review_app(config: StudyConfig) -> gr.Blocks:
    """Build the ReView Gradio app for the given study configuration.

    Returns a gr.Blocks instance ready to .launch().
    """
    highlights = config.highlights_enabled
    logger = InteractionLogger(config)

    # --- Theme (force light-mode colors in dark mode) ---
    _theme = gr.themes.Default()
    _dark_overrides = {}
    for _attr in dir(_theme):
        if _attr.endswith('_dark') and not _attr.startswith('_'):
            _light_attr = _attr[:-5]
            _light_val = getattr(_theme, _light_attr, None)
            if _light_val is not None:
                _dark_overrides[_attr] = _light_val
    _theme.set(**_dark_overrides)

    # --- JS: back-to-top button + radio grey-out + event bridge ---
    app_js = r"""() => {
        var btn = document.createElement('button');
        btn.id = 'back-to-top-btn';
        btn.textContent = '\u2191 Top';
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
"""

    if highlights:
        # Gray-out radio choices containing hourglass while processing
        app_js += r"""
        var _prevRadio = 'No Highlighting';
        setInterval(function() {
            var labels = document.querySelectorAll('.no-border-row label input[type=radio]');
            labels.forEach(function(inp) {
                var lbl = inp.closest('label') || inp.parentElement;
                var txt = (lbl && lbl.textContent) || '';
                if (txt.indexOf('\u23f3') !== -1) {
                    lbl.style.opacity = '0.4';
                    lbl.style.pointerEvents = 'none';
                    lbl.title = 'Still computing\u2026';
                } else {
                    lbl.style.opacity = '';
                    lbl.style.pointerEvents = '';
                    lbl.title = '';
                }
            });
        }, 300);
"""

    # JS event bridge for logging HTML-only controls
    if config.logging_enabled:
        app_js += "\n" + JS_EVENT_BRIDGE + "\n"

    app_js += "}"  # close the outer () => { ... }

    # ------------------------------------------------------------------
    # BUILD THE BLOCKS
    # ------------------------------------------------------------------

    with gr.Blocks(
        title="ReView",
        css=CUSTOM_CSS,
        theme=_theme,
        js=app_js,
    ) as demo:

        # Hidden textbox sink for JS event bridge (logging)
        if config.logging_enabled:
            js_event_sink = gr.Textbox(visible=False, elem_id="js-event-sink")

        # ==============================================================
        # PRE-PROCESSED TAB
        # ==============================================================
        with gr.Tab("Pre-processed Reviews", elem_classes=["results-compact"]):
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
                # In no-highlight mode, force plain display
                if not highlights:
                    score_type = "No Highlighting"

                review_ids = state["review_ids"]
                current_index = state["current_review_index"]
                current_review = state["scored_reviews_for_year"][review_ids[current_index]]

                show_polarity = score_type == "Polarity"
                show_consensuality = score_type == "Agreement"
                show_topic = score_type == "Topic"

                if show_polarity:
                    color_map = {"➕": "#d4fcd6", "➖": "#fcd6d6"}
                elif show_topic:
                    color_map = TOPIC_COLOR_MAP
                elif show_consensuality:
                    color_map = None
                else:
                    color_map = {}

                current_id = review_ids[current_index]
                paper_title = _paper_titles.get(current_id, "")
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

                review_sentence_lists = []
                review_items_cache = []
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

                    meta_for_year = state.get("metadata_for_year", {})
                    submission_meta = meta_for_year.get(current_id, {})
                    if isinstance(submission_meta, dict):
                        rsa_data = submission_meta.get("rsa", {})
                        if rsa_data:
                            prep_listener = rsa_data.get("listener")
                            prep_speaker = rsa_data.get("speaker")

                agreement_updates = []
                divergent_per_review = {}
                if show_consensuality and prep_listener and prep_speaker and consensuality_dict:
                    divergent_per_review = format_divergent_cards(
                        consensuality_dict, review_sentence_lists, prep_listener, prep_speaker,
                    )

                for i in range(MAX_PREPROCESSED_REVIEWS):
                    if i < number_of_displayed_reviews:
                        review_item, rebuttal_html = review_items_cache[i]

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

                general_rebuttal_update = gr.update(visible=False, value="")

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

                if show_polarity:
                    topic_color_map_visibility = gr.update(visible=True, value=POLARITY_LEGEND)
                elif show_topic:
                    topic_color_map_visibility = gr.update(visible=True, value=TOPIC_LEGEND)
                else:
                    topic_color_map_visibility = gr.update(visible=False, value="")

                has_any_rebuttal = any(
                    rebuttal_html
                    for _, rebuttal_html in review_items_cache
                )
                toggle_buttons = [review_toggle_html()]
                if has_any_rebuttal:
                    toggle_buttons.append(rebuttal_toggle_html())
                jump_html = jump_buttons_html(number_of_displayed_reviews, prefix="pre")
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
                    *agreement_updates,
                    most_common_visibility,
                    most_unique_visibility,
                    topic_color_map_visibility,
                    toggle_bar_update,
                    *rebuttal_updates,
                    general_rebuttal_update,
                    state
                )

            init_display = update_review_display(initial_state, score_type="Original")

            with gr.Row():
                with gr.Column(scale=1):
                    review_id = gr.Markdown(value=init_display[0], container=True)
                    with gr.Row():
                        previous_button = gr.Button("Previous", variant="secondary", interactive=True)
                        next_button = gr.Button("Next", variant="primary", interactive=True)

                with gr.Column(scale=1):
                    year = gr.Dropdown(choices=years, label="Select Year", interactive=True, value=initial_year)
                    if highlights:
                        score_type = gr.Radio(
                            choices=["No Highlighting", "Polarity", "Topic", "Agreement"],
                            label="Display Mode:",
                            value="No Highlighting",
                            interactive=True
                        )
                    else:
                        # No-highlight condition: hidden radio fixed to plain
                        score_type = gr.Radio(
                            choices=["No Highlighting"],
                            label="Display Mode:",
                            value="No Highlighting",
                            visible=False,
                            interactive=False
                        )

            with gr.Row():
                most_common_sentences = gr.HTML(visible=False, value="", label="Most Common Opinions")
                most_unique_sentences = gr.HTML(visible=False, value="", label="Most Divergent Opinions")

            topic_text_box = gr.HTML(visible=False, value="")
            prep_toggle_bar = gr.HTML(visible=False, value="")
            prep_reviews = []
            prep_agreements = []
            prep_rebuttals = []
            for _i in range(MAX_PREPROCESSED_REVIEWS):
                gr.HTML(value=f'<div id="pre-review-anchor-{_i+1}"></div>', elem_classes=["review-anchor"])
                prep_reviews.append(gr.HighlightedText(show_legend=False, label=f"📝 Review {_i+1}", visible=number_of_displayed_reviews >= _i+1, key=f"initial_review{_i+1}"))
                prep_agreements.append(gr.HTML(visible=False, value=""))
                prep_rebuttals.append(gr.HTML(visible=False, value=""))
            prep_general_rebuttal = gr.HTML(visible=False, value="")

            # --- Pre-processed callbacks ---

            def year_change(year_val, st, st_type):
                st["year_choice"] = year_val
                st["scored_reviews_for_year"] = get_preprocessed_scores(year_val)
                st["metadata_for_year"] = get_preprocessed_metadata(year_val)
                st["review_ids"] = list(st["scored_reviews_for_year"].keys())
                st["current_review_index"] = 0
                st["current_review"] = st["scored_reviews_for_year"][st["review_ids"][0]]
                logger.log("year_change", tab="pre", source="user",
                           payload={"year": year_val})
                return update_review_display(st, st_type)

            def next_review(st, st_type):
                st["current_review_index"] = (st["current_review_index"] + 1) % len(st["review_ids"])
                st["current_review"] = st["scored_reviews_for_year"][st["review_ids"][st["current_review_index"]]]
                logger.log("next_review", tab="pre", source="user",
                           paper_id=st["review_ids"][st["current_review_index"]])
                return update_review_display(st, st_type)

            def previous_review(st, st_type):
                st["current_review_index"] = (st["current_review_index"] - 1) % len(st["review_ids"])
                st["current_review"] = st["scored_reviews_for_year"][st["review_ids"][st["current_review_index"]]]
                logger.log("previous_review", tab="pre", source="user",
                           paper_id=st["review_ids"][st["current_review_index"]])
                return update_review_display(st, st_type)

            _review_outputs = [review_id, *prep_reviews, *prep_agreements, most_common_sentences, most_unique_sentences, topic_text_box, prep_toggle_bar, *prep_rebuttals, prep_general_rebuttal, state]
            year.change(fn=year_change, inputs=[year, state, score_type], outputs=_review_outputs)

            if highlights:
                def _log_pre_mode_change(st, st_type):
                    logger.log("mode_change", tab="pre", source="user",
                               visible_mode=st_type)
                    return update_review_display(st, st_type)
                score_type.change(fn=_log_pre_mode_change, inputs=[state, score_type], outputs=_review_outputs)
            else:
                score_type.change(fn=update_review_display, inputs=[state, score_type], outputs=_review_outputs)

            next_button.click(fn=next_review, inputs=[state, score_type], outputs=_review_outputs)
            previous_button.click(fn=previous_review, inputs=[state, score_type], outputs=_review_outputs)

        # ==============================================================
        # INTERACTIVE TAB
        # ==============================================================
        with gr.Tab("Interactive", interactive=True):

            # ---- TOP TOGGLE BAR (always visible) ----
            with gr.Row():
                paper_title = gr.Textbox("", visible=False, interactive=False, show_label=False, container=False, elem_classes=["paper-title-heading"])
                back_to_input_btn = gr.Button("✏️ Edit Reviews / New Input", visible=False, variant="secondary")
                view_results_btn = gr.Button("📊 View Results", visible=False, variant="secondary")

            # ---- INPUT SECTION ----
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
                        openreview_title = gr.State("")
                        openreview_rebuttal = gr.State("")

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

            # ---- RESULTS SECTION ----
            with gr.Column(visible=False, elem_classes=["results-compact"]) as results_section:
                if highlights:
                    with gr.Row(elem_classes=["no-border-row"]):
                        focus_radio = gr.Radio(
                            choices=["No Highlighting", "Polarity", "Topic", "Agreement (Processing)"],
                            value="No Highlighting",
                            label="Display Mode:",
                            interactive=True
                        )

                    with gr.Group(elem_classes=["progress-group"]):
                        polarity_progress_html = gr.HTML("", visible=False, elem_classes=["progress-compact"])
                        agreement_progress_html = gr.HTML("", visible=False, elem_classes=["progress-compact"])

                    interactive_legend_html = gr.HTML("", visible=False)

                    with gr.Row():
                        most_divergent = gr.HTML(visible=False, value="", label="Most Divergent Opinions")
                        most_common = gr.HTML(visible=False, value="", label="Most Common Opinions")
                else:
                    # No-highlight: no display mode controls, no progress, no legends
                    focus_radio = gr.Radio(
                        choices=["No Highlighting"],
                        value="No Highlighting",
                        visible=False,
                        interactive=False
                    )
                    polarity_progress_html = gr.HTML("", visible=False)
                    agreement_progress_html = gr.HTML("", visible=False)
                    interactive_legend_html = gr.HTML("", visible=False)
                    most_divergent = gr.HTML(visible=False, value="")
                    most_common = gr.HTML(visible=False, value="")

                interactive_rebuttal_toggle = gr.HTML(visible=False, value="")

                none_texts = []
                agreement_texts = []
                polarity_texts = []
                topic_texts = []
                rebuttal_for_reviews = []
                for _i in range(MAX_INTERACTIVE_REVIEWS):
                    gr.HTML(value=f'<div id="int-review-anchor-{_i+1}"></div>', elem_classes=["review-anchor"])
                    none_texts.append(gr.HTML(visible=(_i == 0), value="", elem_id=f"int-review-{_i+1}"))
                    if highlights:
                        agreement_texts.append(gr.HTML(visible=False, value=""))
                        polarity_texts.append(gr.HTML(visible=False, value=""))
                        topic_texts.append(gr.HTML(visible=False, value=""))
                    else:
                        agreement_texts.append(gr.HTML(visible=False, value=""))
                        polarity_texts.append(gr.HTML(visible=False, value=""))
                        topic_texts.append(gr.HTML(visible=False, value=""))
                    rebuttal_for_reviews.append(gr.HTML(visible=False, value=""))

                interactive_rebuttal_display = gr.HTML(visible=False, value="")

            # ---- CALLBACK DEFINITIONS ----

            interactive_rebuttal_state = gr.State("")
            processing_thread_state = gr.State(None)

            _interactive_inputs = [review1_textbox, review2_textbox, review3_textbox, review4_textbox, review5_textbox, review6_textbox, focus_radio, interactive_rebuttal_state, processing_thread_state]

            rsa_computation_state = gr.State({})

            _none_sinks = [gr.State(None) for _ in range(MAX_INTERACTIVE_REVIEWS)]

            _interactive_outputs = [
                *_none_sinks,
                *agreement_texts,
                most_common, most_divergent,
                *polarity_texts,
                *topic_texts,
                interactive_review_count,
                rsa_computation_state,
            ]

            _rsa_outputs = [
                *agreement_texts,
                most_common, most_divergent,
                agreement_progress_html,
                focus_radio,
            ]

            _show_raw_outputs = [
                *none_texts,
                input_section, results_section, back_to_input_btn, paper_title, view_results_btn, focus_radio,
                polarity_progress_html, agreement_progress_html,
                interactive_rebuttal_toggle,
                *rebuttal_for_reviews,
                interactive_rebuttal_display, interactive_review_count,
                interactive_rebuttal_state, interactive_legend_html,
                processing_thread_state,
            ]

            _toggle_outputs = [
                *none_texts,
                *polarity_texts,
                *topic_texts,
                *agreement_texts,
                most_divergent, most_common,
                interactive_legend_html,
            ]

            # --- Validate & start fetch ---
            def _validate_and_start_fetch(link):
                if not link or not link.strip():
                    raise gr.Error("Please paste a valid OpenReview link before fetching.")
                logger.log("fetch_clicked", tab="int", source="user",
                           payload={"link_domain": link.strip().split("/")[2] if "/" in link else ""})
                return gr.update(value=FETCHING_HTML, visible=True), gr.update(interactive=False)

            no_title_state = gr.State("")

            # --- Show raw reviews and switch to results ---

            if highlights:
                def _show_raw_and_switch(r1, r2, r3, r4, r5, r6, rebuttal, title=""):
                    """Immediately switch to results with raw tokenized reviews.
                    Kicks off polarity+topic in background thread."""
                    import time as _time
                    from dependencies.Glimpse_tokenizer import glimpse_tokenizer
                    texts = [r1, r2, r3, r4, r5, r6]
                    active_count = sum(1 for t in texts if t and t.strip())

                    rebuttal_htmls = [format_rebuttal_for_review(rebuttal or "", i + 1) for i in range(6)]
                    has_per_review = any(rebuttal_htmls)

                    none_out = []
                    for idx, t in enumerate(texts):
                        if t and t.strip():
                            sentences = [s for s in glimpse_tokenizer(t) if s.strip()]
                            plain_items = [(s, {}) for s in sentences]
                            html = build_review_card(f"Review {idx + 1}", review_items=plain_items, mode="plain", rebuttal_html=rebuttal_htmls[idx])
                            none_out.append(gr.update(visible=True, value=html))
                        else:
                            none_out.append(gr.update(visible=False, value=""))

                    per_review = [gr.update(visible=False, value="") for _ in range(6)]

                    general_formatted = format_general_rebuttals(rebuttal or "")
                    has_any = has_per_review or bool(general_formatted)

                    right_buttons = [review_toggle_html()]
                    if has_any:
                        right_buttons.append(rebuttal_toggle_html())
                    toggle_bar = (
                        '<div style="display:flex;align-items:center;gap:8px;">'
                        '<span style="font-size:0.78em;color:#6b7280;white-space:nowrap;">Jump to:</span>'
                        + jump_buttons_html(active_count)
                        + '<span style="flex:1;"></span>'
                        + "".join(right_buttons) + '</div>'
                    )

                    title_text = title.strip() if title and title.strip() else ""

                    # Start polarity+topic in background thread
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

                    thread_key = str(uuid.uuid4())
                    _thread_states[thread_key] = {
                        "thread": bg_thread,
                        "result": _thread_result,
                        "sentence_lists": sentence_lists,
                        "active_texts": active_texts,
                        "all_sentences": all_sentences,
                    }

                    return (
                        *none_out,
                        gr.update(visible=False),                              # input_section
                        gr.update(visible=True),                               # results_section
                        gr.update(visible=True),                               # back_to_input_btn
                        gr.update(visible=bool(title_text), value=title_text), # paper_title
                        gr.update(visible=False),                              # view_results_btn
                        gr.update(choices=["No Highlighting", "Polarity ⏳", "Topic ⏳", "Agreement ⏳"],
                                   value="No Highlighting", interactive=True), # focus_radio
                        gr.update(visible=True, value=POLARITY_PROGRESS_HTML), # polarity_progress_html
                        gr.update(visible=True, value=AGREEMENT_PROGRESS_HTML),# agreement_progress_html
                        gr.update(visible=True, value=toggle_bar),             # interactive_rebuttal_toggle
                        *per_review,
                        gr.update(visible=bool(general_formatted), value=general_formatted),
                        active_count,
                        rebuttal or "",
                        gr.update(visible=False, value=""),
                        thread_key,
                    )

            else:
                # No-highlight: simpler version — no background ML thread
                def _show_raw_and_switch(r1, r2, r3, r4, r5, r6, rebuttal, title=""):
                    """Show raw tokenized reviews. No ML processing."""
                    from dependencies.Glimpse_tokenizer import glimpse_tokenizer
                    texts = [r1, r2, r3, r4, r5, r6]
                    active_count = sum(1 for t in texts if t and t.strip())

                    rebuttal_htmls = [format_rebuttal_for_review(rebuttal or "", i + 1) for i in range(6)]
                    has_per_review = any(rebuttal_htmls)

                    none_out = []
                    for idx, t in enumerate(texts):
                        if t and t.strip():
                            sentences = [s for s in glimpse_tokenizer(t) if s.strip()]
                            plain_items = [(s, {}) for s in sentences]
                            html = build_review_card(f"Review {idx + 1}", review_items=plain_items, mode="plain", rebuttal_html=rebuttal_htmls[idx])
                            none_out.append(gr.update(visible=True, value=html))
                        else:
                            none_out.append(gr.update(visible=False, value=""))

                    per_review = [gr.update(visible=False, value="") for _ in range(6)]

                    general_formatted = format_general_rebuttals(rebuttal or "")
                    has_any = has_per_review or bool(general_formatted)

                    right_buttons = [review_toggle_html()]
                    if has_any:
                        right_buttons.append(rebuttal_toggle_html())
                    toggle_bar = (
                        '<div style="display:flex;align-items:center;gap:8px;">'
                        '<span style="font-size:0.78em;color:#6b7280;white-space:nowrap;">Jump to:</span>'
                        + jump_buttons_html(active_count)
                        + '<span style="flex:1;"></span>'
                        + "".join(right_buttons) + '</div>'
                    )

                    title_text = title.strip() if title and title.strip() else ""

                    logger.log("view_results", tab="int", source="system",
                               paper_id=title_text,
                               payload={"active_reviews": active_count})

                    return (
                        *none_out,
                        gr.update(visible=False),                              # input_section
                        gr.update(visible=True),                               # results_section
                        gr.update(visible=True),                               # back_to_input_btn
                        gr.update(visible=bool(title_text), value=title_text), # paper_title
                        gr.update(visible=False),                              # view_results_btn
                        gr.update(value="No Highlighting"),                    # focus_radio (hidden, no change)
                        gr.update(visible=False, value=""),                    # polarity_progress_html
                        gr.update(visible=False, value=""),                    # agreement_progress_html
                        gr.update(visible=True, value=toggle_bar),             # interactive_rebuttal_toggle
                        *per_review,
                        gr.update(visible=bool(general_formatted), value=general_formatted),
                        active_count,
                        rebuttal or "",
                        gr.update(visible=False, value=""),
                        None,  # processing_thread_state — no thread
                    )

            # --- Fast processing (full condition only) ---

            def process_interactive_reviews_fast(text1, text2, text3, text4, text5, text6, focus, rebuttal_str="", thread_state=None, progress=gr.Progress()):
                """Fast processing: Polarity + Topic only. RSA runs in background."""
                import time as _time
                from dependencies.Glimpse_tokenizer import glimpse_tokenizer

                t_start = _time.time()

                _bg_state = _thread_states.pop(thread_state, None) if isinstance(thread_state, str) else None
                if _bg_state and _bg_state.get("thread"):
                    bg_thread = _bg_state["thread"]
                    _result = _bg_state["result"]
                    sentence_lists = _bg_state["sentence_lists"]
                    active_texts = _bg_state["active_texts"]
                    all_sentences = _bg_state["all_sentences"]

                    progress(0.30, desc="Predicting polarity and topics...")
                    bg_thread.join()

                    if _result.get("error"):
                        raise _result["error"]

                    polarity_map = _result["polarity"]
                    topic_map = _result["topic"]
                    print(f"[TIMING] Polarity+Topic (from early-start thread): {_time.time() - t_start:.1f}s wait")
                else:
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
                    polarity_map, topic_map = _gpu_predict_polarity_topic(all_sentences)
                    print(f"[TIMING] Polarity+Topic (sequential): {_time.time() - t0:.1f}s")

                print(f"[TIMING] Fast processing total: {_time.time() - t_start:.1f}s")

                progress(0.90, desc="Formatting results...")

                rebuttal_htmls = [format_rebuttal_for_review(rebuttal_str or "", i + 1) for i in range(MAX_INTERACTIVE_REVIEWS)]

                none_out, agree_out, polar_out, topic_out = [], [], [], []
                for i in range(MAX_INTERACTIVE_REVIEWS):
                    if i < len(sentence_lists):
                        review_label = f"Review {i + 1}"
                        reb = rebuttal_htmls[i]
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
                    "", "",
                    *polar_out,
                    *topic_out,
                    len(sentence_lists),
                    rsa_state,
                )

            def compute_rsa_in_background(rsa_state_val, current_focus):
                """Generator: streams progress while RSA runs, then yields final agreement HTML."""
                _empty = tuple([gr.update(visible=False, value="")] * (MAX_INTERACTIVE_REVIEWS + 2)
                               + [gr.update(visible=False, value=""),
                                  gr.update(choices=["No Highlighting", "Polarity", "Topic", "Agreement"], interactive=True)])

                if not rsa_state_val or not rsa_state_val.get("sentence_lists"):
                    yield _empty
                    return

                processor = get_interactive_processor()
                sentence_lists = rsa_state_val["sentence_lists"]
                active_texts = rsa_state_val["active_texts"]

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

                _no_op_8 = [gr.update()] * (MAX_INTERACTIVE_REVIEWS + 2)
                while t.is_alive():
                    done, total = _prog["done"], _prog["total"]
                    pct = int(done / total * 100) if total > 0 else 0
                    eta_sec = None
                    elapsed = None
                    rate = None
                    t0 = _prog.get("start_time")
                    if t0 and done > 0:
                        elapsed = time.time() - t0
                        rate = elapsed / done
                        eta_sec = rate * (total - done)
                    yield (*_no_op_8, gr.update(visible=True, value=render_agreement_progress(pct, done, total, eta_sec, elapsed, rate)), gr.update())
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
                listener_map = rsa_result.get("listener", {})
                speaker_map = rsa_result.get("speaker", {})

                polarity_map = rsa_state_val.get("polarity_map", {})
                topic_map = rsa_state_val.get("topic_map", {})
                most_common_text = format_common_themes(
                    sentence_lists, polarity_map, topic_map,
                    speaker=speaker_map, uniqueness=uniqueness, listener=listener_map,
                )

                if uniqueness:
                    divergent_per_review = format_divergent_cards(uniqueness, sentence_lists, listener_map, speaker_map)
                else:
                    divergent_per_review = {}

                show_agreement = current_focus in ("Agreement", "Agreement (Processing)")
                num_reviews = len(active_texts)

                rebuttal_str = rsa_state_val.get("rebuttal_str", "")
                rebuttal_htmls = [format_rebuttal_for_review(rebuttal_str, i + 1) for i in range(MAX_INTERACTIVE_REVIEWS)]

                agree_out = []
                for i in range(MAX_INTERACTIVE_REVIEWS):
                    if i < len(sentence_lists):
                        html_val = build_review_card(
                            f"Agreement in Review {i + 1}",
                            sentences=sentence_lists[i],
                            uniqueness=uniqueness, listener=listener_map, speaker=speaker_map,
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

            # --- Toggle display mode ---
            def toggle_display_mode(focus, active_count):
                effective_focus = focus.split(" ⏳")[0] if " ⏳" in focus else focus
                if effective_focus == "Agreement (Processing)":
                    effective_focus = "Agreement"

                updates = []
                for mode in ["No Highlighting", "Polarity", "Topic", "Agreement"]:
                    for i in range(MAX_INTERACTIVE_REVIEWS):
                        updates.append(gr.update(visible=(mode == effective_focus and i < active_count)))

                show_opinions = effective_focus == "Agreement" and focus != "Agreement (Processing)"
                updates.append(gr.update(visible=False))   # most_divergent
                updates.append(gr.update(visible=show_opinions))  # most_common

                if effective_focus == "Polarity":
                    updates.append(gr.update(visible=True, value=POLARITY_LEGEND))
                elif effective_focus == "Topic":
                    updates.append(gr.update(visible=True, value=TOPIC_LEGEND))
                else:
                    updates.append(gr.update(visible=False, value=""))

                return tuple(updates)

            # ================================================================
            # CALLBACK CHAINS
            # ================================================================

            def _fetch_with_logging(link):
                """Wrap fetch to log success/failure."""
                try:
                    result = fetch_openreview_reviews(link)
                    logger.log("fetch_success", tab="int", source="system",
                               payload={"num_reviews": sum(1 for r in result[:6] if r and r.strip())})
                    return result
                except Exception:
                    logger.log("fetch_failure", tab="int", source="system")
                    raise

            if highlights:
                # --- Full condition: fetch chain ---
                fetch_reviews_button.click(
                    fn=_validate_and_start_fetch,
                    inputs=[openreview_link_input],
                    outputs=[status_html, fetch_reviews_button]
                ).success(
                    fn=_fetch_with_logging,
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
                    outputs=_show_raw_outputs
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
                    fn=toggle_display_mode,
                    inputs=[focus_radio, interactive_review_count],
                    outputs=_toggle_outputs
                ).success(
                    fn=lambda: gr.update(interactive=True),
                    inputs=[],
                    outputs=[fetch_reviews_button]
                ).success(
                    fn=compute_rsa_in_background,
                    inputs=[rsa_computation_state, focus_radio],
                    outputs=_rsa_outputs
                )

                # --- Full condition: submit chain ---
                def _log_and_disable_submit():
                    logger.log("manual_process_submit", tab="int", source="user")
                    return gr.update(interactive=False)

                submit_button.click(
                    fn=_log_and_disable_submit,
                    inputs=[],
                    outputs=[submit_button]
                ).success(
                    fn=_show_raw_and_switch,
                    inputs=[review1_textbox, review2_textbox, review3_textbox, review4_textbox, review5_textbox, review6_textbox,
                            paste_rebuttal, no_title_state],
                    outputs=_show_raw_outputs
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
                    fn=toggle_display_mode,
                    inputs=[focus_radio, interactive_review_count],
                    outputs=_toggle_outputs
                ).success(
                    fn=lambda: gr.update(interactive=True),
                    inputs=[],
                    outputs=[submit_button]
                ).success(
                    fn=compute_rsa_in_background,
                    inputs=[rsa_computation_state, focus_radio],
                    outputs=_rsa_outputs
                )

                # --- Focus radio change (full only) ---
                def _log_int_mode_change(focus, act_count):
                    logger.log("mode_change", tab="int", source="user", visible_mode=focus)
                    return toggle_display_mode(focus, act_count)

                focus_radio.change(
                    fn=_log_int_mode_change,
                    inputs=[focus_radio, interactive_review_count],
                    outputs=_toggle_outputs
                )

            else:
                # --- No-highlight condition: fetch chain (simplified) ---

                fetch_reviews_button.click(
                    fn=_validate_and_start_fetch,
                    inputs=[openreview_link_input],
                    outputs=[status_html, fetch_reviews_button]
                ).success(
                    fn=_fetch_with_logging,
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
                    outputs=_show_raw_outputs
                ).success(
                    fn=lambda: gr.update(interactive=True),
                    inputs=[],
                    outputs=[fetch_reviews_button]
                )

                # --- No-highlight condition: submit chain (simplified) ---
                def _log_manual_submit():
                    logger.log("manual_process_submit", tab="int", source="user")
                    return gr.update(interactive=False)

                submit_button.click(
                    fn=_log_manual_submit,
                    inputs=[],
                    outputs=[submit_button]
                ).success(
                    fn=_show_raw_and_switch,
                    inputs=[review1_textbox, review2_textbox, review3_textbox, review4_textbox, review5_textbox, review6_textbox,
                            paste_rebuttal, no_title_state],
                    outputs=_show_raw_outputs
                ).success(
                    fn=lambda: gr.update(interactive=True),
                    inputs=[],
                    outputs=[submit_button]
                )

            # --- Shared callbacks (both conditions) ---

            def _back_to_input():
                logger.log("back_to_input", tab="int", source="user")
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                )

            back_to_input_btn.click(
                fn=_back_to_input,
                inputs=[],
                outputs=[input_section, results_section, back_to_input_btn, paper_title, view_results_btn]
            )

            def _view_results():
                logger.log("view_results", tab="int", source="user")
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                )

            view_results_btn.click(
                fn=_view_results,
                inputs=[],
                outputs=[input_section, results_section, back_to_input_btn, paper_title, view_results_btn]
            )

            # Clear button
            def _clear():
                logger.log("clear", tab="int", source="user")
                return (
                    "", "", "", "", "", "",
                    "",
                    3,
                    "", "",
                    *([""] * MAX_INTERACTIVE_REVIEWS),
                    *([""] * MAX_INTERACTIVE_REVIEWS),
                    *([""] * MAX_INTERACTIVE_REVIEWS),
                    *([""] * MAX_INTERACTIVE_REVIEWS),
                    {},
                )

            clear_button.click(
                fn=_clear,
                inputs=[],
                outputs=[
                    review1_textbox, review2_textbox, review3_textbox, review4_textbox, review5_textbox, review6_textbox,
                    paste_rebuttal,
                    interactive_review_count,
                    most_common, most_divergent,
                    *none_texts, *agreement_texts, *polarity_texts, *topic_texts,
                    rsa_computation_state,
                ]
            ).then(
                fn=lambda: (
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False, value=""),
                    *[gr.update(visible=False, value="") for _ in range(MAX_INTERACTIVE_REVIEWS)],
                    gr.update(visible=False, value=""),
                    gr.update(visible=False, value=""),
                    gr.update(visible=False, value=""),
                    gr.update(visible=False, value=""),
                ),
                inputs=[],
                outputs=[
                    review4_textbox, review5_textbox, review6_textbox,
                    interactive_rebuttal_toggle,
                    *rebuttal_for_reviews,
                    interactive_rebuttal_display, paper_title,
                    polarity_progress_html, agreement_progress_html,
                ]
            )

            # Add Review button
            def add_review(count):
                logger.log("add_review", tab="int", source="user",
                           payload={"new_count": min(count + 1, MAX_INTERACTIVE_REVIEWS)})
                new_count = min(count + 1, MAX_INTERACTIVE_REVIEWS)
                vis = [gr.update(visible=(i + 4 <= new_count)) for i in range(MAX_INTERACTIVE_REVIEWS - 3)]
                return (new_count, *vis)

            add_review_btn.click(
                fn=add_review,
                inputs=[interactive_review_count],
                outputs=[interactive_review_count, review4_textbox, review5_textbox, review6_textbox]
            )

        # ==============================================================
        # JS EVENT SINK CALLBACK (logging HTML-only controls)
        # ==============================================================
        if config.logging_enabled:
            def _handle_js_event(event_json):
                """Parse JS bridge events and log them."""
                if not event_json or not event_json.strip():
                    return ""
                try:
                    data = _json.loads(event_json)
                    event_type = data.get("event", "unknown")
                    payload_str = data.get("payload", "{}")
                    try:
                        payload = _json.loads(payload_str) if isinstance(payload_str, str) else payload_str
                    except (ValueError, TypeError):
                        payload = {"raw": payload_str}
                    logger.log(event_type, tab="", source="user", payload=payload)
                except Exception:
                    pass
                return ""

            js_event_sink.change(
                fn=_handle_js_event,
                inputs=[js_event_sink],
                outputs=[js_event_sink],
            )

        # ==============================================================
        # ON-LOAD: populate pre-processed tab + log session start
        # ==============================================================
        def _on_load(s):
            logger.log("session_load", source="system",
                       payload={"condition": config.condition,
                                "highlights_enabled": highlights})
            return update_review_display(s, "No Highlighting")

        demo.load(
            fn=_on_load,
            inputs=[state],
            outputs=_review_outputs,
        )

    return demo
