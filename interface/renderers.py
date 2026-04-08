"""
HTML rendering functions for the ReView interface.

All functions that produce HTML strings live here — imported by Demo.py.
Keeps Demo.py focused on Gradio layout and callbacks.
"""
import hashlib
import json
import math
import re as _re
import html as _html
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

from interface.constants import (
    AGREEMENT_AMP_UNIQUE,
    AGREEMENT_AMP_COMMON,
    LISTENER_CONCENTRATION_THRESHOLD,
    INFORMATIVENESS_MULTIPLIER,
    POLARITY_COLORS,
    TOPIC_HTML_COLORS,
    TOGGLE_BTN_STYLE,
    REBUTTAL_PER_REVIEW_STYLE,
    REBUTTAL_GENERAL_STYLE,
)
from dependencies.sentence_filter import (
    is_noise_sentence,
    HIGHLIGHT_THRESHOLD,
    compute_informativeness,
)


# ===== Utility helpers =====

def make_sentence_id(sentence: str) -> str:
    """Deterministic DOM ID for a sentence, used by click-to-scroll."""
    return "sent_" + hashlib.md5(sentence.encode("utf-8")).hexdigest()[:12]


def click_to_scroll_js(sent_id: str, color: str = "#3b82f6") -> str:
    """Return inline onclick JS for smooth-scroll + outline flash."""
    return (
        f"(function(){{var el=document.getElementById('{sent_id}');"
        f"if(el){{el.scrollIntoView({{behavior:'smooth',block:'center'}});"
        f"el.style.outline='3px solid {color}';"
        f"setTimeout(function(){{el.style.outline='';}},2500);}}}})();"
    )


def source_badges_html(sent: str, sentence_lists: list) -> str:
    """Return R# badge HTML for all reviews containing the sentence."""
    source = [r_idx + 1 for r_idx, sl in enumerate(sentence_lists) if sent in sl]
    return " ".join(
        f'<span style="background:#f3f4f6;color:#374151;padding:2px 6px;'
        f'border-radius:4px;font-size:0.72em;font-weight:600;">R{n}</span>'
        for n in source
    )


def listener_dist_bars(sent: str, listener: dict, source_badges: str,
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


# ===== Toggle / navigation buttons =====

def toggle_html(selector: str, text_when_all_open: str,
                text_when_not_all_open: str, initial_label: str,
                log_event: str = "") -> str:
    """Generate a toggle button for expanding/collapsing details elements."""
    log_attr = f' data-log-event="{log_event}"' if log_event else ""
    return (
        '<button onclick="'
        f"let tab=this.closest('.tabitem')||this.closest('.gradio-container');"
        f"let details=tab.querySelectorAll('{selector}');"
        "if(!details.length)return;"
        "let allOpen=Array.from(details).every(d=>d.open);"
        "details.forEach(d=>d.open=!allOpen);"
        f"this.textContent=allOpen?'{text_when_all_open}':'{text_when_not_all_open}';"
        f'" style="{TOGGLE_BTN_STYLE}"'
        f'{log_attr}'
        f'>{initial_label}</button>'
    )


def rebuttal_toggle_html() -> str:
    """Generate an Expand/Collapse All Responses toggle button with inline JS."""
    return toggle_html("details:not(.review-collapse)",
                       "Expand All Responses", "Collapse All Responses",
                       "Expand All Responses",
                       log_event="rebuttal_toggle")


def review_toggle_html() -> str:
    """Generate a Collapse/Expand All Reviews toggle button with inline JS."""
    return toggle_html("details.review-collapse",
                       "Expand All Reviews", "Collapse All Reviews",
                       "Collapse All Reviews",
                       log_event="review_toggle")


def jump_buttons_html(active_count: int, prefix: str = "int") -> str:
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
        payload_json = json.dumps({"review": i, "prefix": prefix}).replace('"', "&quot;")
        buttons.append(
            f'<button onclick="{js}" '
            f'data-log-event="jump_button" data-log-payload="{payload_json}" '
            f'style="{TOGGLE_BTN_STYLE}font-weight:600;">'
            f'R{i}</button>'
        )
    return "".join(buttons)


# ===== Paragraph / header detection =====

def should_break_before(sent: str) -> bool:
    """Detect if a paragraph break should be inserted before this sentence."""
    s = sent.strip()
    if _re.match(r'^[\(\[]?\d+[\)\]\.:]', s):
        return True
    if len(s) > 2 and s[0] in ('-', '•', '*', '–', '—') and s[1] == ' ':
        return True
    if s.startswith('##') or s.startswith('---'):
        return True
    if _re.match(
        r'^\*{0,2}(Rating|Strengths?|Weaknesses?|Questions?|Limitations?|Summary|'
        r'Soundness|Presentation|Contribution|Confidence|Experience|Review Assessment|'
        r'Recommendation|Overall|Minor|Major|Typos?|Suggestions?|Comments?|'
        r'Detailed\s+Comments?|Pros?|Cons?|Flag|Clarity|Significance|Originality)',
        s, _re.IGNORECASE,
    ):
        return True
    return False


def is_review_header(sent: str) -> bool:
    """Detect if a sentence is a review metadata header (Rating:, Experience:, etc.)."""
    return bool(_re.match(
        r'^\*{0,2}(Rating|Confidence|Experience|Review Assessment|Recommendation|Flag)\b',
        sent.strip(), _re.IGNORECASE,
    ))


# ===== Review card wrappers =====

def wrap_review_card(label: str, inner_html: str, collapsible: bool = True) -> str:
    """Wrap review content in a styled card with gray header."""
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


# ===== Core review renderers =====

def render_review_html(
    review_items: list,
    mode: str = "plain",
    label: str = "Review",
    wrap: bool = False,
) -> str:
    """Render a review as HTML with proper paragraph formatting.

    Args:
        review_items: list of (sentence, metadata_dict) tuples
        mode: "plain", "polarity", or "topic"
        label: header label
        wrap: if False, return bare content (caller handles outer wrapper)
    """
    if not review_items:
        return ""

    parts = ['<div style="line-height:1.8;font-size:0.95em;margin-top:6px;">']

    for i, (sent, metadata) in enumerate(review_items):
        if i > 0 and should_break_before(sent):
            parts.append('<br>')

        is_hdr = is_review_header(sent)
        bg = ""
        label_text = ""
        if mode == "polarity":
            polarity = metadata.get("polarity")
            if polarity in POLARITY_COLORS:
                bg = f"background:{POLARITY_COLORS[polarity]};"
        elif mode == "topic":
            topic = metadata.get("topic")
            if topic and topic != "NONE" and topic in TOPIC_HTML_COLORS:
                bg = f"background:{TOPIC_HTML_COLORS[topic]};"
                label_text = topic

        style = f"padding:1px 3px;border-radius:3px;{bg}"
        if is_hdr:
            style += "font-weight:600;color:#92400e;"

        sent_id = make_sentence_id(sent)
        escaped = _html.escape(sent)

        if label_text:
            parts.append(
                f'<span id="{sent_id}" style="{style}" title="{_html.escape(label_text)}">'
                f'{escaped} </span>'
            )
        else:
            parts.append(f'<span id="{sent_id}" style="{style}">{escaped} </span>')

    parts.append('</div>')
    content = "".join(parts)
    if wrap:
        return wrap_review_card(label, content, collapsible=True)
    elif label:
        return wrap_review_card(label, content, collapsible=False)
    return content


def _normalize_polarity(val) -> Optional[str]:
    """Normalize polarity from any format to 'positive'/'negative'/None."""
    if val == "➕" or val == 2 or val == "positive":
        return "positive"
    if val == "➖" or val == 0 or val == "negative":
        return "negative"
    return None


def format_common_themes(
    sentence_lists: list,
    polarity_map: dict,
    topic_map: dict,
    speaker: dict = None,
    uniqueness: dict = None,
    listener: dict = None,
) -> str:
    """Common Themes Across Reviews — groups sentences by topic, then polarity."""
    num_reviews = len(sentence_lists)
    if num_reviews < 2:
        return ""

    # Build per-sentence (topic, polarity) index
    topic_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r_idx, sl in enumerate(sentence_lists):
        for sent in sl:
            if should_break_before(sent) or is_review_header(sent):
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

    # --- RSA uniqueness filter: keep only common sentences ---
    if uniqueness:
        median_u = float(np.median(list(uniqueness.values())))
        for topic in list(topic_data.keys()):
            pol_dict = topic_data[topic]
            for pol in list(pol_dict.keys()):
                rev_dict = pol_dict[pol]
                for r_idx in list(rev_dict.keys()):
                    rev_dict[r_idx] = [
                        s for s in rev_dict[r_idx]
                        if uniqueness.get(s, 0.0) <= median_u
                    ]
                    if not rev_dict[r_idx]:
                        del rev_dict[r_idx]
                if not rev_dict:
                    del pol_dict[pol]
            if not pol_dict:
                del topic_data[topic]

    # Filter to topics with ≥2 unique reviewers
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

    has_sentiment = [t for t in common_topics
                     if any(p in t[3] for p in ("positive", "negative"))]
    if len(has_sentiment) >= 1:
        common_topics = has_sentiment

    common_topics.sort(key=lambda t: (0 if t[0] == "Other" else 1, t[1], t[2]), reverse=True)

    # Fallback: Generic Sentences
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
            sent_id = make_sentence_id(sent)
            badges = source_badges_html(sent, sentence_lists)
            dist_html = listener_dist_bars(sent, listener, badges)
            onclick = click_to_scroll_js(sent_id)
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

    # Render topic cards with polarity breakdown
    _pol_colors = {"negative": "#ef4444", "positive": "#22c55e", "neutral": "#9ca3af"}
    _pol_labels = {"negative": "Negative", "positive": "Positive", "neutral": "Neutral"}
    _pol_order = ["negative", "positive", "neutral"]

    def _pick_best(sents, r_idx):
        r_label = f"R{r_idx + 1}"
        if speaker and r_label in speaker:
            sp = speaker[r_label]
            scored = [
                (s, sp.get(s, 0.0), -(uniqueness.get(s, 0.0) if uniqueness else 0.0))
                for s in sents
            ]
            scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
            return scored[0][0]
        if uniqueness:
            return min(sents, key=lambda s: uniqueness.get(s, 0.0))
        return sents[0]

    def _sent_row(sent, r_idx):
        r_label = f"R{r_idx + 1}"
        sent_id = make_sentence_id(sent)
        onclick = click_to_scroll_js(sent_id)
        return (
            f'<div style="display:flex;align-items:baseline;gap:6px;padding:2px 0;'
            f'padding-left:8px;cursor:pointer;" '
            f'onclick="{_html.escape(onclick)}">'
            f'<span style="background:#f3f4f6;color:#374151;padding:1px 5px;'
            f'border-radius:3px;font-size:0.7em;font-weight:600;flex-shrink:0;">{r_label}</span>'
            f'<span style="color:#374151;font-size:0.85em;line-height:1.4;">{_html.escape(sent)}</span>'
            f'</div>'
        )

    _pol_colors_soft = {"negative": "#f87171", "positive": "#4ade80", "neutral": "#d1d5db"}

    cards = []
    for topic, n_reviewers, total_sents, pol_dict in common_topics:
        border_color = TOPIC_HTML_COLORS.get(topic, "#d1d5db")
        reviewer_text = (
            f"All {num_reviews} reviewers" if n_reviewers == num_reviews
            else f"{n_reviewers} of {num_reviews} reviewers"
        )

        pol_counts = {}
        for pol in _pol_order:
            if pol in pol_dict:
                pol_counts[pol] = sum(len(s) for s in pol_dict[pol].values())
        total = sum(pol_counts.values()) or 1

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

        header = (
            f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">'
            f'<span style="font-weight:600;font-size:0.88em;color:#1f2937;">{_html.escape(topic)}</span>'
            f'<span style="color:#9ca3af;font-size:0.75em;">·</span>'
            f'<span style="font-size:0.75em;color:#6b7280;">{reviewer_text}</span>'
            f'{polarity_bar}'
            f'</div>'
        )

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
    """Most Divergent Opinions — returns per-review HTML dict {review_index: html}."""
    if not uniqueness or not listener or not speaker:
        return {}

    num_reviews = len(speaker)
    if num_reviews == 0:
        return {}

    median_u = float(np.median(list(uniqueness.values())))
    review_labels = [f"R{i+1}" for i in range(num_reviews)]

    k = max(sum(len(v) for v in speaker.values()) // max(len(speaker), 1), 1)
    min_speaker_score = INFORMATIVENESS_MULTIPLIER / k

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
            sent_id = make_sentence_id(sent)
            context_before, context_after = _get_context(sent, sentence_lists)

            dom_pct = 0
            if sent in listener:
                dom_pct = int(round(max(listener[sent].values(), default=0.0) * 100))
            uniqueness_badge = (
                f'<span style="background:#fee2e2;color:#991b1b;padding:2px 6px;'
                f'border-radius:4px;font-size:0.7em;font-weight:600;display:inline-block;margin-bottom:3px;">'
                f'{dom_pct}% listener share</span>'
            )

            onclick = click_to_scroll_js(sent_id, "#ef4444")

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
    """Custom HTML renderer for Agreement mode (replaces gr.HighlightedText)."""
    if not sentences:
        return ""

    legend_html = (
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;'
        'font-size:0.75em;color:#6b7280;">'
        '<span style="background:linear-gradient(to right,rgba(59,130,246,0.7),rgba(209,213,219,0.3),rgba(239,68,68,0.7));'
        'width:120px;height:8px;border-radius:4px;display:inline-block;"></span>'
        '<span>← Common &nbsp;|&nbsp; Unique →</span>'
        '</div>'
    )

    parts = [legend_html, '<div style="line-height:1.8;font-size:0.95em;margin-top:6px;">']

    k = max(len(uniqueness), 1)
    info_threshold = INFORMATIVENESS_MULTIPLIER / k

    for idx, sent in enumerate(sentences):
        if idx > 0 and should_break_before(sent):
            parts.append('<br>')

        sent_id = make_sentence_id(sent)
        score = uniqueness.get(sent)
        header_style = "font-weight:600;color:#92400e;" if is_review_header(sent) else ""

        if score is None or abs(score) < HIGHLIGHT_THRESHOLD:
            parts.append(f'<span id="{sent_id}" style="{header_style}">{_html.escape(sent)} </span>')
            continue

        if score < 0:
            r, g, b = 59, 130, 246
            if listener and sent in listener:
                dist = listener[sent]
                max_prob = max(dist.values(), default=0.0)
                max_entropy = math.log(max(num_reviews, 2))
                entropy = sum(-p * math.log(p) for p in dist.values() if p > 0)
                if max_prob > LISTENER_CONCENTRATION_THRESHOLD:
                    opacity = 0.0
                else:
                    opacity = (entropy / max_entropy) ** AGREEMENT_AMP_COMMON if max_entropy > 0 else 0.0
            else:
                opacity = abs(score) ** AGREEMENT_AMP_COMMON

            if speaker:
                info = compute_informativeness(sent, speaker, num_reviews)
                if info < info_threshold:
                    opacity *= 0.3
        else:
            r, g, b = 239, 68, 68
            opacity = abs(score) ** AGREEMENT_AMP_UNIQUE

        bg_color = f"rgba({r},{g},{b},{opacity:.3f})"

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
    content = "".join(parts)
    if wrap:
        return wrap_review_card(label, content, collapsible=True)
    elif label:
        return wrap_review_card(label, content, collapsible=False)
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
    """Unified review card builder — single entry point for both tabs."""
    if sentences is not None:
        inner = render_agreement_html(
            sentences, uniqueness or {}, listener, speaker,
            num_reviews=num_reviews, label="",
        )
    elif review_items is not None:
        inner = render_review_html(review_items, mode=mode, label="")
    else:
        inner = ""
    return wrap_review_card(label, f"{inner}{divergent_html}{rebuttal_html}", collapsible=collapsible)


# ===== Status / progress HTML =====

def render_status(msg, kind="success"):
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


def fmt_time(sec: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS like tqdm."""
    if sec is None:
        return "?"
    sec = int(sec)
    if sec < 3600:
        return f"{sec // 60:02d}:{sec % 60:02d}"
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def render_agreement_progress(pct: int, done: int, total: int,
                            eta_sec: float = None, elapsed: float = None,
                            rate: float = None) -> str:
    """Progress bar HTML for the agreement computation."""
    if done > 0 and elapsed is not None:
        info = f"{done}/{total} [{fmt_time(elapsed)}<{fmt_time(eta_sec)}, {rate:.1f}s/it]"
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


# ===== Rebuttal formatting =====

def _parse_rebuttal_json(rebuttal: str) -> Optional[list]:
    """Parse rebuttal JSON string, returning list of items or None."""
    if not rebuttal or not rebuttal.strip():
        return None
    try:
        items = json.loads(rebuttal)
        return items if items else None
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None


def format_rebuttal_plain(text: str) -> str:
    """Format a plain text rebuttal as collapsible HTML."""
    if not text or not text.strip():
        return ""
    return (
        f'<details style="{REBUTTAL_PER_REVIEW_STYLE}">'
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
            f'<details style="{REBUTTAL_PER_REVIEW_STYLE}">'
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
            f'<details style="{REBUTTAL_PER_REVIEW_STYLE}">'
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
            f'<details style="{REBUTTAL_GENERAL_STYLE}">'
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
        f'<details style="{REBUTTAL_GENERAL_STYLE}">'
        f'<summary style="{HEADER_STYLE}cursor:pointer;list-style:none;">'
        f'<span style="font-size:1.1em;">💬</span>'
        f'<span style="{TITLE_STYLE}">General Author Response</span></summary>'
        f'<div style="padding:14px 16px;background:white;">'
        f'<div style="white-space:pre-wrap;color:#1f2937;font-size:0.9em;line-height:1.6;">{text}</div>'
        f'</div></details>'
    )
