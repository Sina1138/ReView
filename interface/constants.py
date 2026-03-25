"""
Shared constants for the ReView interface.

All magic numbers, CSS, display modes, progress HTML, and style constants
live here — imported by Demo.py and renderers.py.
"""
import html as _html

# ---------------------------------------------------------------------------
# Display modes
# ---------------------------------------------------------------------------
MODE_NONE = "No Highlighting"
MODE_POLARITY = "Polarity"
MODE_TOPIC = "Topic"
MODE_AGREEMENT = "Agreement"
DISPLAY_MODES = [MODE_NONE, MODE_POLARITY, MODE_TOPIC, MODE_AGREEMENT]

# ---------------------------------------------------------------------------
# RSA / agreement tuning
# ---------------------------------------------------------------------------
AGREEMENT_AMP_UNIQUE = 1.0   # exponent for positive scores (red = unique)
AGREEMENT_AMP_COMMON = 1.0   # exponent for negative scores (blue = common)
LISTENER_CONCENTRATION_THRESHOLD = 0.70  # Above this, listener "wins" over uniqueness
INFORMATIVENESS_MULTIPLIER = 2.0         # Multiplied by uniform baseline (1/K)

# ---------------------------------------------------------------------------
# Layout limits
# ---------------------------------------------------------------------------
MAX_PREPROCESSED_REVIEWS = 10  # Review/agreement/rebuttal slots in pre-processed tab
MAX_INTERACTIVE_REVIEWS = 6    # Expandable review slots in interactive tab

# ---------------------------------------------------------------------------
# Color maps (intentionally separate — each serves a different rendering context)
# ---------------------------------------------------------------------------
POLARITY_COLORS = {
    2: "#d4fcd6", 0: "#fcd6d6",        # integer keys (pre-processed tab)
    "➕": "#d4fcd6", "➖": "#fcd6d6",   # emoji keys (interactive tab)
}  # positive=green, negative=red

TOPIC_HTML_COLORS = {
    "Substance": "#b3e5fc",
    "Clarity": "#c8e6c9",
    "Soundness/Correctness": "#fff9c4",
    "Originality": "#f8bbd0",
    "Motivation/Impact": "#d1c4e9",
    "Meaningful Comparison": "#ffe0b2",
    "Replicability": "#b2dfdb",
}

# Interactive tab topic colors (lighter tints used for HighlightedText)
TOPIC_COLOR_MAP = {
    "Substance": "#cce0ff",
    "Clarity": "#e6ee9c",
    "Soundness/Correctness": "#ffcccc",
    "Originality": "#d1c4e9",
    "Motivation/Impact": "#b2ebf2",
    "Meaningful Comparison": "#fff9c4",
    "Replicability": "#c8e6c9",
}

# ---------------------------------------------------------------------------
# Reusable inline styles
# ---------------------------------------------------------------------------
TOGGLE_BTN_STYLE = (
    'background:none;border:1px solid #d1d5db;border-radius:6px;padding:4px 12px;'
    'font-size:0.78em;color:#6b7280;cursor:pointer;white-space:nowrap;'
    'line-height:1;height:28px;box-sizing:border-box;vertical-align:middle;'
    'display:inline-flex;align-items:center;justify-content:center;flex-shrink:0;'
)

REBUTTAL_PER_REVIEW_STYLE = (
    "margin-top:8px;margin-bottom:12px;border-radius:6px;overflow:hidden;"
    "border:1px solid #fde68a;background:#fffef5;"
)

REBUTTAL_GENERAL_STYLE = (
    "margin-top:16px;border-radius:8px;overflow:hidden;"
    "border:1px solid #fde68a;"
)

# ---------------------------------------------------------------------------
# Legend HTML (shared between pre-processed and interactive tabs)
# ---------------------------------------------------------------------------
POLARITY_LEGEND = (
    '<div style="display:flex;gap:12px;align-items:center;padding:8px 0;font-size:0.8em;">'
    '<span style="background:#d4fcd6;padding:2px 8px;border-radius:4px;">Positive</span>'
    '<span style="background:#fcd6d6;padding:2px 8px;border-radius:4px;">Negative</span>'
    '<span style="color:#9ca3af;">Neutral (no highlight)</span>'
    '</div>'
)

TOPIC_LEGEND = (
    '<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center;padding:8px 0;">'
    + " ".join(
        f'<span style="background:{color};padding:2px 8px;border-radius:4px;'
        f'font-size:0.8em;margin-right:4px;">{_html.escape(name)}</span>'
        for name, color in TOPIC_HTML_COLORS.items()
    )
    + '</div>'
)

# ---------------------------------------------------------------------------
# Example reviews (interactive tab defaults)
# ---------------------------------------------------------------------------
EXAMPLES = [
    "The paper gives really interesting insights on the topic of transfer learning. It is well presented and the experiment are extensive. I believe the authors missed Jane and al 2021. In addition, I think, there is a mistake in the math.",
    "The paper gives really interesting insights on the topic of transfer learning. It is well presented and the experiment are extensive. Some parts remain really unclear and I would like to see a more detailed explanation of the proposed method.",
    "The paper gives really interesting insights on the topic of transfer learning. It is not well presented and lack experiments. Some parts remain really unclear and I would like to see a more detailed explanation of the proposed method.",
]

# ---------------------------------------------------------------------------
# Progress / status HTML templates
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
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
