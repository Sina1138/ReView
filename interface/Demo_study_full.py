"""ReView Interface — FULL study condition.

Launches the app with all highlight features enabled and JSONL interaction
logging to study/interaction_logs/.
"""

import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from interface.study_config import full_study_config
from interface.app_builder import build_review_app, get_interactive_processor

# Pre-load models at startup (full condition uses all ML models)
get_interactive_processor()

demo = build_review_app(full_study_config())
demo.launch(share=False, ssr_mode=False)
