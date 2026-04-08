"""ReView Interface — default launcher (full features, no logging).

This is a thin wrapper around the shared app builder.
For study variants, use Demo_study_full.py or Demo_study_no_highlight.py.
"""

import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from interface.study_config import default_config
from interface.app_builder import build_review_app, get_interactive_processor

# Pre-load interactive processor models at startup so first request isn't slow
get_interactive_processor()

demo = build_review_app(default_config())
demo.launch(share=False, ssr_mode=False)
