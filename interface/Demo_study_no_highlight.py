"""ReView Interface — NO-HIGHLIGHT study condition.

Launches the app with plain review display only (no polarity, topic, or
agreement computation). JSONL interaction logs written to
study/interaction_logs/.

No ML models are loaded — startup is faster.
"""

import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from interface.study_config import no_highlight_study_config
from interface.app_builder import build_review_app

# No model preloading — this condition does not use ML models.

demo = build_review_app(no_highlight_study_config())
demo.launch(share=False, ssr_mode=False)
