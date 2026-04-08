"""JSONL interaction logger for ReView study conditions."""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from interface.study_config import StudyConfig


class InteractionLogger:
    """Append-only JSONL logger for study interaction events.

    One file per session. Thread-safe. No-ops when logging is disabled.
    Never logs raw review or rebuttal text — metadata only.
    """

    def __init__(self, config: StudyConfig) -> None:
        self._enabled = config.logging_enabled
        self._condition = config.condition
        self._session_id = str(uuid.uuid4())
        self._lock = threading.Lock()
        self._file = None

        if not self._enabled:
            return

        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self._session_id}_{self._condition}_{ts}.jsonl"
        self._filepath = log_dir / filename
        self._file = open(self._filepath, "a", encoding="utf-8")

    @property
    def session_id(self) -> str:
        return self._session_id

    def log(
        self,
        event_type: str,
        tab: str = "",
        source: str = "user",
        visible_mode: str = "",
        paper_id: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one JSON record to the session log file."""
        if not self._enabled or self._file is None:
            return

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self._session_id,
            "condition": self._condition,
            "tab": tab,
            "event_type": event_type,
            "source": source,
            "visible_mode": visible_mode,
            "paper_id": paper_id,
            "payload": payload or {},
        }

        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        """Flush and close the log file."""
        if self._file is not None:
            with self._lock:
                self._file.close()
                self._file = None


# ---------------------------------------------------------------------------
# JS bridge: event delegation snippet for HTML-only controls
# ---------------------------------------------------------------------------

JS_EVENT_BRIDGE = r"""
(function() {
    /* Delegate clicks on elements with data-log-event to the hidden sink textbox */
    document.addEventListener('click', function(e) {
        var btn = e.target.closest('[data-log-event]');
        if (!btn) return;
        var sinkWrap = document.querySelector('#js-event-sink');
        if (!sinkWrap) return;
        var sink = sinkWrap.querySelector('textarea') || sinkWrap.querySelector('input');
        if (!sink) return;
        var ev = btn.getAttribute('data-log-event') || '';
        var pl = btn.getAttribute('data-log-payload') || '{}';
        var msg = JSON.stringify({event: ev, payload: pl, ts: Date.now()});
        var nativeSetter = Object.getOwnPropertyDescriptor(
            window.HTMLTextAreaElement.prototype, 'value'
        );
        if (!nativeSetter) {
            nativeSetter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, 'value'
            );
        }
        if (nativeSetter && nativeSetter.set) {
            nativeSetter.set.call(sink, msg);
        } else {
            sink.value = msg;
        }
        sink.dispatchEvent(new Event('input', {bubbles: true}));
    });
})();
"""
