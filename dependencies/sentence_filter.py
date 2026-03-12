"""
Centralized sentence filtering for RSA and polarity/topic scoring.

Filters out structural noise (headers, citations, timestamps, reference sections,
short fragments) so that only meaningful opinion sentences are scored and highlighted.
"""

import re
from typing import List, Optional

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
MIN_WORDS = 5  # Minimum word count for a sentence to be considered meaningful

HIGHLIGHT_THRESHOLD = 0.15  # Absolute score below which sentences get no color

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Standalone section headers (ported from interactive_processor._HEADER_RE)
_HEADER_RE = re.compile(
    r'^(\*{1,2}|#{1,3}\s*)?(summary|strengths?|weaknesses?|questions?|limitations?|minor|'
    r'rating|confidence|correctness|clarity|originality|significance|'
    r'pros?|cons?|comments?|suggestions?|conclusion|recommendation|'
    r'contribution|technical\s+quality|presentation|reproducibility|'
    r'novelty|experiments?|related\s+work|other|additional)\s*:?(\*{1,2})?$',
    re.IGNORECASE
)

# Header keyword prefix followed by actual content (e.g. "Paper Summary: This paper...")
_HEADER_PREFIX_RE = re.compile(
    r'^(\*{1,2}|#{1,3}\s*)?(paper\s+summary|summary|strengths?|weaknesses?|questions?|'
    r'limitations?|minor|comments?|suggestions?|conclusion|recommendation|'
    r'contribution|pros?|cons?|review\s+summary|overall\s+assessment)\s*:\s*(\*{1,2})?\s*',
    re.IGNORECASE
)

# Citation-only fragments: "Hu et al.:", "See et al., 2017:", "(Author et al., Year)"
_CITATION_ONLY_RE = re.compile(
    r'^\s*\(?\w[\w\s,\.\-]*?et\s+al\.?\s*[\),;:.]?\s*$',
    re.IGNORECASE
)

# Edit timestamps: "EDIT Nov. 20, 2019:", "UPDATE: ..."
_EDIT_TIMESTAMP_RE = re.compile(
    r'^(EDIT|UPDATE)\s+.{0,40}:\s*$',
    re.IGNORECASE
)

# Reference lines: "[1] Author...", "[12] ..."
_REFERENCE_LINE_RE = re.compile(r'^\[\d+\]')

# References section header
_REFERENCES_HEADER_RE = re.compile(
    r'^(\*{1,2}|#{1,3}\s*)?references?\s*:?\s*(\*{1,2})?$',
    re.IGNORECASE
)

# Rating/confidence metadata: "Rating: 6", "Confidence: 4/5", "Soundness: 3"
_RATING_RE = re.compile(
    r'^(rating|confidence|overall\s+score|soundness|presentation|contribution|'
    r'correctness|significance|originality|clarity)\s*:\s*\d',
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_section_header(sentence: str) -> bool:
    """Return True if sentence is a standalone structural section header."""
    return bool(_HEADER_RE.match(sentence.strip()))


def is_noise_sentence(sentence: str) -> bool:
    """
    Return True if the sentence is structural noise that should be excluded
    from RSA / polarity / topic scoring.

    Catches: standalone headers, citation-only fragments, edit timestamps,
    reference lines, rating metadata, and short fragments (< MIN_WORDS).
    """
    s = sentence.strip()
    if not s:
        return True

    # Standalone section header
    if _HEADER_RE.match(s):
        return True

    # Citation-only fragment
    if _CITATION_ONLY_RE.match(s):
        return True

    # Edit timestamp
    if _EDIT_TIMESTAMP_RE.match(s):
        return True

    # Reference line
    if _REFERENCE_LINE_RE.match(s):
        return True

    # References section header
    if _REFERENCES_HEADER_RE.match(s):
        return True

    # Rating/confidence metadata
    if _RATING_RE.match(s):
        return True

    # Too short to be a meaningful opinion
    if len(s.split()) < MIN_WORDS:
        return True

    return False


def strip_header_prefix(sentence: str) -> str:
    """
    Strip structural header prefixes from sentences that mix header + content.

    E.g. "Paper Summary: This paper proposes..." → "This paper proposes..."
    Returns the original sentence if no prefix is found.
    """
    s = sentence.strip()
    m = _HEADER_PREFIX_RE.match(s)
    if m:
        remainder = s[m.end():].strip()
        # Only strip if there's substantial content after the prefix
        if len(remainder.split()) >= MIN_WORDS:
            return remainder
    return s


def detect_references_start(sentences: List[str]) -> Optional[int]:
    """
    Return the index where the references section begins, or None.

    Heuristic: looks for a "References" header or the first `[1]`-style citation
    that is followed by more `[N]` lines (to avoid false positives on single
    bracketed numbers in review text).
    """
    for i, s in enumerate(sentences):
        stripped = s.strip()
        # Explicit "References" header
        if _REFERENCES_HEADER_RE.match(stripped):
            return i
        # First [1]-style line followed by at least one more [N] line
        if _REFERENCE_LINE_RE.match(stripped):
            following_refs = sum(
                1 for j in range(i + 1, min(i + 4, len(sentences)))
                if _REFERENCE_LINE_RE.match(sentences[j].strip())
            )
            if following_refs >= 1:
                return i
    return None


def filter_and_clean_sentences(sentences: List[str]) -> List[str]:
    """
    Full filtering pipeline: truncate at references, strip header prefixes,
    remove noise sentences.

    Args:
        sentences: Raw tokenized sentences from a single review or combined reviews.

    Returns:
        Cleaned sentence list ready for scoring.
    """
    # 1. Truncate at references section
    ref_start = detect_references_start(sentences)
    if ref_start is not None:
        sentences = sentences[:ref_start]

    # 2. Strip header prefixes and filter noise
    result = []
    for s in sentences:
        cleaned = strip_header_prefix(s)
        if not is_noise_sentence(cleaned):
            result.append(cleaned)

    return result
