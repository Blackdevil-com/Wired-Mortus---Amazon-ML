"""
src/preprocessing/text_normalizer.py

Text normalization pipeline for entity resolution.

Responsibilities:
    - nfkd_lowercase: NFKD decomposition + lowercase (retaining punctuation)
    - normalize_text: Full normalization pipeline:
        Input -> Missing value handling -> Unicode NFKD -> Strip accents (Mn)
        -> Lowercase -> Replace punctuation/separators with whitespace
        -> Normalize repeated whitespace -> Strip leading/trailing whitespace
"""
from __future__ import annotations

import re

from .unicode_normalizer import nfkd_normalize, safe_str, strip_combining_marks

# Any character that is punctuation, symbol, or separator (including _)
_PUNCTUATION_RE = re.compile(r"[^\w\s]|_", re.UNICODE)

# Repeated whitespace collapse pattern
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)


def nfkd_lowercase(raw_value: object) -> str:
    """
    Apply Unicode NFKD normalization and lowercase conversion.

    Punctuation, separators, and numbers are preserved.
    Used for the `*_nfkd` output columns.

    Args:
        raw_value: Any Python object or string.

    Returns:
        NFKD-normalized lowercased string, or "" for missing input.
    """
    s = safe_str(raw_value).strip()
    if not s:
        return ""
    return nfkd_normalize(s).lower()


def normalize_text(raw_value: object) -> str:
    """
    Apply full Stage 1 text normalization.

    Pipeline:
        1. Handle missing / NaN / None values safely -> ""
        2. Unicode NFKD normalization
        3. Strip combining diacritical marks (Mn) for Latin characters
        4. Lowercase conversion
        5. Replace punctuation and separators with whitespace (never concatenate)
        6. Collapse multiple consecutive whitespaces into a single space
        7. Strip leading and trailing whitespace

    Examples:
        "Orelee's Barbershop" -> "orelee s barbershop"
        "B+ Retail Inc"       -> "b retail inc"
        "1795 Westchester Dr, High Point, NC" -> "1795 westchester dr high point nc"
        "2100 Cameron Drive, Unit APARTMENT G" -> "2100 cameron drive unit apartment g"
        "A12B, 500C, Unit7"   -> "a12b 500c unit7"
        None / float("nan")   -> ""

    Args:
        raw_value: Any raw input value.

    Returns:
        Clean normalized string.
    """
    s = safe_str(raw_value).strip()
    if not s:
        return ""

    # 1. Unicode NFKD
    text = nfkd_normalize(s)

    # 2. Strip diacritical marks (e.g. accented Latin characters)
    text = strip_combining_marks(text)

    # 3. Lowercase
    text = text.lower()

    # 4. Replace punctuation & separators with whitespace (preserves token boundaries)
    text = _PUNCTUATION_RE.sub(" ", text)

    # 5. Normalize repeated whitespace & strip
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


# Alias for backward compatibility
normalize = normalize_text
