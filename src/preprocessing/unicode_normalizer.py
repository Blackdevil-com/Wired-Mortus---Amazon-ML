"""
src/preprocessing/unicode_normalizer.py

Low-level Unicode utilities used by the preprocessing pipeline.

Responsibilities:
    - Safe coercion of any Python value (None, NaN, numbers, strings) to str.
    - Unicode NFKD decomposition (nfkd_normalize).
    - Stripping of Unicode non-spacing combining marks (accent removal).
"""
from __future__ import annotations

import math
import unicodedata


def safe_str(value: object) -> str:
    """
    Safely convert *value* to a plain string.

    - None -> ""
    - float("nan") -> ""
    - Everything else -> str(value)
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def nfkd_normalize(text: object) -> str:
    """
    Apply Unicode NFKD normalization to *text*.

    Handles missing / None / NaN values safely.

    Args:
        text: Input string or value.

    Returns:
        NFKD-normalized string, or "" for missing input.
    """
    s = safe_str(text)
    if not s:
        return ""
    return unicodedata.normalize("NFKD", s)


# Alias for backward compatibility
apply_nfkd = nfkd_normalize


def strip_combining_marks(text: str) -> str:
    """
    Remove Unicode non-spacing combining marks (category Mn).

    Applied after NFKD decomposition to strip accents from Latin letters
    (e.g., 'e' + U+0301 -> 'e') without destroying non-Latin scripts.
    """
    if not text:
        return ""
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
