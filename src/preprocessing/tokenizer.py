"""
src/preprocessing/tokenizer.py

Lightweight, Unicode-aware tokenizer for business names and addresses.

Design principles:
    - Tokenization happens AFTER normalization; input is already clean.
    - Split on whitespace only (simple and language-agnostic).
    - All non-empty substrings are kept as tokens.
    - Numbers are preserved as tokens (never removed).
    - Returns a JSON-serializable list for reliable storage in TSV cells.
    - Non-ASCII tokens are serialized without Unicode hex-escaping.
"""
from __future__ import annotations

import json
import re

# Non-whitespace run -- the basic "token" pattern.
_TOKEN_RE = re.compile(r"\S+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """
    Split a whitespace-normalised string into tokens.

    Input is assumed to already be the output of ``normalize()`` -- i.e.
    it has no leading/trailing whitespace and runs of whitespace have been
    collapsed to a single space.

    Args:
        text: A normalised string (output of text_normalizer.normalize).

    Returns:
        A list of string tokens.  Returns an empty list for empty input.

    Examples::

        tokenize("orelees barbershop")
        # => ['orelees', 'barbershop']

        tokenize("1795 westchester drive high point nc")
        # => ['1795', 'westchester', 'drive', 'high', 'point', 'nc']

        tokenize("b2b solutions")
        # => ['b2b', 'solutions']

        tokenize("")
        # => []
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text)


def tokenize_to_json(text: str) -> str:
    """
    Tokenize *text* and return the result as a compact JSON array string.

    The returned string is always valid JSON, parseable with ``json.loads()``.
    Non-ASCII characters are written without Unicode escaping so that the
    TSV remains human-readable.

    Args:
        text: A normalised string (may be empty).

    Returns:
        A compact JSON array such as ``'["orelees","barbershop"]'`` or ``'[]'``.

    Examples::

        tokenize_to_json("orelees barbershop")
        # => '["orelees","barbershop"]'

        tokenize_to_json("")
        # => '[]'
    """
    tokens = tokenize(text)
    return json.dumps(tokens, ensure_ascii=False, separators=(",", ":"))


# Numeric digit run pattern
_NUMBER_RE = re.compile(r"\d+", re.UNICODE)


def extract_numbers(text: str) -> list[str]:
    """
    Extract all numeric digit sequences from *text* in order of appearance.

    Works on raw, unnormalized, or normalized strings to guarantee that
    no digit information is lost.

    Args:
        text: Any text string (e.g. from row columns).

    Returns:
        A list of numeric string tokens. Empty list if no digits found.

    Examples::

        extract_numbers("1795 Westchester Drive Apt 4B, 27262")
        # => ['1795', '4', '27262']

        extract_numbers("7-Eleven #402")
        # => ['7', '11', '402']

        extract_numbers("No numbers here")
        # => []
    """
    if not text:
        return []
    return _NUMBER_RE.findall(text)


def extract_numbers_to_json(text: str) -> str:
    """
    Extract all numeric digit sequences and return as a compact JSON array string.

    Args:
        text: Any text string (may be empty or None).

    Returns:
        A compact JSON array such as ``'["1795","4","27262"]'`` or ``'[]'``.

    Examples::

        extract_numbers_to_json("1795 Westchester Drive Apt 4B")
        # => '["1795","4"]'

        extract_numbers_to_json("")
        # => '[]'
    """
    numbers = extract_numbers(text)
    return json.dumps(numbers, ensure_ascii=False, separators=(",", ":"))

