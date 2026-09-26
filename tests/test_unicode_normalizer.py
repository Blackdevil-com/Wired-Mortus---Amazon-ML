"""
tests/test_unicode_normalizer.py

Unit tests for src.preprocessing.unicode_normalizer.
"""
import math
import unicodedata

import pytest

from src.preprocessing.unicode_normalizer import (
    apply_nfkd,
    nfkd_normalize,
    safe_str,
    strip_combining_marks,
)



# ---------------------------------------------------------------------------
# safe_str
# ---------------------------------------------------------------------------


class TestSafeStr:
    def test_none_returns_empty(self):
        assert safe_str(None) == ""

    def test_float_nan_returns_empty(self):
        assert safe_str(float("nan")) == ""

    def test_math_nan_returns_empty(self):
        assert safe_str(math.nan) == ""

    def test_empty_string_unchanged(self):
        assert safe_str("") == ""

    def test_normal_string_unchanged(self):
        assert safe_str("hello") == "hello"

    def test_integer_converted(self):
        assert safe_str(42) == "42"

    def test_string_nan_NOT_converted(self):
        # The string literal "nan" is a valid value -- must NOT be mapped to ""
        assert safe_str("nan") == "nan"

    def test_string_none_NOT_converted(self):
        assert safe_str("none") == "none"

    def test_whitespace_string_preserved(self):
        # safe_str does NOT strip; that is the caller's responsibility
        assert safe_str("  hello  ") == "  hello  "

    def test_unicode_string_preserved(self):
        assert safe_str("Cafe\u0301") == "Cafe\u0301"


# ---------------------------------------------------------------------------
# apply_nfkd
# ---------------------------------------------------------------------------


class TestApplyNFKD:
    def test_empty(self):
        assert apply_nfkd("") == ""

    def test_ascii_unchanged(self):
        assert apply_nfkd("hello world") == "hello world"

    def test_cafe_decomposed(self):
        # "e with acute" decomposes into "e" + combining acute
        result = apply_nfkd("Caf\u00e9")  # Café
        assert result[3] == "e"
        assert unicodedata.category(result[4]) == "Mn"

    def test_fullwidth_mapped(self):
        # Fullwidth Latin letter should map to ASCII equivalent
        result = apply_nfkd("\uff21")  # Fullwidth A
        assert result == "A"

    def test_fi_ligature_decomposed(self):
        result = apply_nfkd("\ufb01")  # fi ligature
        assert result == "fi"

    def test_numbers_preserved(self):
        assert apply_nfkd("17560") == "17560"

    def test_result_is_nfkd(self):
        text = "Caf\u00e9 \u00c9lite"
        result = apply_nfkd(text)
        assert unicodedata.is_normalized("NFKD", result)


class TestNFKDNormalize:
    def test_nfkd_normalize_empty_and_none(self):
        assert nfkd_normalize("") == ""
        assert nfkd_normalize(None) == ""
        assert nfkd_normalize(float("nan")) == ""

    def test_nfkd_normalize_ascii_and_unicode(self):
        assert nfkd_normalize("Hello World") == "Hello World"
        assert nfkd_normalize("Caf\u00e9") == apply_nfkd("Caf\u00e9")



# ---------------------------------------------------------------------------
# strip_combining_marks
# ---------------------------------------------------------------------------


class TestStripCombiningMarks:
    def test_empty(self):
        assert strip_combining_marks("") == ""

    def test_ascii_unchanged(self):
        assert strip_combining_marks("hello") == "hello"

    def test_removes_acute(self):
        # e + combining acute -> e
        text = "e\u0301"
        assert strip_combining_marks(text) == "e"

    def test_cafe_after_nfkd(self):
        nfkd = apply_nfkd("Caf\u00e9")  # NFKD("Café")
        result = strip_combining_marks(nfkd)
        assert result == "Cafe"

    def test_numbers_unchanged(self):
        assert strip_combining_marks("12345") == "12345"

    def test_cjk_unchanged(self):
        # CJK ideographs carry no Mn marks
        assert strip_combining_marks("\u516c\u53f8") == "\u516c\u53f8"  # 公司
