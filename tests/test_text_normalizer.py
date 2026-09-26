"""
tests/test_text_normalizer.py

Unit tests for src.preprocessing.text_normalizer.
"""
import pytest

from src.preprocessing.text_normalizer import nfkd_lowercase, normalize, normalize_text


# ---------------------------------------------------------------------------
# nfkd_lowercase
# ---------------------------------------------------------------------------


class TestNFKDLowercase:
    def test_basic_lowercase(self):
        assert nfkd_lowercase("PRIME MONEY") == "prime money"

    def test_preserves_apostrophe(self):
        result = nfkd_lowercase("Orelee's Barbershop")
        assert "'" in result or "\u2019" in result
        assert result == result.lower()

    def test_none(self):
        assert nfkd_lowercase(None) == ""

    def test_float_nan(self):
        assert nfkd_lowercase(float("nan")) == ""

    def test_empty_string(self):
        assert nfkd_lowercase("") == ""

    def test_whitespace_only(self):
        assert nfkd_lowercase("   ") == ""

    def test_cafe_lowercased(self):
        result = nfkd_lowercase("Caf\u00e9")  # Café
        assert result.startswith("cafe")  # e + combining mark

    def test_numbers_preserved(self):
        result = nfkd_lowercase("17560 Ellis Road")
        assert "17560" in result

    def test_punctuation_preserved(self):
        result = nfkd_lowercase("B+ Retail Inc.")
        assert "+" in result
        assert "." in result


# ---------------------------------------------------------------------------
# normalize_text / normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    # ── Apostrophe / Punctuation separator handling ─────────────────────────

    def test_apostrophe_converted_to_space(self):
        # Section 6: "Orelee's Barbershop" -> "orelee s barbershop"
        assert normalize_text("Orelee's Barbershop") == "orelee s barbershop"
        assert normalize("Orelee's Barbershop") == "orelee s barbershop"

    def test_curly_apostrophe(self):
        assert normalize_text("Orelee\u2019s Barbershop") == "orelee s barbershop"

    def test_plus_replaced_by_space(self):
        # "B+ Retail Inc" -> "b retail inc"
        result = normalize_text("B+ Retail Inc.")
        assert result == "b retail inc"

    def test_slash_boundary_preserved(self):
        # "A/B" -> "a b"
        result = normalize_text("A/B")
        assert result == "a b"

    def test_hyphen_boundary_preserved(self):
        result = normalize_text("7-Eleven")
        assert result == "7 eleven"

    def test_comma_stripped(self):
        result = normalize_text("High Point, NC")
        assert result == "high point nc"

    def test_period_stripped(self):
        result = normalize_text("Inc.")
        assert result == "inc"

    # ── Case standardization ──────────────────────────────────────────────

    def test_all_uppercase(self):
        assert normalize_text("PRIME MONEY") == "prime money"

    def test_mixed_case(self):
        assert normalize_text("Prime Money LLC") == "prime money llc"

    # ── Numbers and Alphanumerics ──────────────────────────────────────────

    def test_address_number_preserved(self):
        result = normalize_text("17560 Ellis Road")
        assert "17560" in result

    def test_address_2100_preserved(self):
        result = normalize_text("2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD")
        assert "2100" in result
        assert "g" in result

    def test_mixed_alphanumerics_preserved(self):
        # Section 8 & 22: A12B, 500C, Unit7, B2, 12A
        result = normalize_text("A12B 500C Unit7 B2 12A")
        assert result == "a12b 500c unit7 b2 12a"

    def test_b2b_preserved(self):
        result = normalize_text("B2B Solutions")
        assert result == "b2b solutions"

    # ── Whitespace ────────────────────────────────────────────────────────

    def test_multiple_spaces_collapsed(self):
        assert normalize_text("Prime    Money") == "prime money"

    def test_leading_trailing_stripped(self):
        assert normalize_text("  Prime Money   ") == "prime money"

    def test_tab_collapsed(self):
        assert normalize_text("Prime\tMoney") == "prime money"

    # ── Missing values ────────────────────────────────────────────────────

    def test_none(self):
        assert normalize_text(None) == ""

    def test_float_nan(self):
        assert normalize_text(float("nan")) == ""

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_whitespace_only(self):
        assert normalize_text("   ") == ""

    # ── Accent stripping ─────────────────────────────────────────────────

    def test_cafe_accent_stripped(self):
        assert normalize_text("Caf\u00e9") == "cafe"

    def test_elite_accent_stripped(self):
        assert normalize_text("\u00c9lite") == "elite"

    # ── Non-Latin scripts ─────────────────────────────────────────────────

    def test_arabic_not_destroyed(self):
        result = normalize_text("\u0634\u0631\u0643\u0629")
        assert len(result) > 0

    def test_cjk_not_destroyed(self):
        result = normalize_text("\u516c\u53f8")
        assert len(result) > 0

    def test_devanagari_not_destroyed(self):
        result = normalize_text("\u0915\u0902\u092a\u0928\u0940")
        assert len(result) > 0

    # ── Legal suffixes kept ───────────────────────────────────────────────

    def test_llc_preserved(self):
        result = normalize_text("Custom Wealth Services LLC")
        assert "llc" in result

    def test_inc_preserved(self):
        result = normalize_text("Acme Corp Inc")
        assert "inc" in result
