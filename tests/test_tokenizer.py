"""
tests/test_tokenizer.py

Unit tests for src.preprocessing.tokenizer.
"""
import json

import pytest

from src.preprocessing.tokenizer import (
    extract_numbers,
    extract_numbers_to_json,
    tokenize,
    tokenize_to_json,
)


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_empty_string(self):
        assert tokenize("") == []

    def test_single_word(self):
        assert tokenize("hello") == ["hello"]

    def test_two_words(self):
        assert tokenize("orelees barbershop") == ["orelees", "barbershop"]

    def test_address_tokens(self):
        result = tokenize("1795 westchester drive high point nc")
        assert result == ["1795", "westchester", "drive", "high", "point", "nc"]

    def test_number_token(self):
        result = tokenize("2100 cameron drive unit apartment g dundalk md")
        assert "2100" in result
        assert "g" in result  # single-char token must survive

    def test_b2b_alphanumeric(self):
        result = tokenize("b2b solutions")
        assert "b2b" in result

    def test_no_empty_tokens(self):
        # Even with leading/trailing spaces (shouldn't happen after normalization)
        result = tokenize("  hello  world  ")
        assert "" not in result

    def test_numbers_only(self):
        result = tokenize("12345")
        assert result == ["12345"]

    def test_multiple_numbers(self):
        result = tokenize("17560 ellis road")
        assert result[0] == "17560"

    def test_unicode_tokens(self):
        # Arabic text -- each "word" is a token
        result = tokenize("\u0634\u0631\u0643\u0629 \u0645\u062d\u062f\u0648\u062f\u0629")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# tokenize_to_json
# ---------------------------------------------------------------------------


class TestTokenizeToJson:
    def test_empty_returns_empty_array(self):
        assert tokenize_to_json("") == "[]"

    def test_valid_json(self):
        result = tokenize_to_json("hello world")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed == ["hello", "world"]

    def test_numbers_preserved_as_strings(self):
        result = tokenize_to_json("1795 westchester drive")
        parsed = json.loads(result)
        assert "1795" in parsed
        assert isinstance(parsed[0], str)  # must be string, not int

    def test_unicode_not_escaped(self):
        # Non-ASCII characters should appear as-is, not as \uXXXX
        result = tokenize_to_json("\u516c\u53f8 \u54c1\u724c")  # CJK
        assert "\\u" not in result  # ensure_ascii=False

    def test_round_trip(self):
        original = "b2b solutions 2024"
        tokens = tokenize(original)
        json_str = tokenize_to_json(original)
        parsed = json.loads(json_str)
        assert parsed == tokens

    def test_single_token(self):
        result = tokenize_to_json("us")
        assert json.loads(result) == ["us"]


# ---------------------------------------------------------------------------
# extract_numbers
# ---------------------------------------------------------------------------


class TestExtractNumbers:
    def test_empty_string(self):
        assert extract_numbers("") == []

    def test_no_numbers(self):
        assert extract_numbers("Orelee's Barbershop") == []

    def test_single_number(self):
        assert extract_numbers("1795 Westchester Drive") == ["1795"]

    def test_multiple_numbers(self):
        result = extract_numbers("2100 Cameron Drive, Unit Apt 4B, Zip 21222")
        assert result == ["2100", "4", "21222"]

    def test_alphanumeric_isolated_digits(self):
        # 7-Eleven, 3M, B2B, 7-11
        assert extract_numbers("7-Eleven Store #402") == ["7", "402"]
        assert extract_numbers("7-11 Store #402") == ["7", "11", "402"]
        assert extract_numbers("3M Innovation Center") == ["3"]
        assert extract_numbers("B2B Solutions") == ["2"]

    def test_preserves_order_and_duplicates(self):
        result = extract_numbers("100 Main Street Suite 100")
        assert result == ["100", "100"]


# ---------------------------------------------------------------------------
# extract_numbers_to_json
# ---------------------------------------------------------------------------


class TestExtractNumbersToJson:
    def test_empty_returns_empty_array(self):
        assert extract_numbers_to_json("") == "[]"

    def test_no_numbers_returns_empty_array(self):
        assert extract_numbers_to_json("Hello World") == "[]"

    def test_valid_json(self):
        result = extract_numbers_to_json("1795 Westchester Drive Apt 4")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed == ["1795", "4"]

    def test_round_trip(self):
        text = "2100 Cameron Drive Apt 4B 21222"
        nums = extract_numbers(text)
        json_str = extract_numbers_to_json(text)
        parsed = json.loads(json_str)
        assert parsed == nums

