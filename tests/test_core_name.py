"""
tests/test_core_name.py

Unit tests for src.blocking.core_name.
"""
import pandas as pd
import pytest

from src.blocking.core_name import core_name_block, extract_core_name


def test_extract_core_name():
    assert extract_core_name("abc private limited") == "abc"
    assert extract_core_name("abc pvt ltd") == "abc"
    assert extract_core_name("abc sarl") == "abc"
    assert extract_core_name("prime money llc") == "prime money"
    assert extract_core_name("custom wealth services inc") == "custom wealth"
    assert extract_core_name("acme corp") == "acme"
    assert extract_core_name("") == ""


def test_core_name_block():
    s1_df = pd.DataFrame([
        {"entity_id": "S1-001", "business_name_normalized": "abc private limited", "country_normalized": "in"},
        {"entity_id": "S1-002", "business_name_normalized": "prime money llc", "country_normalized": "us"},
    ])

    s2_df = pd.DataFrame([
        {"entity_id": "S2-101", "business_name_normalized": "abc pvt ltd", "country_normalized": "in"},
        {"entity_id": "S2-102", "business_name_normalized": "prime money incorporated", "country_normalized": "us"},
        {"entity_id": "S2-103", "business_name_normalized": "abc ltd", "country_normalized": "us"}, # diff country
    ])

    res = core_name_block(s1_df, s2_df, cand_source="S2", country_aware=True)
    pairs = set(zip(res["s1_entity_id"], res["candidate_entity_id"]))

    assert ("S1-001", "S2-101") in pairs
    assert ("S1-002", "S2-102") in pairs
    assert ("S1-001", "S2-103") not in pairs
