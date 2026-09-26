"""
tests/test_house_street.py

Unit tests for src.blocking.house_street.
"""
import pandas as pd
import pytest

from src.blocking.house_street import extract_house_and_street, house_street_block


def test_extract_house_and_street():
    house, tokens = extract_house_and_street("1795 westchester drive high point nc")
    assert house == "1795"
    assert "westchester" in tokens
    assert "point" in tokens
    # Stopwords like drive must not be the only differentiator
    assert "drive" not in tokens

    house2, tokens2 = extract_house_and_street("2100 cameron drive unit apartment g")
    assert house2 == "2100"
    assert "cameron" in tokens2


def test_house_street_block():
    s1_df = pd.DataFrame([
        {
            "entity_id": "S1-001",
            "business_address_normalized": "1795 westchester drive high point nc",
            "country_normalized": "us",
            "row_numbers": '["1795"]',
        },
        {
            "entity_id": "S1-002",
            "business_address_normalized": "2100 cameron drive dundalk md",
            "country_normalized": "us",
            "row_numbers": '["2100"]',
        },
    ])

    s2_df = pd.DataFrame([
        {
            "entity_id": "S2-101",
            "business_address_normalized": "1795 westchester dr ste 200",
            "country_normalized": "us",
            "row_numbers": '["1795", "200"]',
        },
        {
            "entity_id": "S2-102",
            "business_address_normalized": "1795 broadway ave", # Same house, different street!
            "country_normalized": "us",
            "row_numbers": '["1795"]',
        },
        {
            "entity_id": "S2-103",
            "business_address_normalized": "2100 cameron dr unit g",
            "country_normalized": "us",
            "row_numbers": '["2100"]',
        },
    ])

    res = house_street_block(s1_df, s2_df, cand_source="S2", country_aware=True)
    pairs = set(zip(res["s1_entity_id"], res["candidate_entity_id"]))

    assert ("S1-001", "S2-101") in pairs
    assert ("S1-001", "S2-102") not in pairs  # different street tokens!
    assert ("S1-002", "S2-103") in pairs
