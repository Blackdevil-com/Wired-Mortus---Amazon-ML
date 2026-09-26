"""
tests/test_exact_name.py

Unit tests for src.blocking.exact_name.
"""
import pandas as pd
import pytest

from src.blocking.exact_name import exact_name_block


@pytest.fixture
def sample_s1():
    return pd.DataFrame([
        {"entity_id": "S1-001", "business_name_normalized": "prime money", "country_normalized": "us"},
        {"entity_id": "S1-002", "business_name_normalized": "orelee s barbershop", "country_normalized": "us"},
        {"entity_id": "S1-003", "business_name_normalized": "tata motors", "country_normalized": "in"},
        {"entity_id": "S1-004", "business_name_normalized": "", "country_normalized": "us"},
    ])


@pytest.fixture
def sample_s2():
    return pd.DataFrame([
        {"entity_id": "S2-101", "business_name_normalized": "prime money", "country_normalized": "us"},
        {"entity_id": "S2-102", "business_name_normalized": "tata motors", "country_normalized": "us"}, # different country!
        {"entity_id": "S2-103", "business_name_normalized": "tata motors", "country_normalized": "in"},
    ])


def test_exact_name_match(sample_s1, sample_s2):
    res = exact_name_block(sample_s1, sample_s2, cand_source="S2", country_aware=True)
    pairs = set(zip(res["s1_entity_id"], res["candidate_entity_id"]))

    assert ("S1-001", "S2-101") in pairs
    assert ("S1-003", "S2-103") in pairs
    # S1-003 (IN) should NOT match S2-102 (US) when country_aware is True
    assert ("S1-003", "S2-102") not in pairs
    assert ("S1-002", "S2-101") not in pairs


def test_exact_name_not_country_aware(sample_s1, sample_s2):
    res = exact_name_block(sample_s1, sample_s2, cand_source="S2", country_aware=False)
    pairs = set(zip(res["s1_entity_id"], res["candidate_entity_id"]))

    # When country_aware is False, both US and IN match tata motors
    assert ("S1-003", "S2-102") in pairs
    assert ("S1-003", "S2-103") in pairs


def test_exact_name_empty_cases():
    empty_df = pd.DataFrame(columns=["entity_id", "business_name_normalized", "country_normalized"])
    res = exact_name_block(empty_df, empty_df, cand_source="S2")
    assert res.empty
