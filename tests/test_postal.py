"""
tests/test_postal.py

Unit tests for src.blocking.postal.
"""
import pandas as pd
import pytest

from src.blocking.postal import extract_postal_code, postal_block


def test_extract_postal_code():
    assert extract_postal_code("1795 westchester drive high point nc 27262") == "27262"
    assert extract_postal_code("mumbai maharashtra 400001 india") == "400001"
    assert extract_postal_code("no postal here", '["27262"]') == "27262"
    assert extract_postal_code("no postal here", "[]") == ""


def test_postal_block():
    s1_df = pd.DataFrame([
        {"entity_id": "S1-001", "business_address_normalized": "high point nc 27262", "country_normalized": "us", "row_numbers": '["27262"]'},
        {"entity_id": "S1-002", "business_address_normalized": "mumbai 400001", "country_normalized": "in", "row_numbers": '["400001"]'},
    ])

    s2_df = pd.DataFrame([
        {"entity_id": "S2-101", "business_address_normalized": "different street 27262", "country_normalized": "us", "row_numbers": '["27262"]'},
        {"entity_id": "S2-102", "business_address_normalized": "delhi 110001", "country_normalized": "in", "row_numbers": '["110001"]'},
        {"entity_id": "S2-103", "business_address_normalized": "mumbai 400001", "country_normalized": "in", "row_numbers": '["400001"]'},
    ])

    res = postal_block(s1_df, s2_df, cand_source="S2", country_aware=True)
    pairs = set(zip(res["s1_entity_id"], res["candidate_entity_id"]))

    assert ("S1-001", "S2-101") in pairs
    assert ("S1-002", "S2-103") in pairs
    assert ("S1-002", "S2-102") not in pairs
