"""
tests/test_tfidf_blocking.py

Unit tests for src.blocking.tfidf_blocking.
"""
import pandas as pd
import pytest

from src.blocking.tfidf_blocking import tfidf_block


def test_tfidf_retrieval_similar_strings():
    s1_df = pd.DataFrame([
        {
            "entity_id": "S1-001",
            "business_name_normalized": "orelee s barbershop",
            "business_address_normalized": "1795 westchester drive high point nc",
            "country_normalized": "us",
        }
    ])

    s2_df = pd.DataFrame([
        {
            "entity_id": "S2-101",
            "business_name_normalized": "orelee barbershop",
            "business_address_normalized": "1795 westchester dr",
            "country_normalized": "us",
        },
        {
            "entity_id": "S2-102",
            "business_name_normalized": "completely unrelated bakery",
            "business_address_normalized": "999 main st",
            "country_normalized": "us",
        },
    ])

    res = tfidf_block(s1_df, s2_df, cand_source="S2", top_k=5, country_aware=True)
    assert not res.empty
    assert "S2-101" in list(res["candidate_entity_id"])
    row = res[res["candidate_entity_id"] == "S2-101"].iloc[0]
    assert row["tfidf_rank"] == 1
    assert row["tfidf_similarity"] > 0.5


def test_tfidf_respects_top_k():
    s1_df = pd.DataFrame([
        {"entity_id": "S1-001", "business_name_normalized": "acme supply", "business_address_normalized": "1st ave", "country_normalized": "us"}
    ])

    s2_df = pd.DataFrame([
        {"entity_id": f"S2-{i:03d}", "business_name_normalized": "acme supply", "business_address_normalized": f"{i} ave", "country_normalized": "us"}
        for i in range(10)
    ])

    res = tfidf_block(s1_df, s2_df, cand_source="S2", top_k=3, country_aware=True)
    assert len(res) <= 3
