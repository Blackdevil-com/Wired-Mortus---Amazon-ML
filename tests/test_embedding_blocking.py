"""
tests/test_embedding_blocking.py

Unit tests for src.blocking.embedding_blocking.
"""
import pandas as pd
import pytest

from src.blocking.embedding_blocking import embedding_block


def test_embedding_retrieval_ranking():
    s1_df = pd.DataFrame([
        {
            "entity_id": "S1-001",
            "business_name_normalized": "prime money",
            "business_address_normalized": "17560 ellis road tahlequah ok",
            "country_normalized": "us",
        }
    ])

    s2_df = pd.DataFrame([
        {
            "entity_id": "S2-101",
            "business_name_normalized": "prime money",
            "business_address_normalized": "17560 ellis rd",
            "country_normalized": "us",
        },
        {
            "entity_id": "S2-102",
            "business_name_normalized": "flower boutique",
            "business_address_normalized": "55 rose avenue",
            "country_normalized": "us",
        },
    ])

    res = embedding_block(s1_df, s2_df, cand_source="S2", top_k=2, cache_embeddings=False, country_aware=True)
    assert not res.empty
    assert "S2-101" in list(res["candidate_entity_id"])
    assert "embedding_rank" in res.columns
    assert "embedding_similarity" in res.columns
