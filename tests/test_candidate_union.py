"""
tests/test_candidate_union.py

Unit tests for src.blocking.candidate_union.
"""
import json
import pandas as pd
import pytest

from src.blocking.candidate_union import union_candidates


def test_union_deduplication_and_metadata_aggregation():
    # Channel 1: exact_name
    df_exact = pd.DataFrame([
        {
            "s1_entity_id": "S1-001",
            "candidate_entity_id": "S2-101",
            "candidate_source": "S2",
            "country": "us",
            "blocking_channel": "exact_name",
        },
        {
            "s1_entity_id": "S1-002",
            "candidate_entity_id": "S2-102",
            "candidate_source": "S2",
            "country": "us",
            "blocking_channel": "exact_name",
        },
    ])

    # Channel 2: tfidf (retrieves S2-101 again + new candidate S2-103)
    df_tfidf = pd.DataFrame([
        {
            "s1_entity_id": "S1-001",
            "candidate_entity_id": "S2-101",
            "candidate_source": "S2",
            "country": "us",
            "blocking_channel": "tfidf",
            "tfidf_rank": 1,
            "tfidf_similarity": 0.88,
        },
        {
            "s1_entity_id": "S1-001",
            "candidate_entity_id": "S2-103",
            "candidate_source": "S2",
            "country": "us",
            "blocking_channel": "tfidf",
            "tfidf_rank": 2,
            "tfidf_similarity": 0.72,
        },
    ])

    union_df = union_candidates([df_exact, df_tfidf])

    # S1-001 -> S2-101 should appear ONLY ONCE
    s1_001_pairs = union_df[union_df["s1_entity_id"] == "S1-001"]
    assert len(s1_001_pairs) == 2  # S2-101 and S2-103

    # Check S2-101 metadata
    row_101 = union_df[(union_df["s1_entity_id"] == "S1-001") & (union_df["candidate_entity_id"] == "S2-101")].iloc[0]
    channels = json.loads(row_101["blocking_channels"])
    assert "exact_name" in channels
    assert "tfidf" in channels
    assert row_101["blocking_channel_count"] == 2
    assert row_101["tfidf_rank"] == 1
    assert row_101["tfidf_similarity"] == 0.88

    # Check S2-102 metadata
    row_102 = union_df[union_df["candidate_entity_id"] == "S2-102"].iloc[0]
    assert json.loads(row_102["blocking_channels"]) == ["exact_name"]
    assert row_102["blocking_channel_count"] == 1
    assert pd.isna(row_102["tfidf_rank"]) or row_102["tfidf_rank"] is None
