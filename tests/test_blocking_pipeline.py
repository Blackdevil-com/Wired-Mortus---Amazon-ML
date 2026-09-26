"""
tests/test_blocking_pipeline.py

Integration tests for src.blocking.pipeline.run_stage1_blocking.
"""
import pandas as pd
import pytest

from src.blocking.config import BlockingConfig
from src.blocking.pipeline import run_stage1_blocking


@pytest.fixture
def sample_s1():
    return pd.DataFrame([
        {
            "entity_id": "S1-001",
            "business_name_normalized": "orelee s barbershop",
            "business_address_normalized": "1795 westchester drive high point nc",
            "country_normalized": "us",
            "row_numbers": '["1795"]',
        },
        {
            "entity_id": "S1-002",
            "business_name_normalized": "prime money llc",
            "business_address_normalized": "17560 ellis road tahlequah ok 74464",
            "country_normalized": "us",
            "row_numbers": '["17560", "74464"]',
        },
    ])


@pytest.fixture
def sample_s2():
    return pd.DataFrame([
        {
            "entity_id": "S2-101",
            "business_name_normalized": "orelee barbershop",
            "business_address_normalized": "1795 westchester dr",
            "country_normalized": "us",
            "row_numbers": '["1795"]',
        },
    ])


@pytest.fixture
def sample_s3():
    return pd.DataFrame([
        {
            "entity_id": "S3-201",
            "business_name_normalized": "prime money incorporated",
            "business_address_normalized": "17560 ellis rd 74464",
            "country_normalized": "us",
            "row_numbers": '["17560", "74464"]',
        },
    ])


def test_pipeline_integration_e2e(tmp_path, sample_s1, sample_s2, sample_s3):
    config = BlockingConfig(
        output_dir=str(tmp_path / "candidates"),
        cache_dir=str(tmp_path / "cache"),
        cache_embeddings=False,
        tfidf_top_k=5,
        embedding_top_k=5,
    )

    union_df, channel_dict = run_stage1_blocking(sample_s1, sample_s2, sample_s3, config=config)

    assert not union_df.empty
    assert (tmp_path / "candidates" / "stage1_candidate_union.tsv").exists()

    # S1-001 should find S2-101 (via house_street, tfidf, embedding, core_name)
    s1_001_cands = set(union_df[union_df["s1_entity_id"] == "S1-001"]["candidate_entity_id"])
    assert "S2-101" in s1_001_cands

    # S1-002 should find S3-201 (via core_name, postal, house_street, tfidf, embedding)
    s1_002_cands = set(union_df[union_df["s1_entity_id"] == "S1-002"]["candidate_entity_id"])
    assert "S3-201" in s1_002_cands
