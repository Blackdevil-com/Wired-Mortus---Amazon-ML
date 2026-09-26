"""
src/blocking/exact_name.py

Channel 1: Exact Normalized Name Blocking.
Generates candidate pairs when business_name_normalized matches exactly
(within the same country if country_aware is True).
"""
from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

CHANNEL_NAME = "exact_name"


def exact_name_block(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    cand_source: str,
    country_aware: bool = True,
) -> pd.DataFrame:
    """
    Generate candidate pairs where S1 and candidate entity have the exact same
    normalized business name.

    Args:
        s1_df: DataFrame of S1 query records (needs entity_id, business_name_normalized, country_normalized).
        cand_df: DataFrame of candidate records (S2 or S3).
        cand_source: Label of candidate source ('S2' or 'S3').
        country_aware: Whether to restrict matches to the same country.

    Returns:
        DataFrame of candidate pairs:
            - s1_entity_id
            - candidate_entity_id
            - candidate_source
            - country
            - blocking_channel (='exact_name')
    """
    logger.info("Running Channel [Exact Name Block] for S1 -> %s...", cand_source)

    if s1_df.empty or cand_df.empty:
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "candidate_source", "country", "blocking_channel"])

    # Build index on candidate side: key -> list of (candidate_entity_id, country)
    cand_index: dict[tuple[str, str], list[str]] = defaultdict(list)

    cand_name_col = "business_name_normalized"
    country_col = "country_normalized"

    for _, row in cand_df[[cand_name_col, country_col, "entity_id"]].iterrows():
        name = str(row[cand_name_col]).strip()
        country = str(row[country_col]).strip()
        if not name:
            continue
        key = (country, name) if country_aware else ("", name)
        cand_index[key].append(row["entity_id"])

    # Query using S1
    results: list[dict[str, str]] = []
    for _, row in s1_df[[cand_name_col, country_col, "entity_id"]].iterrows():
        name = str(row[cand_name_col]).strip()
        country = str(row[country_col]).strip()
        if not name:
            continue
        key = (country, name) if country_aware else ("", name)
        matched_cand_ids = cand_index.get(key, [])
        for cand_id in matched_cand_ids:
            results.append({
                "s1_entity_id": row["entity_id"],
                "candidate_entity_id": cand_id,
                "candidate_source": cand_source,
                "country": country,
                "blocking_channel": CHANNEL_NAME,
            })

    res_df = pd.DataFrame(results)
    if res_df.empty:
        res_df = pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "candidate_source", "country", "blocking_channel"])
    else:
        res_df = res_df.drop_duplicates(subset=["s1_entity_id", "candidate_entity_id"]).reset_index(drop=True)

    logger.info("Channel [Exact Name Block] generated %d candidate pairs for S1 -> %s", len(res_df), cand_source)
    return res_df
