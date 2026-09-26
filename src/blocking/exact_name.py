"""
src/blocking/exact_name.py

Channel 1: Exact Normalized Name Blocking.
Optimized with vector extraction and hash-based indexing for multi-million row datasets.
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
    """
    logger.info("Running Channel [Exact Name Block] for S1 -> %s...", cand_source)

    if s1_df.empty or cand_df.empty:
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "candidate_source", "country", "blocking_channel"])

    cand_name_col = "business_name_normalized"
    country_col = "country_normalized"

    # Fast vector extraction
    cand_names = cand_df[cand_name_col].fillna("").astype(str).to_numpy()
    cand_countries = cand_df[country_col].fillna("").astype(str).to_numpy() if country_col in cand_df.columns else [""] * len(cand_df)
    cand_ids = cand_df["entity_id"].astype(str).to_numpy()

    cand_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for i in range(len(cand_ids)):
        name = cand_names[i].strip()
        if not name:
            continue
        country = cand_countries[i].strip() if country_aware else ""
        cand_index[(country, name)].append(cand_ids[i])

    # Query with S1
    s1_names = s1_df[cand_name_col].fillna("").astype(str).to_numpy()
    s1_countries = s1_df[country_col].fillna("").astype(str).to_numpy() if country_col in s1_df.columns else [""] * len(s1_df)
    s1_ids = s1_df["entity_id"].astype(str).to_numpy()

    s1_res: list[str] = []
    cand_res: list[str] = []
    country_res: list[str] = []

    for i in range(len(s1_ids)):
        name = s1_names[i].strip()
        if not name:
            continue
        country = s1_countries[i].strip() if country_aware else ""
        matched = cand_index.get((country, name))
        if matched:
            for cid in matched:
                s1_res.append(s1_ids[i])
                cand_res.append(cid)
                country_res.append(country)

    if not s1_res:
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "candidate_source", "country", "blocking_channel"])

    res_df = pd.DataFrame({
        "s1_entity_id": s1_res,
        "candidate_entity_id": cand_res,
        "candidate_source": cand_source,
        "country": country_res,
        "blocking_channel": CHANNEL_NAME,
    }).drop_duplicates(subset=["s1_entity_id", "candidate_entity_id"]).reset_index(drop=True)

    logger.info("Channel [Exact Name Block] generated %d candidate pairs for S1 -> %s", len(res_df), cand_source)
    return res_df
