"""
src/blocking/candidate_union.py

Union and deduplication module for Stage 1 candidate generation.
Combines all candidate pairs discovered by any blocking channel, aggregates
channel metadata as a JSON array, counts discovering channels, and retains
retrieval ranks/similarities.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

FINAL_COLUMNS = [
    "s1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "country",
    "blocking_channels",
    "blocking_channel_count",
    "tfidf_rank",
    "tfidf_similarity",
    "embedding_rank",
    "embedding_similarity",
]


def union_candidates(channel_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Union multiple candidate DataFrames and deduplicate on (s1_entity_id, candidate_entity_id).

    Args:
        channel_dfs: List of candidate DataFrames from each blocking channel.

    Returns:
        Deduplicated DataFrame adhering to the Stage 1 output schema.
    """
    logger.info("Aggregating and unioning candidates across %d channel outputs...", len(channel_dfs))

    if not channel_dfs:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    # Filter non-empty DataFrames
    valid_dfs = [df for df in channel_dfs if df is not None and not df.empty]
    if not valid_dfs:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    # Dictionary to aggregate attributes by pair key
    # key: (s1_entity_id, candidate_entity_id) -> data dict
    merged: dict[tuple[str, str], dict] = {}

    for df in valid_dfs:
        s1_arr = df["s1_entity_id"].astype(str).to_numpy()
        cand_arr = df["candidate_entity_id"].astype(str).to_numpy()
        src_arr = df["candidate_source"].astype(str).to_numpy() if "candidate_source" in df.columns else [""] * len(df)
        country_arr = df["country"].astype(str).to_numpy() if "country" in df.columns else [""] * len(df)
        ch_arr = df["blocking_channel"].astype(str).to_numpy() if "blocking_channel" in df.columns else [""] * len(df)

        has_tfidf_rank = "tfidf_rank" in df.columns
        has_tfidf_sim = "tfidf_similarity" in df.columns
        has_emb_rank = "embedding_rank" in df.columns
        has_emb_sim = "embedding_similarity" in df.columns

        tfidf_rank_arr = df["tfidf_rank"].to_numpy() if has_tfidf_rank else None
        tfidf_sim_arr = df["tfidf_similarity"].to_numpy() if has_tfidf_sim else None
        emb_rank_arr = df["embedding_rank"].to_numpy() if has_emb_rank else None
        emb_sim_arr = df["embedding_similarity"].to_numpy() if has_emb_sim else None

        n = len(s1_arr)
        for i in range(n):
            s1_id = s1_arr[i].strip()
            cand_id = cand_arr[i].strip()
            key = (s1_id, cand_id)

            if key not in merged:
                merged[key] = {
                    "s1_entity_id": s1_id,
                    "candidate_entity_id": cand_id,
                    "candidate_source": src_arr[i].strip() if i < len(src_arr) else "",
                    "country": country_arr[i].strip() if i < len(country_arr) else "",
                    "channels_set": set(),
                    "tfidf_rank": None,
                    "tfidf_similarity": None,
                    "embedding_rank": None,
                    "embedding_similarity": None,
                }

            record = merged[key]
            ch = ch_arr[i].strip() if i < len(ch_arr) else ""
            if ch:
                record["channels_set"].add(ch)

            if tfidf_rank_arr is not None:
                val = tfidf_rank_arr[i]
                if pd.notna(val) and val is not None:
                    record["tfidf_rank"] = int(val)
            if tfidf_sim_arr is not None:
                val = tfidf_sim_arr[i]
                if pd.notna(val) and val is not None:
                    record["tfidf_similarity"] = float(val)
            if emb_rank_arr is not None:
                val = emb_rank_arr[i]
                if pd.notna(val) and val is not None:
                    record["embedding_rank"] = int(val)
            if emb_sim_arr is not None:
                val = emb_sim_arr[i]
                if pd.notna(val) and val is not None:
                    record["embedding_similarity"] = float(val)

    # Format into rows
    logger.info("Formatting %d unique candidate pairs into output schema...", len(merged))
    rows: list[dict] = []
    for (s1_id, cand_id), record in merged.items():
        channels_list = sorted(list(record["channels_set"]))
        rows.append({
            "s1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "candidate_source": record["candidate_source"],
            "country": record["country"],
            "blocking_channels": json.dumps(channels_list, ensure_ascii=False),
            "blocking_channel_count": len(channels_list),
            "tfidf_rank": record["tfidf_rank"],
            "tfidf_similarity": record["tfidf_similarity"],
            "embedding_rank": record["embedding_rank"],
            "embedding_similarity": record["embedding_similarity"],
        })

    union_df = pd.DataFrame(rows)
    if union_df.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    # Sort deterministically
    union_df = union_df.sort_values(by=["s1_entity_id", "candidate_source", "candidate_entity_id"]).reset_index(drop=True)
    logger.info("Stage 1 Union produced %d unique candidate pairs.", len(union_df))
    return union_df[FINAL_COLUMNS]
