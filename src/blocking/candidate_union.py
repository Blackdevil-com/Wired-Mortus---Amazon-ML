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
        for _, row in df.iterrows():
            s1_id = str(row["s1_entity_id"]).strip()
            cand_id = str(row["candidate_entity_id"]).strip()
            key = (s1_id, cand_id)

            channel = str(row.get("blocking_channel", "")).strip()
            source = str(row.get("candidate_source", "")).strip()
            country = str(row.get("country", "")).strip()

            tfidf_rank = row.get("tfidf_rank", None)
            tfidf_sim = row.get("tfidf_similarity", None)
            emb_rank = row.get("embedding_rank", None)
            emb_sim = row.get("embedding_similarity", None)

            if key not in merged:
                merged[key] = {
                    "s1_entity_id": s1_id,
                    "candidate_entity_id": cand_id,
                    "candidate_source": source,
                    "country": country,
                    "channels_set": set(),
                    "tfidf_rank": None,
                    "tfidf_similarity": None,
                    "embedding_rank": None,
                    "embedding_similarity": None,
                }

            record = merged[key]
            if channel:
                record["channels_set"].add(channel)
            if source and not record["candidate_source"]:
                record["candidate_source"] = source
            if country and not record["country"]:
                record["country"] = country

            # Update retrieval metadata if present
            if pd.notna(tfidf_rank) and tfidf_rank is not None:
                record["tfidf_rank"] = int(tfidf_rank)
            if pd.notna(tfidf_sim) and tfidf_sim is not None:
                record["tfidf_similarity"] = float(tfidf_sim)
            if pd.notna(emb_rank) and emb_rank is not None:
                record["embedding_rank"] = int(emb_rank)
            if pd.notna(emb_sim) and emb_sim is not None:
                record["embedding_similarity"] = float(emb_sim)

    # Format into rows
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
