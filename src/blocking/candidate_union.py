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


class CandidateUnionAccumulator:
    """Memory-efficient incremental accumulator for candidate pairs across channels."""

    def __init__(self) -> None:
        # key: (s1_entity_id, candidate_entity_id) -> list:
        # [0: source, 1: country, 2: channels_set, 3: tfidf_rank, 4: tfidf_sim, 5: emb_rank, 6: emb_sim]
        self.merged: dict[tuple[str, str], list] = {}

    def add_dataframe(self, df: pd.DataFrame | None) -> None:
        """Add candidate DataFrame to accumulator."""
        if df is None or df.empty:
            return

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

            if key not in self.merged:
                self.merged[key] = [
                    src_arr[i].strip() if i < len(src_arr) else "",
                    country_arr[i].strip() if i < len(country_arr) else "",
                    set(),
                    None,
                    None,
                    None,
                    None,
                ]

            rec = self.merged[key]
            ch = ch_arr[i].strip() if i < len(ch_arr) else ""
            if ch:
                rec[2].add(ch)

            if tfidf_rank_arr is not None:
                val = tfidf_rank_arr[i]
                if pd.notna(val) and val is not None:
                    rec[3] = int(val)
            if tfidf_sim_arr is not None:
                val = tfidf_sim_arr[i]
                if pd.notna(val) and val is not None:
                    rec[4] = float(val)
            if emb_rank_arr is not None:
                val = emb_rank_arr[i]
                if pd.notna(val) and val is not None:
                    rec[5] = int(val)
            if emb_sim_arr is not None:
                val = emb_sim_arr[i]
                if pd.notna(val) and val is not None:
                    rec[6] = float(val)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert accumulated pairs into final deduplicated DataFrame."""
        if not self.merged:
            return pd.DataFrame(columns=FINAL_COLUMNS)

        logger.info("Formatting %d unique candidate pairs into output schema...", len(self.merged))
        rows: list[dict] = []
        for (s1_id, cand_id), rec in self.merged.items():
            channels_list = sorted(list(rec[2]))
            rows.append({
                "s1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "candidate_source": rec[0],
                "country": rec[1],
                "blocking_channels": json.dumps(channels_list, ensure_ascii=False),
                "blocking_channel_count": len(channels_list),
                "tfidf_rank": rec[3],
                "tfidf_similarity": rec[4],
                "embedding_rank": rec[5],
                "embedding_similarity": rec[6],
            })

        union_df = pd.DataFrame(rows)
        if union_df.empty:
            return pd.DataFrame(columns=FINAL_COLUMNS)

        # Sort deterministically
        union_df = union_df.sort_values(by=["s1_entity_id", "candidate_source", "candidate_entity_id"]).reset_index(drop=True)
        logger.info("Stage 1 Union produced %d unique candidate pairs.", len(union_df))
        return union_df[FINAL_COLUMNS]


def union_candidates(channel_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Union multiple candidate DataFrames and deduplicate on (s1_entity_id, candidate_entity_id).

    Args:
        channel_dfs: List of candidate DataFrames from each blocking channel.

    Returns:
        Deduplicated DataFrame adhering to the Stage 1 output schema.
    """
    logger.info("Aggregating and unioning candidates across %d channel outputs...", len(channel_dfs))
    accumulator = CandidateUnionAccumulator()
    for df in channel_dfs:
        accumulator.add_dataframe(df)
    return accumulator.to_dataframe()
