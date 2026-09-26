"""
src/evaluation/blocking_recall.py

Evaluation module for Stage 1: Wide Blocking Recall.
Measures what proportion of true matches (from train_ground_truth.tsv)
were captured by each blocking channel and by the final candidate union.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def parse_ground_truth_pairs(gt_df: pd.DataFrame) -> set[tuple[str, str]]:
    """
    Parse ground truth pairs into a set of (s1_id, matched_id) tuples.
    Supports either:
      - (source1_entity_id, matched_entity_ids) where matched_entity_ids can be comma-separated or JSON list
      - (s1_entity_id, candidate_entity_id) tabular format
    """
    true_pairs: set[tuple[str, str]] = set()

    # Case 1: source1_entity_id, matched_entity_ids
    if "source1_entity_id" in gt_df.columns and "matched_entity_ids" in gt_df.columns:
        for _, row in gt_df.iterrows():
            s1_id = str(row["source1_entity_id"]).strip()
            raw_matched = str(row["matched_entity_ids"]).strip()
            if not s1_id or not raw_matched or raw_matched == "nan":
                continue

            # Try parsing as JSON list
            if raw_matched.startswith("[") and raw_matched.endswith("]"):
                try:
                    parsed = json.loads(raw_matched)
                    if isinstance(parsed, list):
                        for m_id in parsed:
                            true_pairs.add((s1_id, str(m_id).strip()))
                        continue
                except Exception:
                    pass

            # Split by comma or semicolon
            for m_id in raw_matched.replace(";", ",").split(","):
                m_clean = m_id.strip()
                if m_clean:
                    true_pairs.add((s1_id, m_clean))

    # Case 2: tabular s1_entity_id, candidate_entity_id
    elif "s1_entity_id" in gt_df.columns and "candidate_entity_id" in gt_df.columns:
        for _, row in gt_df.iterrows():
            s1_id = str(row["s1_entity_id"]).strip()
            cand_id = str(row["candidate_entity_id"]).strip()
            if s1_id and cand_id:
                true_pairs.add((s1_id, cand_id))

    return true_pairs


def evaluate_blocking_recall(
    candidate_union_df: pd.DataFrame,
    ground_truth: pd.DataFrame | str | Path,
    channel_dict: dict[str, pd.DataFrame] | None = None,
) -> dict[str, float]:
    """
    Compute candidate recall for each channel and the overall union against ground truth.

    Args:
        candidate_union_df: DataFrame of Stage 1 candidate union.
        ground_truth: DataFrame or file path to train_ground_truth.tsv.
        channel_dict: Optional dictionary of individual channel DataFrames.

    Returns:
        Dictionary mapping channel names & 'union' to recall percentages.
    """
    if isinstance(ground_truth, (str, Path)):
        gt_path = Path(ground_truth)
        if not gt_path.exists():
            logger.warning("Ground truth file not found: %s", gt_path)
            return {}
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    else:
        gt_df = ground_truth

    true_pairs = parse_ground_truth_pairs(gt_df)
    total_true = len(true_pairs)
    if total_true == 0:
        logger.warning("No valid ground truth pairs found for evaluation.")
        return {}

    # Check Union recall
    union_pairs = {
        (str(row["s1_entity_id"]).strip(), str(row["candidate_entity_id"]).strip())
        for _, row in candidate_union_df.iterrows()
    }
    union_recovered = len(true_pairs & union_pairs)
    union_recall = (union_recovered / total_true) * 100.0

    report: dict[str, float] = {
        "total_true_matches": total_true,
        "union_recovered": union_recovered,
        "union_recall": round(union_recall, 2),
    }

    logger.info("=" * 60)
    logger.info("STAGE 1 BLOCKING RECALL EVALUATION")
    logger.info("=" * 60)
    logger.info("Total True Matches in Ground Truth: %d", total_true)
    logger.info("True Matches Recovered by Union:    %d", union_recovered)
    logger.info("OVERALL UNION BLOCKING RECALL:      %.2f%%", union_recall)
    logger.info("-" * 60)
    logger.info("Per-Channel Recall Breakdown:")

    # Per-channel evaluation
    if channel_dict:
        for ch_name, ch_df in channel_dict.items():
            ch_pairs = {
                (str(row["s1_entity_id"]).strip(), str(row["candidate_entity_id"]).strip())
                for _, row in ch_df.iterrows()
            }
            ch_recovered = len(true_pairs & ch_pairs)
            ch_recall = (ch_recovered / total_true) * 100.0
            report[f"{ch_name}_recovered"] = ch_recovered
            report[f"{ch_name}_recall"] = round(ch_recall, 2)
            logger.info("  * %-15s: %6d / %d  (%6.2f%%)", ch_name, ch_recovered, total_true, ch_recall)

    logger.info("=" * 60)
    return report
