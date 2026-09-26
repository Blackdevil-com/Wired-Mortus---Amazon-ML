"""
src/blocking/pipeline.py

Stage 1: Wide Blocking Pipeline.
Coordinates candidate generation across all blocking channels (exact name, core name,
house+street, postal, character TF-IDF, dense embeddings/FAISS), unions candidates,
saves the internal candidate union, and returns execution metrics.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from .candidate_union import FINAL_COLUMNS, union_candidates
from .config import BlockingConfig
from .core_name import core_name_block
from .embedding_blocking import embedding_block
from .exact_name import exact_name_block
from .house_street import house_street_block
from .postal import postal_block
from .tfidf_blocking import tfidf_block

logger = logging.getLogger(__name__)


def run_stage1_blocking(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    config: BlockingConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Execute all enabled Stage 1 blocking channels for S1 -> S2 and S1 -> S3,
    and generate the deduplicated candidate union.

    Args:
        s1_df: Normalized S1 query entities DataFrame.
        s2_df: Normalized S2 candidate entities DataFrame.
        s3_df: Normalized S3 candidate entities DataFrame.
        config: Optional BlockingConfig instance (uses defaults if None).

    Returns:
        (union_df, channel_dict)
            union_df: The final deduplicated candidate union DataFrame.
            channel_dict: Dictionary mapping channel names to their generated candidate pairs.
    """
    if config is None:
        config = BlockingConfig()

    logger.info("=" * 60)
    logger.info("CASCADE-ER v2 -- STAGE 1: WIDE BLOCKING")
    logger.info("=" * 60)
    logger.info("S1 Query records: %d", len(s1_df))
    logger.info("S2 Candidate records: %d", len(s2_df))
    logger.info("S3 Candidate records: %d", len(s3_df))
    logger.info("Country-aware blocking: %s", config.country_aware)

    channel_dfs: list[pd.DataFrame] = []
    channel_dict: dict[str, pd.DataFrame] = {}

    t0 = time.perf_counter()

    # ── 1. Exact Name Blocking ──────────────────────────────────────────────
    if config.use_exact_name_block:
        logger.info("[1/6] Running Exact Normalized Name Blocking...")
        exact_s2 = exact_name_block(s1_df, s2_df, "S2", country_aware=config.country_aware)
        exact_s3 = exact_name_block(s1_df, s3_df, "S3", country_aware=config.country_aware)
        exact_all = pd.concat([exact_s2, exact_s3], ignore_index=True)
        channel_dfs.append(exact_all)
        channel_dict["exact_name"] = exact_all
        logger.info("Exact Name candidates: %d", len(exact_all))

    # ── 2. Core Name Blocking ───────────────────────────────────────────────
    if config.use_core_name_block:
        logger.info("[2/6] Running Core Name Blocking...")
        core_s2 = core_name_block(s1_df, s2_df, "S2", country_aware=config.country_aware)
        core_s3 = core_name_block(s1_df, s3_df, "S3", country_aware=config.country_aware)
        core_all = pd.concat([core_s2, core_s3], ignore_index=True)
        channel_dfs.append(core_all)
        channel_dict["core_name"] = core_all
        logger.info("Core Name candidates: %d", len(core_all))

    # ── 3. House + Street Blocking ──────────────────────────────────────────
    if config.use_house_street_block:
        logger.info("[3/6] Running House Number + Street Token Blocking...")
        hs_s2 = house_street_block(s1_df, s2_df, "S2", country_aware=config.country_aware)
        hs_s3 = house_street_block(s1_df, s3_df, "S3", country_aware=config.country_aware)
        hs_all = pd.concat([hs_s2, hs_s3], ignore_index=True)
        channel_dfs.append(hs_all)
        channel_dict["house_street"] = hs_all
        logger.info("House + Street candidates: %d", len(hs_all))

    # ── 4. Postal Blocking ──────────────────────────────────────────────────
    if config.use_postal_block:
        logger.info("[4/6] Running Postal / Zip Code Blocking...")
        post_s2 = postal_block(s1_df, s2_df, "S2", country_aware=config.country_aware)
        post_s3 = postal_block(s1_df, s3_df, "S3", country_aware=config.country_aware)
        post_all = pd.concat([post_s2, post_s3], ignore_index=True)
        channel_dfs.append(post_all)
        channel_dict["postal"] = post_all
        logger.info("Postal candidates: %d", len(post_all))

    # ── 5. Character TF-IDF Retrieval ───────────────────────────────────────
    if config.use_tfidf_block:
        logger.info("[5/6] Running Character TF-IDF Retrieval (Top-K=%d)...", config.tfidf_top_k)
        tfidf_s2 = tfidf_block(
            s1_df, s2_df, "S2",
            top_k=config.tfidf_top_k,
            ngram_range=config.tfidf_ngram_range,
            min_sim=config.tfidf_min_sim,
            country_aware=config.country_aware,
        )
        tfidf_s3 = tfidf_block(
            s1_df, s3_df, "S3",
            top_k=config.tfidf_top_k,
            ngram_range=config.tfidf_ngram_range,
            min_sim=config.tfidf_min_sim,
            country_aware=config.country_aware,
        )
        tfidf_all = pd.concat([tfidf_s2, tfidf_s3], ignore_index=True)
        channel_dfs.append(tfidf_all)
        channel_dict["tfidf"] = tfidf_all
        logger.info("TF-IDF candidates: %d", len(tfidf_all))

    # ── 6. Multilingual Dense Embedding / FAISS ─────────────────────────────
    if config.use_embedding_block:
        logger.info("[6/6] Running Dense Embedding & FAISS Retrieval (Top-K=%d)...", config.embedding_top_k)
        emb_s2 = embedding_block(
            s1_df, s2_df, "S2",
            model_name=config.embedding_model,
            top_k=config.embedding_top_k,
            batch_size=config.embedding_batch_size,
            min_sim=config.embedding_min_sim,
            device=config.device,
            cache_dir=config.cache_dir,
            cache_embeddings=config.cache_embeddings,
            country_aware=config.country_aware,
        )
        emb_s3 = embedding_block(
            s1_df, s3_df, "S3",
            model_name=config.embedding_model,
            top_k=config.embedding_top_k,
            batch_size=config.embedding_batch_size,
            min_sim=config.embedding_min_sim,
            device=config.device,
            cache_dir=config.cache_dir,
            cache_embeddings=config.cache_embeddings,
            country_aware=config.country_aware,
        )
        emb_all = pd.concat([emb_s2, emb_s3], ignore_index=True)
        channel_dfs.append(emb_all)
        channel_dict["embedding"] = emb_all
        logger.info("Dense Embedding candidates: %d", len(emb_all))

    # ── Candidate Union & Deduplication ─────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Performing Candidate Union & Deduplication...")
    union_df = union_candidates(channel_dfs)

    elapsed = time.perf_counter() - t0
    logger.info("Stage 1 Wide Blocking completed in %.2f seconds.", elapsed)
    logger.info("Total Unique Candidate Union: %d pairs", len(union_df))
    logger.info("=" * 60)

    # ── Save Outputs ────────────────────────────────────────────────────────
    out_path = config.get_output_path()
    union_df.to_csv(out_path, sep="\t", index=False)
    logger.info("Saved Stage 1 candidate union to: %s", out_path)

    if config.save_debug_channels:
        debug_dir = Path(config.output_dir)
        for ch_name, ch_df in channel_dict.items():
            debug_file = debug_dir / f"stage1_{ch_name}.tsv"
            ch_df.to_csv(debug_file, sep="\t", index=False)
            logger.info("Saved debug channel file: %s (%d pairs)", debug_file, len(ch_df))

    return union_df, channel_dict
