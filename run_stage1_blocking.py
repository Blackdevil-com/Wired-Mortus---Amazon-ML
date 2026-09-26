#!/usr/bin/env python
"""
run_stage1_blocking.py

CLI entry point for CASCADE-ER v2 -- Stage 1: Wide Blocking.
Optimized for multi-million row datasets in Google Colab and local environments.

Usage:
    # Run on default train datasets:
    python run_stage1_blocking.py --split train

    # Run fast rule-based channels only (exact, core, house_street, postal):
    python run_stage1_blocking.py --split train --skip-embeddings

    # Run all channels with GPU embeddings:
    python run_stage1_blocking.py \
        --split train \
        --s1 data/processed/train/train_source1_preprocessed.tsv \
        --s2 data/processed/train/train_source2_preprocessed.tsv \
        --s3 data/processed/train/train_source3_preprocessed.tsv \
        --ground-truth data/raw/train/train_ground_truth.tsv \
        --output data/candidates/stage1_candidate_union.tsv
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src.blocking.config import BlockingConfig
from src.blocking.pipeline import run_stage1_blocking
from src.evaluation.blocking_recall import evaluate_blocking_recall

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Essential columns required for blocking
_BLOCKING_USECOLS = [
    "entity_id",
    "business_name_normalized",
    "business_address_normalized",
    "country_normalized",
    "row_numbers",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_stage1_blocking.py",
        description="CASCADE-ER v2 -- Stage 1: Wide Blocking (High-Recall Candidate Generation)",
    )
    parser.add_argument("--s1", help="Path to preprocessed S1 TSV file.")
    parser.add_argument("--s2", help="Path to preprocessed S2 TSV file.")
    parser.add_argument("--s3", help="Path to preprocessed S3 TSV file.")
    parser.add_argument("--ground-truth", help="Path to train_ground_truth.tsv (for recall evaluation).")
    parser.add_argument("--split", choices=["train", "test"], default="train", help="Default dataset split ('train' or 'test').")
    parser.add_argument("--output", default="data/candidates/stage1_candidate_union.tsv", help="Output path for candidate union TSV.")
    parser.add_argument("--output-dir", default="data/candidates", help="Directory for output files.")

    # Retrieval tuning
    parser.add_argument("--tfidf-top-k", type=int, default=30, help="Top-K candidates per S1 entity for TF-IDF (default: 30).")
    parser.add_argument("--embedding-top-k", type=int, default=20, help="Top-K candidates per S1 entity for FAISS/embeddings (default: 20).")
    parser.add_argument("--embedding-model", default="paraphrase-multilingual-MiniLM-L12-v2", help="SentenceTransformer model name.")
    parser.add_argument("--embedding-batch-size", type=int, default=512, help="Batch size for embedding generation (default: 512).")

    # Channel toggles
    parser.add_argument("--skip-exact", action="store_true", help="Skip Exact Name blocking channel.")
    parser.add_argument("--skip-core", action="store_true", help="Skip Core Name blocking channel.")
    parser.add_argument("--skip-house-street", action="store_true", help="Skip House + Street blocking channel.")
    parser.add_argument("--skip-postal", action="store_true", help="Skip Postal blocking channel.")
    parser.add_argument("--skip-tfidf", action="store_true", help="Skip TF-IDF retrieval channel.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip Dense Embedding / FAISS retrieval channel.")

    parser.add_argument("--no-country-aware", action="store_true", help="Disable country partitioning during blocking.")
    parser.add_argument("--save-debug", action="store_true", help="Save individual channel TSV files in output directory.")
    parser.add_argument("--no-cache", action="store_true", help="Disable embedding disk caching.")
    parser.add_argument("--device", default=None, help="Device for embeddings ('cuda', 'cpu', or None for auto).")

    return parser.parse_args()


def _load_tsv_fast(path: Path) -> pd.DataFrame:
    """Load only the required columns from preprocessed TSV with fast C parser."""
    # First inspect header to see which required columns exist
    header = pd.read_csv(path, sep="\t", nrows=0)
    avail_cols = [c for c in _BLOCKING_USECOLS if c in header.columns]

    logger.info("Loading %s (columns: %s)...", path.name, avail_cols)
    t0 = time.perf_counter()
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=avail_cols,
        dtype=str,
        keep_default_na=False,
        engine="c",
    )
    elapsed = time.perf_counter() - t0
    logger.info("Loaded %d rows from %s in %.2f seconds.", len(df), path.name, elapsed)
    return df


def _resolve_input_files(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None, Path | None]:
    """Resolve paths to S1, S2, S3, and optional ground truth files."""
    split = args.split

    s1_cand = [
        args.s1,
        f"data/processed/{split}/train_source1_preprocessed.tsv" if split == "train" else f"data/processed/{split}/test_source1_preprocessed.tsv",
        f"data/processed/{split}_source1_preprocessed.tsv",
        f"data/processed/train_source1_preprocessed.tsv" if split == "train" else "data/processed/test_source1_preprocessed.tsv",
        "data/raw/train_source1_sample.tsv",
    ]

    s2_cand = [
        args.s2,
        f"data/processed/{split}/train_source2_preprocessed.tsv" if split == "train" else f"data/processed/{split}/test_source2_preprocessed.tsv",
        f"data/processed/{split}_source2_preprocessed.tsv",
        f"data/processed/train_source2_preprocessed.tsv" if split == "train" else "data/processed/test_source2_preprocessed.tsv",
    ]

    s3_cand = [
        args.s3,
        f"data/processed/{split}/train_source3_preprocessed.tsv" if split == "train" else f"data/processed/{split}/test_source3_preprocessed.tsv",
        f"data/processed/{split}_source3_preprocessed.tsv",
        f"data/processed/train_source3_preprocessed.tsv" if split == "train" else "data/processed/test_source3_preprocessed.tsv",
    ]

    s1_path = next((Path(p) for p in s1_cand if p and Path(p).exists()), None)
    s2_path = next((Path(p) for p in s2_cand if p and Path(p).exists()), None)
    s3_path = next((Path(p) for p in s3_cand if p and Path(p).exists()), None)

    if not s1_path:
        logger.error("Could not find S1 input file. Please provide with --s1 <path>.")
        sys.exit(1)

    gt_cand = [
        args.ground_truth,
        "data/raw/train_ground_truth.tsv",
        "data/raw/train/train_ground_truth.tsv",
    ]
    gt_path = next((Path(p) for p in gt_cand if p and Path(p).exists()), None)

    return s1_path, s2_path, s3_path, gt_path


def main() -> None:
    args = _parse_args()

    s1_path, s2_path, s3_path, gt_path = _resolve_input_files(args)

    logger.info("=" * 60)
    logger.info("CASCADE-ER v2 -- STAGE 1: WIDE BLOCKING")
    logger.info("=" * 60)

    s1_df = _load_tsv_fast(s1_path)

    if s2_path and s2_path.exists():
        s2_df = _load_tsv_fast(s2_path)
    else:
        logger.warning("S2 not found; candidate pool for S2 will be empty.")
        s2_df = pd.DataFrame(columns=s1_df.columns)

    if s3_path and s3_path.exists():
        s3_df = _load_tsv_fast(s3_path)
    else:
        logger.warning("S3 not found; candidate pool for S3 will be empty.")
        s3_df = pd.DataFrame(columns=s1_df.columns)

    # Initialize configuration
    config = BlockingConfig(
        use_exact_name_block=not args.skip_exact,
        use_core_name_block=not args.skip_core,
        use_house_street_block=not args.skip_house_street,
        use_postal_block=not args.skip_postal,
        use_tfidf_block=not args.skip_tfidf,
        use_embedding_block=not args.skip_embeddings,
        country_aware=not args.no_country_aware,
        tfidf_top_k=args.tfidf_top_k,
        embedding_top_k=args.embedding_top_k,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        save_debug_channels=args.save_debug,
        output_dir=args.output_dir,
        cache_embeddings=not args.no_cache,
        device=args.device,
    )

    # Run Stage 1 Wide Blocking pipeline
    union_df, channel_dict = run_stage1_blocking(s1_df, s2_df, s3_df, config=config)

    # If ground truth is available and we are on training data, evaluate recall
    if gt_path and args.split == "train":
        logger.info("Ground truth detected (%s). Evaluating Stage 1 Blocking Recall...", gt_path)
        evaluate_blocking_recall(union_df, gt_path, channel_dict=channel_dict)
    elif args.split == "test":
        logger.info("Test split execution: skipping ground truth evaluation (test ground truth does not exist).")

    logger.info("=" * 60)
    logger.info("STAGE 1: WIDE BLOCKING SUCCESSFUL.")
    logger.info("Output saved to: %s (%d candidate pairs)", config.get_output_path(), len(union_df))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
