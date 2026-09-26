#!/usr/bin/env python
"""
run_preprocessing.py

Entry point for Stage 1: Entity Resolution Preprocessing.

Usage
-----
Process all available files automatically:
    python run_preprocessing.py

Process a single file:
    python run_preprocessing.py \\
        --input  data/raw/train_source1.tsv \\
        --output data/processed/train_source1_preprocessed.tsv

With custom encoding:
    python run_preprocessing.py --input ... --output ... --encoding utf-8-sig
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.preprocessing.pipeline import process_file

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default file mapping (raw -> processed)
# ---------------------------------------------------------------------------

_RAW = Path("data/raw")
_PROC = Path("data/processed")

_DEFAULT_FILES: list[tuple[str, str]] = [
    ("train_source1.tsv", "train_source1_preprocessed.tsv"),
    ("train_source2.tsv", "train_source2_preprocessed.tsv"),
    ("train_source3.tsv", "train_source3_preprocessed.tsv"),
    ("test_source1.tsv",  "test_source1_preprocessed.tsv"),
    ("test_source2.tsv",  "test_source2_preprocessed.tsv"),
    ("test_source3.tsv",  "test_source3_preprocessed.tsv"),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_preprocessing.py",
        description="Entity Resolution — Stage 1: Preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        metavar="PATH",
        help="Path to a single raw TSV input file. Requires --output.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Path for the preprocessed TSV output file. Requires --input.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        metavar="ENC",
        help="File encoding (default: utf-8).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    # ── Single-file mode ────────────────────────────────────────────────────
    if args.input or args.output:
        if not (args.input and args.output):
            logger.error("Both --input and --output must be provided together.")
            sys.exit(1)
        try:
            n = process_file(args.input, args.output, encoding=args.encoding)
            logger.info("Done. %d records written to %s.", n, args.output)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Preprocessing failed: %s", exc)
            sys.exit(1)
        return

    # ── Batch mode: process all default files ───────────────────────────────
    logger.info("=" * 60)
    logger.info("ENTITY RESOLUTION -- STAGE 1  (BATCH MODE)")
    logger.info("=" * 60)

    success = skipped = failed = 0

    for raw_name, proc_name in _DEFAULT_FILES:
        in_path = _RAW / raw_name
        out_path = _PROC / proc_name

        if not in_path.exists():
            logger.warning("[SKIP] Input not found: %s", in_path)
            skipped += 1
            continue

        try:
            process_file(in_path, out_path, encoding=args.encoding)
            success += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("[FAIL] %s  -->  %s", in_path, exc)
            failed += 1

    logger.info("=" * 60)
    logger.info(
        "BATCH COMPLETE  success=%d  skipped=%d  failed=%d",
        success, skipped, failed,
    )
    logger.info("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
