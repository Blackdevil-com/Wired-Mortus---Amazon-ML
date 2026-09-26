#!/usr/bin/env python
"""
run_preprocessing.py

Entry point for Stage 1: Entity Resolution Preprocessing.

Supports separate processing for:
  - Training datasets: data/raw/train -> data/processed/train
  - Testing datasets:  data/raw/test  -> data/processed/test
  - All datasets:      data/raw       -> data/processed

Usage
-----
Process all datasets:
    python run_preprocessing.py --split all

Process only training datasets:
    python run_preprocessing.py --split train

Process only test datasets:
    python run_preprocessing.py --split test

With explicit folder paths:
    python run_preprocessing.py --raw-dir data/raw/train --processed-dir data/processed/train
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
# Default file mappings
# ---------------------------------------------------------------------------

_TRAIN_NAMES = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]
_TEST_NAMES = ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]


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
        "--split",
        choices=["all", "train", "test"],
        default="all",
        help="Which split to process: 'train', 'test', or 'all' (default: 'all').",
    )
    parser.add_argument(
        "--raw-dir",
        default="",
        metavar="DIR",
        help="Directory containing raw TSV files (default: auto-detected from split).",
    )
    parser.add_argument(
        "--processed-dir",
        default="",
        metavar="DIR",
        help="Directory where preprocessed TSV files will be written (default: auto-detected from split).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        metavar="ENC",
        help="File encoding (default: utf-8).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_paths(split: str, raw_dir_arg: str, proc_dir_arg: str) -> tuple[Path, Path]:
    """Resolve raw and processed directories based on split and user arguments."""
    if raw_dir_arg:
        raw_dir = Path(raw_dir_arg)
    else:
        if split == "train" and Path("data/raw/train").exists():
            raw_dir = Path("data/raw/train")
        elif split == "test" and Path("data/raw/test").exists():
            raw_dir = Path("data/raw/test")
        else:
            raw_dir = Path("data/raw")

    if proc_dir_arg:
        proc_dir = Path(proc_dir_arg)
    else:
        if split == "train" and (raw_dir == Path("data/raw/train") or Path("data/raw/train").exists()):
            proc_dir = Path("data/processed/train")
        elif split == "test" and (raw_dir == Path("data/raw/test") or Path("data/raw/test").exists()):
            proc_dir = Path("data/processed/test")
        else:
            proc_dir = Path("data/processed")

    return raw_dir, proc_dir


def _collect_files(raw_dir: Path, split: str) -> list[tuple[Path, str]]:
    """
    Collect input files to process.
    Matches expected file names or any TSV file found in the raw directory.
    """
    pairs: list[tuple[Path, str]] = []

    if split == "train":
        expected = _TRAIN_NAMES
    elif split == "test":
        expected = _TEST_NAMES
    else:
        expected = _TRAIN_NAMES + _TEST_NAMES

    # Check for expected files
    found_expected = set()
    for name in expected:
        p = raw_dir / name
        if p.exists():
            out_name = f"{p.stem}_preprocessed.tsv"
            pairs.append((p, out_name))
            found_expected.add(name)

    # Also discover any other .tsv files in the raw_dir not in expected list
    for p in sorted(raw_dir.glob("*.tsv")):
        if p.name not in found_expected and not p.name.endswith("_preprocessed.tsv"):
            out_name = f"{p.stem}_preprocessed.tsv"
            pairs.append((p, out_name))

    return pairs


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

    # ── Batch mode ──────────────────────────────────────────────────────────
    raw_dir, processed_dir = _resolve_paths(args.split, args.raw_dir, args.processed_dir)
    file_pairs = _collect_files(raw_dir, args.split)

    logger.info("=" * 60)
    logger.info("ENTITY RESOLUTION -- STAGE 1 (BATCH MODE: %s)", args.split.upper())
    logger.info("Raw Directory:       %s", raw_dir)
    logger.info("Processed Directory: %s", processed_dir)
    logger.info("Files detected:      %d", len(file_pairs))
    logger.info("=" * 60)

    if not file_pairs:
        logger.warning("No TSV files found to process in %s", raw_dir)
        return

    success = failed = 0

    for in_path, out_name in file_pairs:
        out_path = processed_dir / out_name
        try:
            process_file(in_path, out_path, encoding=args.encoding)
            success += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("[FAIL] %s  -->  %s", in_path, exc)
            failed += 1

    logger.info("=" * 60)
    logger.info(
        "BATCH COMPLETE (%s)  success=%d  failed=%d",
        args.split.upper(), success, failed,
    )
    logger.info("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
