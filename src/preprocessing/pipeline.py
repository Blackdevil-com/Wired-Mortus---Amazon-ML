"""
src/preprocessing/pipeline.py

Stage 1 Preprocessing Pipeline.

Reads a raw TSV file in streaming chunks, applies normalization and
tokenization to relevant fields, and writes a preprocessed TSV.

Key guarantees:
    - Original columns are NEVER modified.
    - Input/output row counts are validated.
    - Missing values produce "" (not "nan", "none", "null").
    - Numbers in names and addresses are preserved.
    - Non-Latin scripts are not destroyed.
    - Output columns are ordered: original cols first, then preprocessing cols.

No matching, blocking, or candidate-generation logic is present here.
This is Stage 1 only.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from .text_normalizer import nfkd_lowercase, normalize
from .tokenizer import extract_numbers_to_json, tokenize_to_json
from .unicode_normalizer import safe_str

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 50_000  # rows read per iteration

# Source columns we produce preprocessing representations for
_NAME_COL = "business_name"
_ADDR_COL = "business_address"
_CTRY_COL = "country"

# Preferred order for the leading original columns
_CORE_COLS = ["entity_id", _NAME_COL, _ADDR_COL, _CTRY_COL]

# New preprocessing columns appended after all original columns
PREPROCESSING_COLS: list[str] = [
    "business_name_nfkd",
    "business_name_normalized",
    "business_name_tokens",
    "business_address_nfkd",
    "business_address_normalized",
    "business_address_tokens",
    "country_normalized",
    "row_numbers",
]



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_lines(path: Path, encoding: str) -> int:
    """Count data rows (excluding the header line) in *path*."""
    with open(path, encoding=encoding, errors="replace") as fh:
        total = sum(1 for _ in fh)
    return max(0, total - 1)


def _build_output_order(original_cols: list[str]) -> list[str]:
    """
    Determine the final column order for the output TSV.

    1. Core columns (entity_id, business_name, business_address, country),
       in the defined order, if present in the input.
    2. Any remaining original columns in their original order.
    3. All preprocessing columns.
    """
    core = [c for c in _CORE_COLS if c in original_cols]
    extras = [c for c in original_cols if c not in _CORE_COLS]
    return core + extras + PREPROCESSING_COLS


def _process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing operations to a single DataFrame chunk.

    The original columns in *chunk* are not mutated; new columns are added.
    """
    df = chunk.copy()

    # ── business_name ───────────────────────────────────────────────────────
    if _NAME_COL in df.columns:
        raw_name = df[_NAME_COL]
        df["business_name_nfkd"] = raw_name.apply(nfkd_lowercase)
        df["business_name_normalized"] = raw_name.apply(normalize)
    else:
        df["business_name_nfkd"] = ""
        df["business_name_normalized"] = ""

    df["business_name_tokens"] = df["business_name_normalized"].apply(tokenize_to_json)

    # ── business_address ────────────────────────────────────────────────────
    if _ADDR_COL in df.columns:
        raw_addr = df[_ADDR_COL]
        df["business_address_nfkd"] = raw_addr.apply(nfkd_lowercase)
        df["business_address_normalized"] = raw_addr.apply(normalize)
    else:
        df["business_address_nfkd"] = ""
        df["business_address_normalized"] = ""

    df["business_address_tokens"] = df["business_address_normalized"].apply(tokenize_to_json)

    # ── country ─────────────────────────────────────────────────────────────
    if _CTRY_COL in df.columns:
        df["country_normalized"] = df[_CTRY_COL].apply(normalize)
    else:
        df["country_normalized"] = ""

    # ── row_numbers ─────────────────────────────────────────────────────────
    # Preserve every number in the row in a separate column (JSON array of strings)
    # Extracts all numeric digit sequences across all raw data attributes in the row
    data_cols = [c for c in chunk.columns if c != "entity_id"]
    if data_cols:
        row_text = chunk[data_cols].fillna("").astype(str).agg(" ".join, axis=1)
        df["row_numbers"] = row_text.apply(extract_numbers_to_json)
    else:
        df["row_numbers"] = "[]"

    return df


def _validate(
    input_path: Path,
    output_path: Path,
    input_rows: int,
    output_rows: int,
    encoding: str,
) -> None:
    """
    Post-processing sanity checks.

    Validates:
        1. Input and output row counts match.
        2. Output file exists and is readable.
        3. All expected preprocessing columns are present in the output.
    """
    errors: list[str] = []

    if input_rows != output_rows:
        errors.append(
            f"Row count mismatch: input={input_rows:,}, output={output_rows:,}"
        )

    if not output_path.exists():
        errors.append(f"Output file not found: {output_path}")
    else:
        try:
            sample = pd.read_csv(
                output_path,
                sep="\t",
                encoding=encoding,
                nrows=2,
                dtype=str,
                keep_default_na=False,
            )
            for col in PREPROCESSING_COLS:
                if col not in sample.columns:
                    errors.append(f"Missing preprocessing column: '{col}'")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Could not read output for validation: {exc}")

    if errors:
        formatted = "\n".join(f"  * {e}" for e in errors)
        raise RuntimeError(f"Validation failed:\n{formatted}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_file(
    input_path: str | Path,
    output_path: str | Path,
    encoding: str = "utf-8",
) -> int:
    """
    Preprocess a single source TSV file (Stage 1).

    Reads the file in streaming chunks, applies normalization and
    tokenization, writes the preprocessed TSV, and validates the result.

    Args:
        input_path:  Path to the raw source TSV file.
        output_path: Destination path for the preprocessed TSV.
        encoding:    Character encoding (default "utf-8").

    Returns:
        Number of records (rows) written to *output_path*.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        RuntimeError:      If post-processing validation fails.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ENTITY RESOLUTION -- STAGE 1 PREPROCESSING")
    logger.info("=" * 60)
    logger.info("Input:  %s", input_path)

    # ── Pre-count rows ──────────────────────────────────────────────────────
    logger.info("Counting input rows...")
    input_rows = _count_lines(input_path, encoding)
    logger.info("Records loaded: %d", input_rows)

    logger.info("Processing business names, addresses, countries...")
    logger.info("Tokenizing...")
    logger.info("Writing output: %s", output_path)

    # ── Stream-process ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    output_rows = 0
    first_chunk = True
    col_order: list[str] | None = None
    last_logged = 0

    reader = pd.read_csv(
        input_path,
        sep="\t",
        encoding=encoding,
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:
        processed = _process_chunk(chunk)

        if col_order is None:
            col_order = _build_output_order(list(chunk.columns))

        write_df = processed[col_order]
        write_df.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            encoding=encoding,
        )

        output_rows += len(write_df)
        first_chunk = False

        # Log progress at every 100k-row boundary
        bucket = output_rows // 100_000
        if bucket > last_logged // 100_000 or output_rows == input_rows:
            elapsed = time.perf_counter() - t0
            rate = output_rows / elapsed if elapsed > 0 else 0
            logger.info(
                "  Processed %d / %d rows  (%.0f rows/s)",
                output_rows,
                input_rows,
                rate,
            )
            last_logged = output_rows

    # ── Validate ────────────────────────────────────────────────────────────
    _validate(input_path, output_path, input_rows, output_rows, encoding)

    elapsed = time.perf_counter() - t0
    rate = output_rows / elapsed if elapsed > 0 else 0

    logger.info("-" * 60)
    logger.info("Records written: %d  (%.0f rows/s,  %.1f s total)", output_rows, rate, elapsed)
    logger.info("STATUS: SUCCESS")
    logger.info("=" * 60)

    return output_rows
