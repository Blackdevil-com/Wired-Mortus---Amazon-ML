"""
tests/test_pipeline.py

Integration tests for src.preprocessing.pipeline.process_file.

Uses temporary files so no real data is required.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.pipeline import PREPROCESSING_COLS, process_file

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_TSV = (
    "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    "S1-925783039\tOrelee's Barbershop\t1795 Westchester Drive, High Point, NC\tUS\n"
    "S1-773889195\tPrime Money\t17560 Ellis Road, Tahlequah, OK\tUS\n"
    "S1-377745466\tB+ Retail Inc\t1712 Montebello Avenue, Phoenix, AZ\tUS\n"
    "S1-133037285\tChrist Chapel\t2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD\tUS\n"
    "S1-111111111\tCaf\u00e9 \u00c9lite\t10 Rue de la Paix, Paris\tFR\n"
    "S1-222222222\t\t\t\n"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_tsv(tmp_path: Path) -> Path:
    p = tmp_path / "input.tsv"
    p.write_text(SAMPLE_TSV, encoding="utf-8")
    return p


@pytest.fixture()
def out_tsv(tmp_path: Path) -> Path:
    return tmp_path / "output.tsv"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProcessFile:

    # ── Basic ────────────────────────────────────────────────────────────────

    def test_returns_row_count(self, sample_tsv, out_tsv):
        n = process_file(sample_tsv, out_tsv)
        assert n == 6

    def test_output_file_created(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        assert out_tsv.exists()

    # ── Column presence ──────────────────────────────────────────────────────

    def test_all_preprocessing_columns_present(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        for col in PREPROCESSING_COLS:
            assert col in df.columns, f"Missing: {col}"

    def test_original_columns_present(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        for col in ["entity_id", "business_name", "business_address", "country"]:
            assert col in df.columns

    # ── Data preservation ────────────────────────────────────────────────────

    def test_original_columns_not_modified(self, sample_tsv, out_tsv):
        in_df = pd.read_csv(sample_tsv, sep="\t", dtype=str, keep_default_na=False)
        process_file(sample_tsv, out_tsv)
        out_df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        for col in ["entity_id", "business_name", "business_address", "country"]:
            assert list(in_df[col]) == list(out_df[col]), (
                f"Column '{col}' was modified by preprocessing"
            )

    def test_row_count_preserved(self, sample_tsv, out_tsv):
        in_df = pd.read_csv(sample_tsv, sep="\t", dtype=str, keep_default_na=False)
        n = process_file(sample_tsv, out_tsv)
        assert n == len(in_df)

    # ── Token JSON validity ──────────────────────────────────────────────────

    def test_name_tokens_valid_json(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        for val in df["business_name_tokens"]:
            parsed = json.loads(val)
            assert isinstance(parsed, list)

    def test_address_tokens_valid_json(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        for val in df["business_address_tokens"]:
            parsed = json.loads(val)
            assert isinstance(parsed, list)

    # ── Normalization correctness ────────────────────────────────────────────

    def test_orelees_normalized(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        row = df[df["entity_id"] == "S1-925783039"].iloc[0]
        assert row["business_name_normalized"] == "orelee s barbershop"
        tokens = json.loads(row["business_name_tokens"])
        assert tokens == ["orelee", "s", "barbershop"]


    def test_b_plus_normalized(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        row = df[df["entity_id"] == "S1-377745466"].iloc[0]
        assert row["business_name_normalized"] == "b retail inc"

    def test_country_normalized(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        row = df[df["entity_id"] == "S1-925783039"].iloc[0]
        assert row["country_normalized"] == "us"

    # ── Number preservation ──────────────────────────────────────────────────

    def test_address_number_preserved_in_normalized(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        row = df[df["entity_id"] == "S1-133037285"].iloc[0]
        assert "2100" in row["business_address_normalized"]

    def test_address_number_preserved_in_tokens(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        row = df[df["entity_id"] == "S1-133037285"].iloc[0]
        tokens = json.loads(row["business_address_tokens"])
        assert "2100" in tokens

    def test_large_address_number_preserved(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        row = df[df["entity_id"] == "S1-773889195"].iloc[0]
        assert "17560" in row["business_address_normalized"]
        tokens = json.loads(row["business_address_tokens"])
        assert "17560" in tokens

    def test_row_numbers_valid_json(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        for val in df["row_numbers"]:
            parsed = json.loads(val)
            assert isinstance(parsed, list)

    def test_row_numbers_extracted_correctly(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        # S1-925783039: 1795 Westchester Drive
        row = df[df["entity_id"] == "S1-925783039"].iloc[0]
        assert json.loads(row["row_numbers"]) == ["1795"]

        # S1-133037285: 2100 Cameron Drive
        row2 = df[df["entity_id"] == "S1-133037285"].iloc[0]
        assert json.loads(row2["row_numbers"]) == ["2100"]

        # S1-111111111: 10 Rue de la Paix
        row3 = df[df["entity_id"] == "S1-111111111"].iloc[0]
        assert json.loads(row3["row_numbers"]) == ["10"]

    def test_row_numbers_empty_for_no_digits(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        row = df[df["entity_id"] == "S1-222222222"].iloc[0]
        assert json.loads(row["row_numbers"]) == []


    # ── Missing value handling ───────────────────────────────────────────────

    def test_missing_name_gives_empty_normalized(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        row = df[df["entity_id"] == "S1-222222222"].iloc[0]
        assert row["business_name_normalized"] == ""

    def test_missing_name_gives_empty_token_array(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        row = df[df["entity_id"] == "S1-222222222"].iloc[0]
        assert json.loads(row["business_name_tokens"]) == []

    def test_missing_values_not_nan_string(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        row = df[df["entity_id"] == "S1-222222222"].iloc[0]
        assert row["business_name_normalized"] != "nan"
        assert row["business_address_normalized"] != "nan"

    # ── Accent handling ──────────────────────────────────────────────────────

    def test_accent_stripped_in_normalized(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        row = df[df["entity_id"] == "S1-111111111"].iloc[0]
        assert row["business_name_normalized"] == "cafe elite"

    def test_accent_preserved_in_nfkd(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        row = df[df["entity_id"] == "S1-111111111"].iloc[0]
        # nfkd column keeps combining marks
        nfkd_val = row["business_name_nfkd"]
        assert "cafe" in nfkd_val  # still has base letters

    # ── Error handling ───────────────────────────────────────────────────────

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            process_file(tmp_path / "nonexistent.tsv", tmp_path / "out.tsv")

    # ── Column ordering ──────────────────────────────────────────────────────

    def test_original_columns_come_first(self, sample_tsv, out_tsv):
        process_file(sample_tsv, out_tsv)
        df = pd.read_csv(out_tsv, sep="\t", dtype=str, keep_default_na=False)
        cols = list(df.columns)
        # entity_id must appear before preprocessing columns
        assert cols.index("entity_id") < cols.index("business_name_nfkd")
        assert cols.index("business_name") < cols.index("business_name_nfkd")
