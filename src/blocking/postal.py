"""
src/blocking/postal.py

Channel 4: Postal / Zip Code Blocking.
Generates candidate pairs when:
    - Same country (if country_aware)
    - Same non-empty postal / zip code (e.g. 5-digit, 6-digit, or alphanumeric postal format)
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

CHANNEL_NAME = "postal"

# Match standard postal code formats (5-digit ZIP, 6-digit PIN, UK alphanumeric, Canadian alphanumeric)
_POSTAL_RE = re.compile(
    r"\b("
    r"\d{5}(?:-\d{4})?|"       # US 5-digit ZIP or ZIP+4
    r"\d{6}|"                  # 6-digit postal code (India, China, etc.)
    r"[A-Z]\d[A-Z]\s?\d[A-Z]\d|" # Canadian postal
    r"[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}" # UK postal
    r")\b",
    re.IGNORECASE,
)


def extract_postal_code(address_normalized: str, row_numbers_json: str = "") -> str:
    """
    Extract a normalized postal code from the address or row numbers.

    Returns the postal code string (lowercased, spaces stripped) or "" if none.
    """
    if address_normalized:
        m = _POSTAL_RE.search(address_normalized)
        if m:
            return m.group(1).replace(" ", "").lower()

    if row_numbers_json:
        try:
            nums = json.loads(row_numbers_json)
            if isinstance(nums, list):
                # Search for a token of length 5 or 6 (common postal lengths)
                for n in nums:
                    sn = str(n).strip()
                    if len(sn) in (5, 6) and sn.isdigit():
                        return sn
        except Exception:
            pass

    return ""


def postal_block(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    cand_source: str,
    country_aware: bool = True,
) -> pd.DataFrame:
    """
    Generate candidate pairs where S1 and candidate entity share the same postal code.

    Args:
        s1_df: DataFrame of S1 query records.
        cand_df: DataFrame of candidate records (S2 or S3).
        cand_source: Label ('S2' or 'S3').
        country_aware: Whether to restrict matches to the same country.

    Returns:
        DataFrame of candidate pairs for the 'postal' channel.
    """
    logger.info("Running Channel [Postal Block] for S1 -> %s...", cand_source)

    if s1_df.empty or cand_df.empty:
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "candidate_source", "country", "blocking_channel"])

    country_col = "country_normalized"
    addr_col = "business_address_normalized"
    num_col = "row_numbers" if "row_numbers" in cand_df.columns else ""

    # Build candidate index: (country, postal_code) -> list of entity_id
    cand_index: dict[tuple[str, str], list[str]] = defaultdict(list)

    for _, row in cand_df.iterrows():
        addr = str(row.get(addr_col, "")).strip()
        row_nums = str(row.get(num_col, "")) if num_col else ""
        country = str(row.get(country_col, "")).strip()

        postal = extract_postal_code(addr, row_nums)
        if not postal:
            continue

        key = (country, postal) if country_aware else ("", postal)
        cand_index[key].append(row["entity_id"])

    # Query with S1
    results: list[dict[str, str]] = []
    for _, row in s1_df.iterrows():
        addr = str(row.get(addr_col, "")).strip()
        row_nums = str(row.get(num_col, "")) if num_col else ""
        country = str(row.get(country_col, "")).strip()

        postal = extract_postal_code(addr, row_nums)
        if not postal:
            continue

        key = (country, postal) if country_aware else ("", postal)
        matched_cand_ids = cand_index.get(key, [])

        for cand_id in matched_cand_ids:
            results.append({
                "s1_entity_id": row["entity_id"],
                "candidate_entity_id": cand_id,
                "candidate_source": cand_source,
                "country": country,
                "blocking_channel": CHANNEL_NAME,
            })

    res_df = pd.DataFrame(results)
    if res_df.empty:
        res_df = pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "candidate_source", "country", "blocking_channel"])
    else:
        res_df = res_df.drop_duplicates(subset=["s1_entity_id", "candidate_entity_id"]).reset_index(drop=True)

    logger.info("Channel [Postal Block] generated %d candidate pairs for S1 -> %s", len(res_df), cand_source)
    return res_df
