"""
src/blocking/house_street.py

Channel 3: House Number + Street Token Blocking.
Generates candidate pairs when:
    - Same country (if country_aware)
    - Same house number
    - At least one meaningful street token overlaps
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

CHANNEL_NAME = "house_street"

# Generic street type designators and unit keywords to ignore during token overlap
_STREET_STOPWORDS = {
    "st", "street", "rd", "road", "ave", "avenue", "dr", "drive", "ln", "lane",
    "blvd", "boulevard", "ct", "court", "pl", "place", "pkwy", "parkway", "way",
    "hwy", "highway", "cir", "circle", "ste", "suite", "apt", "apartment", "unit",
    "fl", "floor", "rm", "room", "bldg", "building", "box", "po", "pob", "dept",
    "no", "num", "north", "south", "east", "west", "n", "s", "e", "w",
}

_HOUSE_NUM_RE = re.compile(r"\b(\d+[a-zA-Z]?)\b")


def extract_house_and_street(address_normalized: str, row_numbers_json: str = "") -> tuple[str, set[str]]:
    """
    Extract the primary house/building number and meaningful street tokens.

    Examples:
        "1795 westchester drive high point nc" -> ("1795", {"westchester", "high", "point", "nc"})
        "2100 cameron drive unit apartment g"  -> ("2100", {"cameron"})
    """
    if not address_normalized:
        return "", set()

    tokens = address_normalized.split()
    if not tokens:
        return "", set()

    house_number = ""

    # Check first token for number
    m = _HOUSE_NUM_RE.match(tokens[0])
    if m:
        house_number = m.group(1).lower()
    elif row_numbers_json:
        try:
            nums = json.loads(row_numbers_json)
            if nums and isinstance(nums, list):
                house_number = str(nums[0]).lower()
        except Exception:
            pass

    # Extract meaningful street tokens (length >= 3 and not generic street stopwords)
    street_tokens = {
        tok.lower()
        for tok in tokens
        if len(tok) >= 3 and tok.lower() not in _STREET_STOPWORDS and not tok.isdigit()
    }

    return house_number, street_tokens


def house_street_block(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    cand_source: str,
    country_aware: bool = True,
) -> pd.DataFrame:
    """
    Generate candidate pairs where S1 and candidate entity share the same house number
    and have overlapping street tokens.

    Args:
        s1_df: DataFrame of S1 query records.
        cand_df: DataFrame of candidate records (S2 or S3).
        cand_source: Label ('S2' or 'S3').
        country_aware: Whether to restrict matches to the same country.

    Returns:
        DataFrame of candidate pairs for the 'house_street' channel.
    """
    logger.info("Running Channel [House + Street Block] for S1 -> %s...", cand_source)

    if s1_df.empty or cand_df.empty:
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "candidate_source", "country", "blocking_channel"])

    country_col = "country_normalized"
    addr_col = "business_address_normalized"
    num_col = "row_numbers" if "row_numbers" in cand_df.columns else ""

    # Build candidate index: (country, house_number) -> list of (entity_id, street_tokens)
    cand_index: dict[tuple[str, str], list[tuple[str, set[str]]]] = defaultdict(list)

    for _, row in cand_df.iterrows():
        addr = str(row.get(addr_col, "")).strip()
        row_nums = str(row.get(num_col, "")) if num_col else ""
        country = str(row.get(country_col, "")).strip()

        house_num, street_tokens = extract_house_and_street(addr, row_nums)
        if not house_num or not street_tokens:
            continue

        key = (country, house_num) if country_aware else ("", house_num)
        cand_index[key].append((row["entity_id"], street_tokens))

    # Query with S1
    results: list[dict[str, str]] = []
    for _, row in s1_df.iterrows():
        addr = str(row.get(addr_col, "")).strip()
        row_nums = str(row.get(num_col, "")) if num_col else ""
        country = str(row.get(country_col, "")).strip()

        house_num, s1_tokens = extract_house_and_street(addr, row_nums)
        if not house_num or not s1_tokens:
            continue

        key = (country, house_num) if country_aware else ("", house_num)
        candidates_with_same_house = cand_index.get(key, [])

        for cand_id, cand_tokens in candidates_with_same_house:
            # Check for at least 1 overlapping street token
            if s1_tokens & cand_tokens:
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

    logger.info("Channel [House + Street Block] generated %d candidate pairs for S1 -> %s", len(res_df), cand_source)
    return res_df
