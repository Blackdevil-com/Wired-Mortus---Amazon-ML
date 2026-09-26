"""
src/blocking/house_street.py

Channel 3: House Number + Street Token Blocking.
Optimized with vector extraction and index maps.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

CHANNEL_NAME = "house_street"

_STREET_STOPWORDS = {
    "st", "street", "rd", "road", "ave", "avenue", "dr", "drive", "ln", "lane",
    "blvd", "boulevard", "ct", "court", "pl", "place", "pkwy", "parkway", "way",
    "hwy", "highway", "cir", "circle", "ste", "suite", "apt", "apartment", "unit",
    "fl", "floor", "rm", "room", "bldg", "building", "box", "po", "pob", "dept",
    "no", "num", "north", "south", "east", "west", "n", "s", "e", "w",
}

_HOUSE_NUM_RE = re.compile(r"\b(\d+[a-zA-Z]?)\b")


def extract_house_and_street(address_normalized: str, row_numbers_json: str = "") -> tuple[str, set[str]]:
    """Extract primary house number and meaningful street tokens."""
    if not address_normalized:
        return "", set()

    tokens = address_normalized.split()
    if not tokens:
        return "", set()

    house_number = ""
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
    max_candidates_per_key: int = 100,
    max_block_size: int = 1000,
) -> pd.DataFrame:
    """Generate candidate pairs based on same house number + street token overlap."""
    logger.info("Running Channel [House + Street Block] for S1 -> %s...", cand_source)

    if s1_df.empty or cand_df.empty:
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "candidate_source", "country", "blocking_channel"])

    country_col = "country_normalized"
    addr_col = "business_address_normalized"
    num_col = "row_numbers"

    # Fast vector extraction
    cand_addrs = cand_df[addr_col].fillna("").astype(str).to_numpy() if addr_col in cand_df.columns else [""] * len(cand_df)
    cand_countries = cand_df[country_col].fillna("").astype(str).to_numpy() if country_col in cand_df.columns else [""] * len(cand_df)
    cand_nums = cand_df[num_col].fillna("").astype(str).to_numpy() if num_col in cand_df.columns else [""] * len(cand_df)
    cand_ids = cand_df["entity_id"].astype(str).to_numpy()

    logger.info("  Indexing %d candidate records for House + Street...", len(cand_ids))
    cand_index: dict[tuple[str, str], list[tuple[str, set[str]]]] = defaultdict(list)
    for i in range(len(cand_ids)):
        addr = cand_addrs[i].strip()
        row_nums = cand_nums[i].strip()
        country = cand_countries[i].strip() if country_aware else ""

        house_num, street_tokens = extract_house_and_street(addr, row_nums)
        if not house_num or not street_tokens:
            continue

        cand_index[(country, house_num)].append((cand_ids[i], street_tokens))

    # Query with S1
    s1_addrs = s1_df[addr_col].fillna("").astype(str).to_numpy() if addr_col in s1_df.columns else [""] * len(s1_df)
    s1_countries = s1_df[country_col].fillna("").astype(str).to_numpy() if country_col in s1_df.columns else [""] * len(s1_df)
    s1_nums = s1_df[num_col].fillna("").astype(str).to_numpy() if num_col in s1_df.columns else [""] * len(s1_df)
    s1_ids = s1_df["entity_id"].astype(str).to_numpy()

    s1_res: list[str] = []
    cand_res: list[str] = []
    country_res: list[str] = []

    total_s1 = len(s1_ids)
    log_interval = max(500_000, total_s1 // 4)
    logger.info("  Querying %d S1 records against House + Street index...", total_s1)

    for i in range(total_s1):
        if (i + 1) % log_interval == 0:
            logger.info("  [House + Street S1 -> %s] Queried %d / %d S1 records (%d pairs found)...", cand_source, i + 1, total_s1, len(s1_res))
        addr = s1_addrs[i].strip()
        row_nums = s1_nums[i].strip()
        country = s1_countries[i].strip() if country_aware else ""

        house_num, s1_tokens = extract_house_and_street(addr, row_nums)
        if not house_num or not s1_tokens:
            continue

        candidates_with_same_house = cand_index.get((country, house_num))
        if candidates_with_same_house:
            if max_block_size > 0 and len(candidates_with_same_house) > max_block_size:
                candidates_to_check = candidates_with_same_house[:max_candidates_per_key]
            else:
                candidates_to_check = candidates_with_same_house

            matched_count = 0
            for cand_id, cand_tokens in candidates_to_check:
                if s1_tokens & cand_tokens:
                    s1_res.append(s1_ids[i])
                    cand_res.append(cand_id)
                    country_res.append(country)
                    matched_count += 1
                    if max_candidates_per_key > 0 and matched_count >= max_candidates_per_key:
                        break

    if not s1_res:
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "candidate_source", "country", "blocking_channel"])

    res_df = pd.DataFrame({
        "s1_entity_id": s1_res,
        "candidate_entity_id": cand_res,
        "candidate_source": cand_source,
        "country": country_res,
        "blocking_channel": CHANNEL_NAME,
    }).drop_duplicates(subset=["s1_entity_id", "candidate_entity_id"]).reset_index(drop=True)

    logger.info("Channel [House + Street Block] generated %d candidate pairs for S1 -> %s", len(res_df), cand_source)
    return res_df
