"""
src/blocking/postal.py

Channel 4: Postal / Zip Code Blocking.
Optimized with vector extraction and index maps.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

CHANNEL_NAME = "postal"

_POSTAL_RE = re.compile(
    r"\b("
    r"\d{5}(?:-\d{4})?|"
    r"\d{6}|"
    r"[A-Z]\d[A-Z]\s?\d[A-Z]\d|"
    r"[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}"
    r")\b",
    re.IGNORECASE,
)


def extract_postal_code(address_normalized: str, row_numbers_json: str = "") -> str:
    """Extract normalized postal code from address or row numbers."""
    if address_normalized:
        m = _POSTAL_RE.search(address_normalized)
        if m:
            return m.group(1).replace(" ", "").lower()

    if row_numbers_json:
        try:
            nums = json.loads(row_numbers_json)
            if isinstance(nums, list):
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
    """Generate candidate pairs based on same postal / zip code."""
    logger.info("Running Channel [Postal Block] for S1 -> %s...", cand_source)

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

    # Build candidate index: (country, postal) -> list of entity_id
    cand_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for i in range(len(cand_ids)):
        addr = cand_addrs[i].strip()
        row_nums = cand_nums[i].strip()
        country = cand_countries[i].strip() if country_aware else ""

        postal = extract_postal_code(addr, row_nums)
        if not postal:
            continue

        cand_index[(country, postal)].append(cand_ids[i])

    # Query with S1
    s1_addrs = s1_df[addr_col].fillna("").astype(str).to_numpy() if addr_col in s1_df.columns else [""] * len(s1_df)
    s1_countries = s1_df[country_col].fillna("").astype(str).to_numpy() if country_col in s1_df.columns else [""] * len(s1_df)
    s1_nums = s1_df[num_col].fillna("").astype(str).to_numpy() if num_col in s1_df.columns else [""] * len(s1_df)
    s1_ids = s1_df["entity_id"].astype(str).to_numpy()

    s1_res: list[str] = []
    cand_res: list[str] = []
    country_res: list[str] = []

    for i in range(len(s1_ids)):
        addr = s1_addrs[i].strip()
        row_nums = s1_nums[i].strip()
        country = s1_countries[i].strip() if country_aware else ""

        postal = extract_postal_code(addr, row_nums)
        if not postal:
            continue

        matched = cand_index.get((country, postal))
        if matched:
            for cid in matched:
                s1_res.append(s1_ids[i])
                cand_res.append(cid)
                country_res.append(country)

    if not s1_res:
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "candidate_source", "country", "blocking_channel"])

    res_df = pd.DataFrame({
        "s1_entity_id": s1_res,
        "candidate_entity_id": cand_res,
        "candidate_source": cand_source,
        "country": country_res,
        "blocking_channel": CHANNEL_NAME,
    }).drop_duplicates(subset=["s1_entity_id", "candidate_entity_id"]).reset_index(drop=True)

    logger.info("Channel [Postal Block] generated %d candidate pairs for S1 -> %s", len(res_df), cand_source)
    return res_df
