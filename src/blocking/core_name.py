"""
src/blocking/core_name.py

Channel 2: Core Name Blocking.
Generates candidate pairs when the core business name (stripped of legal corporate
suffixes such as LLC, Inc, Corp, Ltd, Pvt, etc.) matches.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

CHANNEL_NAME = "core_name"

# Common legal entity designators and corporate suffixes across countries
_LEGAL_SUFFIXES_RE = re.compile(
    r"\b("
    r"private limited|pvt ltd|pvt limited|ltd|limited|inc|incorporated|"
    r"corp|corporation|llc|l l c|llp|l l p|co|company|gmbh|sarl|s a r l|"
    r"sa|s a|ag|a g|bv|b v|nv|n v|plc|p l c|holding|holdings|group|"
    r"enterprises|services|solutions|management|consulting|international|intl"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)


def extract_core_name(name_normalized: str) -> str:
    """
    Extract the core business name by stripping common corporate suffixes
    and collapsing whitespace.

    Examples:
        "abc private limited" -> "abc"
        "abc pvt ltd"         -> "abc"
        "abc sarl"            -> "abc"
        "prime money llc"     -> "prime money"
        "custom wealth services llc" -> "custom wealth"
    """
    if not name_normalized:
        return ""
    # Strip suffixes
    core = _LEGAL_SUFFIXES_RE.sub(" ", name_normalized)
    # Collapse whitespace and strip
    core = " ".join(core.split()).strip()
    return core if len(core) >= 2 else name_normalized.strip()


def core_name_block(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    cand_source: str,
    country_aware: bool = True,
) -> pd.DataFrame:
    """
    Generate candidate pairs where S1 and candidate entity share the same core name.

    Args:
        s1_df: DataFrame of S1 query records.
        cand_df: DataFrame of candidate records (S2 or S3).
        cand_source: Label ('S2' or 'S3').
        country_aware: Whether to restrict matches to the same country.

    Returns:
        DataFrame of candidate pairs for the 'core_name' channel.
    """
    logger.info("Running Channel [Core Name Block] for S1 -> %s...", cand_source)

    if s1_df.empty or cand_df.empty:
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "candidate_source", "country", "blocking_channel"])

    country_col = "country_normalized"
    name_col = "business_name_normalized"

    # Build candidate index
    cand_index: dict[tuple[str, str], list[str]] = defaultdict(list)

    for _, row in cand_df[[name_col, country_col, "entity_id"]].iterrows():
        raw_name = str(row[name_col]).strip()
        country = str(row[country_col]).strip()
        core = extract_core_name(raw_name)
        if not core:
            continue
        key = (country, core) if country_aware else ("", core)
        cand_index[key].append(row["entity_id"])

    # Query with S1
    results: list[dict[str, str]] = []
    for _, row in s1_df[[name_col, country_col, "entity_id"]].iterrows():
        raw_name = str(row[name_col]).strip()
        country = str(row[country_col]).strip()
        core = extract_core_name(raw_name)
        if not core:
            continue
        key = (country, core) if country_aware else ("", core)
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

    logger.info("Channel [Core Name Block] generated %d candidate pairs for S1 -> %s", len(res_df), cand_source)
    return res_df
