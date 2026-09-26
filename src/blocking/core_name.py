"""
src/blocking/core_name.py

Channel 2: Core Name Blocking.
Optimized with vector extraction and hash-based indexing.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

CHANNEL_NAME = "core_name"

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
    """Extract core business name by stripping common corporate suffixes."""
    if not name_normalized:
        return ""
    core = _LEGAL_SUFFIXES_RE.sub(" ", name_normalized)
    core = " ".join(core.split()).strip()
    return core if len(core) >= 2 else name_normalized.strip()


def core_name_block(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    cand_source: str,
    country_aware: bool = True,
) -> pd.DataFrame:
    """Generate candidate pairs where S1 and candidate entity share the same core name."""
    logger.info("Running Channel [Core Name Block] for S1 -> %s...", cand_source)

    if s1_df.empty or cand_df.empty:
        return pd.DataFrame(columns=["s1_entity_id", "candidate_entity_id", "candidate_source", "country", "blocking_channel"])

    country_col = "country_normalized"
    name_col = "business_name_normalized"

    # Fast vector extraction
    cand_names = cand_df[name_col].fillna("").astype(str).to_numpy()
    cand_countries = cand_df[country_col].fillna("").astype(str).to_numpy() if country_col in cand_df.columns else [""] * len(cand_df)
    cand_ids = cand_df["entity_id"].astype(str).to_numpy()

    cand_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for i in range(len(cand_ids)):
        raw_name = cand_names[i].strip()
        if not raw_name:
            continue
        core = extract_core_name(raw_name)
        if not core:
            continue
        country = cand_countries[i].strip() if country_aware else ""
        cand_index[(country, core)].append(cand_ids[i])

    # Query with S1
    s1_names = s1_df[name_col].fillna("").astype(str).to_numpy()
    s1_countries = s1_df[country_col].fillna("").astype(str).to_numpy() if country_col in s1_df.columns else [""] * len(s1_df)
    s1_ids = s1_df["entity_id"].astype(str).to_numpy()

    s1_res: list[str] = []
    cand_res: list[str] = []
    country_res: list[str] = []

    for i in range(len(s1_ids)):
        raw_name = s1_names[i].strip()
        if not raw_name:
            continue
        core = extract_core_name(raw_name)
        if not core:
            continue
        country = s1_countries[i].strip() if country_aware else ""
        matched = cand_index.get((country, core))
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

    logger.info("Channel [Core Name Block] generated %d candidate pairs for S1 -> %s", len(res_df), cand_source)
    return res_df
