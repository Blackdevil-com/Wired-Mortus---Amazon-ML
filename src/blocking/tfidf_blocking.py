"""
src/blocking/tfidf_blocking.py

Channel 5: Character-level TF-IDF Retrieval.
Uses character n-grams (default: 3-5) and sparse cosine similarity to retrieve
the Top-K (default: 30) most similar candidates per S1 entity.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

CHANNEL_NAME = "tfidf"


def build_tfidf_text(df: pd.DataFrame) -> pd.Series:
    """Combine normalized name and address into a single text string."""
    name_col = "business_name_normalized" if "business_name_normalized" in df.columns else "business_name"
    addr_col = "business_address_normalized" if "business_address_normalized" in df.columns else "business_address"

    name_part = df[name_col].fillna("").astype(str)
    addr_part = df[addr_col].fillna("").astype(str)
    return (name_part + " " + addr_part).str.strip()


def _retrieve_top_k_sparse(
    s1_texts: list[str],
    s1_ids: list[str],
    cand_texts: list[str],
    cand_ids: list[str],
    cand_source: str,
    country: str,
    ngram_range: tuple[int, int] = (3, 5),
    top_k: int = 30,
    min_sim: float = 0.15,
) -> list[dict]:
    """Retrieve top-K candidates using sparse TF-IDF matrix multiplication."""
    if not s1_texts or not cand_texts:
        return []

    # Fit vectorizer on candidate texts + s1 texts to ensure vocabulary coverage
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=1,
        sublinear_tf=True,
    )
    cand_matrix = vectorizer.fit_transform(cand_texts)
    s1_matrix = vectorizer.transform(s1_texts)

    # Compute similarity: (N_s1, N_cand) = s1_matrix * cand_matrix.T
    sim_matrix = s1_matrix.dot(cand_matrix.T)

    results: list[dict] = []
    num_cands = len(cand_ids)
    actual_k = min(top_k, num_cands)

    for i in range(s1_matrix.shape[0]):
        row_sims = sim_matrix.getrow(i).toarray().ravel()
        if actual_k == num_cands:
            top_indices = np.argsort(-row_sims)[:actual_k]
        else:
            partitioned = np.argpartition(-row_sims, actual_k)[:actual_k]
            top_indices = partitioned[np.argsort(-row_sims[partitioned])]

        rank = 1
        s1_id = s1_ids[i]
        for idx in top_indices:
            score = float(row_sims[idx])
            if score < min_sim:
                continue
            results.append({
                "s1_entity_id": s1_id,
                "candidate_entity_id": cand_ids[idx],
                "candidate_source": cand_source,
                "country": country,
                "blocking_channel": CHANNEL_NAME,
                "tfidf_rank": rank,
                "tfidf_similarity": round(score, 4),
            })
            rank += 1

    return results


def tfidf_block(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    cand_source: str,
    top_k: int = 30,
    ngram_range: tuple[int, int] = (3, 5),
    min_sim: float = 0.15,
    country_aware: bool = True,
) -> pd.DataFrame:
    """
    Perform character TF-IDF candidate retrieval.

    Args:
        s1_df: DataFrame of S1 query records.
        cand_df: DataFrame of candidate records (S2 or S3).
        cand_source: 'S2' or 'S3'.
        top_k: Number of candidates to retrieve per S1 entity.
        ngram_range: Character n-gram range for TfidfVectorizer.
        min_sim: Minimum cosine similarity threshold.
        country_aware: If True, partition by country.

    Returns:
        DataFrame of candidate pairs with tfidf_rank and tfidf_similarity.
    """
    logger.info("Running Channel [TF-IDF Retrieval (top_k=%d)] for S1 -> %s...", top_k, cand_source)

    if s1_df.empty or cand_df.empty:
        return pd.DataFrame(columns=[
            "s1_entity_id", "candidate_entity_id", "candidate_source",
            "country", "blocking_channel", "tfidf_rank", "tfidf_similarity",
        ])

    s1_copy = s1_df.copy()
    cand_copy = cand_df.copy()

    s1_copy["_tfidf_text"] = build_tfidf_text(s1_copy)
    cand_copy["_tfidf_text"] = build_tfidf_text(cand_copy)

    country_col = "country_normalized"

    all_results: list[dict] = []

    if country_aware and country_col in s1_copy.columns and country_col in cand_copy.columns:
        # Group by country
        s1_by_country = defaultdict(list)
        for _, row in s1_copy[["entity_id", "_tfidf_text", country_col]].iterrows():
            s1_by_country[str(row[country_col]).strip()].append((row["entity_id"], row["_tfidf_text"]))

        cand_by_country = defaultdict(list)
        for _, row in cand_copy[["entity_id", "_tfidf_text", country_col]].iterrows():
            cand_by_country[str(row[country_col]).strip()].append((row["entity_id"], row["_tfidf_text"]))

        for country, s1_items in s1_by_country.items():
            cand_items = cand_by_country.get(country, [])
            if not cand_items:
                continue

            s1_ids = [item[0] for item in s1_items]
            s1_texts = [item[1] for item in s1_items]
            cand_ids = [item[0] for item in cand_items]
            cand_texts = [item[1] for item in cand_items]

            country_res = _retrieve_top_k_sparse(
                s1_texts=s1_texts,
                s1_ids=s1_ids,
                cand_texts=cand_texts,
                cand_ids=cand_ids,
                cand_source=cand_source,
                country=country,
                ngram_range=ngram_range,
                top_k=top_k,
                min_sim=min_sim,
            )
            all_results.extend(country_res)
    else:
        # Global retrieval
        s1_ids = list(s1_copy["entity_id"])
        s1_texts = list(s1_copy["_tfidf_text"])
        cand_ids = list(cand_copy["entity_id"])
        cand_texts = list(cand_copy["_tfidf_text"])
        all_results = _retrieve_top_k_sparse(
            s1_texts=s1_texts,
            s1_ids=s1_ids,
            cand_texts=cand_texts,
            cand_ids=cand_ids,
            cand_source=cand_source,
            country="",
            ngram_range=ngram_range,
            top_k=top_k,
            min_sim=min_sim,
        )

    res_df = pd.DataFrame(all_results)
    if res_df.empty:
        res_df = pd.DataFrame(columns=[
            "s1_entity_id", "candidate_entity_id", "candidate_source",
            "country", "blocking_channel", "tfidf_rank", "tfidf_similarity",
        ])
    else:
        res_df = res_df.drop_duplicates(subset=["s1_entity_id", "candidate_entity_id"]).reset_index(drop=True)

    logger.info("Channel [TF-IDF Retrieval] generated %d candidate pairs for S1 -> %s", len(res_df), cand_source)
    return res_df
