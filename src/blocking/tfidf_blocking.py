"""
src/blocking/tfidf_blocking.py

Channel 5: Character-level TF-IDF Retrieval.
Uses character n-grams (default: 3-5) and chunked sparse matrix multiplication
to retrieve the Top-K (default: 30) candidates per S1 entity with bounded memory.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

CHANNEL_NAME = "tfidf"
QUERY_BATCH_SIZE = 25_000


def build_tfidf_text(df: pd.DataFrame) -> list[str]:
    """Combine normalized name and address into a single text list."""
    name_col = "business_name_normalized" if "business_name_normalized" in df.columns else "business_name"
    addr_col = "business_address_normalized" if "business_address_normalized" in df.columns else "business_address"

    names = df[name_col].fillna("").astype(str).to_numpy() if name_col in df.columns else [""] * len(df)
    addrs = df[addr_col].fillna("").astype(str).to_numpy() if addr_col in df.columns else [""] * len(df)

    return [(names[i].strip() + " " + addrs[i].strip()).strip() for i in range(len(df))]


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
    """Retrieve top-K candidates using chunked sparse TF-IDF matrix multiplication."""
    if not s1_texts or not cand_texts:
        return []

    # Fit vectorizer on candidate texts
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=1,
        sublinear_tf=True,
    )
    cand_matrix = vectorizer.fit_transform(cand_texts)

    results: list[dict] = []
    cand_ids_arr = np.array(cand_ids, dtype=object)
    total_queries = len(s1_texts)

    # Process S1 queries in batches to bound RAM
    for start_idx in range(0, total_queries, QUERY_BATCH_SIZE):
        end_idx = min(start_idx + QUERY_BATCH_SIZE, total_queries)
        batch_s1_texts = s1_texts[start_idx:end_idx]
        batch_s1_ids = s1_ids[start_idx:end_idx]

        s1_matrix = vectorizer.transform(batch_s1_texts)
        # Sparse matrix multiplication produces a scipy.sparse.csr_matrix
        sim_matrix = s1_matrix.dot(cand_matrix.T).tocsr()

        for i in range(len(batch_s1_ids)):
            row_start = sim_matrix.indptr[i]
            row_end = sim_matrix.indptr[i + 1]
            if row_start == row_end:
                continue

            row_indices = sim_matrix.indices[row_start:row_end]
            row_data = sim_matrix.data[row_start:row_end]

            # Filter by min_sim
            valid_mask = row_data >= min_sim
            if not np.any(valid_mask):
                continue

            valid_indices = row_indices[valid_mask]
            valid_data = row_data[valid_mask]

            # Top-K selection
            if len(valid_data) > top_k:
                part = np.argpartition(-valid_data, top_k)[:top_k]
                sorted_sub = part[np.argsort(-valid_data[part])]
                top_cand_idx = valid_indices[sorted_sub]
                top_scores = valid_data[sorted_sub]
            else:
                order = np.argsort(-valid_data)
                top_cand_idx = valid_indices[order]
                top_scores = valid_data[order]

            s1_id = batch_s1_ids[i]
            for rank_0, (c_idx, score) in enumerate(zip(top_cand_idx, top_scores)):
                results.append({
                    "s1_entity_id": s1_id,
                    "candidate_entity_id": cand_ids_arr[c_idx],
                    "candidate_source": cand_source,
                    "country": country,
                    "blocking_channel": CHANNEL_NAME,
                    "tfidf_rank": rank_0 + 1,
                    "tfidf_similarity": round(float(score), 4),
                })

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
    """
    logger.info("Running Channel [TF-IDF Retrieval (top_k=%d)] for S1 -> %s...", top_k, cand_source)

    if s1_df.empty or cand_df.empty:
        return pd.DataFrame(columns=[
            "s1_entity_id", "candidate_entity_id", "candidate_source",
            "country", "blocking_channel", "tfidf_rank", "tfidf_similarity",
        ])

    s1_texts = build_tfidf_text(s1_df)
    cand_texts = build_tfidf_text(cand_df)

    country_col = "country_normalized"
    all_results: list[dict] = []

    if country_aware and country_col in s1_df.columns and country_col in cand_df.columns:
        s1_countries = s1_df[country_col].fillna("").astype(str).tolist()
        cand_countries = cand_df[country_col].fillna("").astype(str).tolist()
        s1_ids = s1_df["entity_id"].astype(str).tolist()
        cand_ids = cand_df["entity_id"].astype(str).tolist()

        s1_by_country = defaultdict(list)
        for idx, c in enumerate(s1_countries):
            s1_by_country[c.strip()].append(idx)

        cand_by_country = defaultdict(list)
        for idx, c in enumerate(cand_countries):
            cand_by_country[c.strip()].append(idx)

        for country, s1_indices in s1_by_country.items():
            cand_indices = cand_by_country.get(country, [])
            if not cand_indices:
                continue

            sub_s1_ids = [s1_ids[i] for i in s1_indices]
            sub_s1_texts = [s1_texts[i] for i in s1_indices]
            sub_cand_ids = [cand_ids[i] for i in cand_indices]
            sub_cand_texts = [cand_texts[i] for i in cand_indices]

            country_res = _retrieve_top_k_sparse(
                s1_texts=sub_s1_texts,
                s1_ids=sub_s1_ids,
                cand_texts=sub_cand_texts,
                cand_ids=sub_cand_ids,
                cand_source=cand_source,
                country=country,
                ngram_range=ngram_range,
                top_k=top_k,
                min_sim=min_sim,
            )
            all_results.extend(country_res)
    else:
        s1_ids = s1_df["entity_id"].astype(str).tolist()
        cand_ids = cand_df["entity_id"].astype(str).tolist()
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
