"""
src/blocking/embedding_blocking.py

Channel 6: Multilingual Dense Embedding & FAISS Nearest-Neighbor Retrieval.
Encodes business entities using a dense pretrained embedding model and retrieves
the Top-K (default: 20) candidates per S1 entity.
Supports GPU acceleration, embedding caching, and country-partitioned search.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .faiss_index import VectorIndex

logger = logging.getLogger(__name__)

CHANNEL_NAME = "embedding"


def build_embedding_text(df: pd.DataFrame) -> list[str]:
    """Build candidate text for embedding from normalized name and address."""
    name_col = "business_name_normalized" if "business_name_normalized" in df.columns else "business_name"
    addr_col = "business_address_normalized" if "business_address_normalized" in df.columns else "business_address"

    name_part = df[name_col].fillna("").astype(str)
    addr_part = df[addr_col].fillna("").astype(str)
    return (name_part + " " + addr_part).str.strip().tolist()


def get_embedding_model(model_name: str, device: str | None = None):
    """Load SentenceTransformer model with appropriate device."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading embedding model '%s' on device '%s'...", model_name, device)
        model = SentenceTransformer(model_name, device=device)
        return model
    except ImportError as e:
        logger.warning("sentence_transformers or torch not available: %s", e)
        return None


def encode_texts(
    texts: list[str],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 256,
    device: str | None = None,
    cache_path: Path | None = None,
) -> np.ndarray:
    """
    Encode texts into normalized float32 embeddings with optional caching.
    """
    if cache_path and cache_path.exists():
        logger.info("Loading cached embeddings from %s", cache_path)
        try:
            return np.load(cache_path)
        except Exception as exc:
            logger.warning("Could not load cached embeddings (%s), re-encoding...", exc)

    model = get_embedding_model(model_name, device=device)
    if model is None:
        # Fallback dummy embeddings for lightweight test environments without torch
        logger.warning("Using deterministic hash fallback embeddings (test environment).")
        dim = 128
        vecs = []
        for t in texts:
            h = hashlib.md5(t.encode("utf-8")).digest()
            v = np.frombuffer(h * (dim // 16), dtype=np.uint8).astype(np.float32)
            norm = np.linalg.norm(v)
            vecs.append(v / (norm + 1e-9))
        arr = np.array(vecs, dtype=np.float32)
    else:
        logger.info("Encoding %d texts with batch_size=%d...", len(texts), batch_size)
        arr = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 1000,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.save(cache_path, arr)
            logger.info("Saved embeddings cache to %s", cache_path)
        except Exception as exc:
            logger.warning("Failed to save embeddings cache: %s", exc)

    return arr


def embedding_block(
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    cand_source: str,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    top_k: int = 20,
    batch_size: int = 256,
    min_sim: float = 0.20,
    device: str | None = None,
    cache_dir: str = "data/cache",
    cache_embeddings: bool = True,
    country_aware: bool = True,
) -> pd.DataFrame:
    """
    Perform dense embedding retrieval using FAISS.

    Args:
        s1_df: DataFrame of S1 query records.
        cand_df: DataFrame of candidate records (S2 or S3).
        cand_source: 'S2' or 'S3'.
        model_name: Pretrained SentenceTransformer model identifier.
        top_k: Top-K candidates per S1 entity.
        batch_size: Embedding batch size.
        min_sim: Minimum cosine similarity score.
        device: Device to use ('cuda', 'cpu', or None for auto).
        cache_dir: Directory for storing intermediate .npy embeddings.
        cache_embeddings: Whether to cache and reuse embeddings.
        country_aware: Whether to partition retrieval by country.

    Returns:
        DataFrame of candidate pairs with embedding_rank and embedding_similarity.
    """
    logger.info("Running Channel [Dense Embedding Retrieval (top_k=%d)] for S1 -> %s...", top_k, cand_source)

    if s1_df.empty or cand_df.empty:
        return pd.DataFrame(columns=[
            "s1_entity_id", "candidate_entity_id", "candidate_source",
            "country", "blocking_channel", "embedding_rank", "embedding_similarity",
        ])

    s1_texts = build_embedding_text(s1_df)
    cand_texts = build_embedding_text(cand_df)

    cache_p = Path(cache_dir) if cache_embeddings else None
    s1_cache = cache_p / f"s1_emb_{len(s1_df)}.npy" if cache_p else None
    cand_cache = cache_p / f"{cand_source.lower()}_emb_{len(cand_df)}.npy" if cache_p else None

    s1_vecs = encode_texts(s1_texts, model_name=model_name, batch_size=batch_size, device=device, cache_path=s1_cache)
    cand_vecs = encode_texts(cand_texts, model_name=model_name, batch_size=batch_size, device=device, cache_path=cand_cache)

    s1_ids = list(s1_df["entity_id"])
    cand_ids = list(cand_df["entity_id"])

    country_col = "country_normalized"
    all_results: list[dict] = []

    if country_aware and country_col in s1_df.columns and country_col in cand_df.columns:
        # Group by country
        s1_countries = s1_df[country_col].fillna("").astype(str).tolist()
        cand_countries = cand_df[country_col].fillna("").astype(str).tolist()

        s1_by_country: dict[str, list[int]] = {}
        for idx, c in enumerate(s1_countries):
            s1_by_country.setdefault(c, []).append(idx)

        cand_by_country: dict[str, list[int]] = {}
        for idx, c in enumerate(cand_countries):
            cand_by_country.setdefault(c, []).append(idx)

        for country, s1_indices in s1_by_country.items():
            cand_indices = cand_by_country.get(country, [])
            if not cand_indices:
                continue

            sub_cand_vecs = cand_vecs[cand_indices]
            sub_cand_ids = [cand_ids[i] for i in cand_indices]

            index = VectorIndex(dimension=sub_cand_vecs.shape[1])
            index.add(sub_cand_vecs, sub_cand_ids)

            sub_s1_vecs = s1_vecs[s1_indices]
            sub_s1_ids = [s1_ids[i] for i in s1_indices]

            scores, indices = index.search(sub_s1_vecs, top_k=top_k)

            for i in range(len(sub_s1_ids)):
                s1_id = sub_s1_ids[i]
                rank = 1
                for k_pos in range(scores.shape[1]):
                    score = float(scores[i, k_pos])
                    if score < min_sim:
                        continue
                    idx = indices[i, k_pos]
                    all_results.append({
                        "s1_entity_id": s1_id,
                        "candidate_entity_id": sub_cand_ids[idx],
                        "candidate_source": cand_source,
                        "country": country,
                        "blocking_channel": CHANNEL_NAME,
                        "embedding_rank": rank,
                        "embedding_similarity": round(score, 4),
                    })
                    rank += 1
    else:
        index = VectorIndex(dimension=cand_vecs.shape[1])
        index.add(cand_vecs, cand_ids)
        scores, indices = index.search(s1_vecs, top_k=top_k)

        for i in range(len(s1_ids)):
            s1_id = s1_ids[i]
            rank = 1
            for k_pos in range(scores.shape[1]):
                score = float(scores[i, k_pos])
                if score < min_sim:
                    continue
                idx = indices[i, k_pos]
                all_results.append({
                    "s1_entity_id": s1_id,
                    "candidate_entity_id": cand_ids[idx],
                    "candidate_source": cand_source,
                    "country": "",
                    "blocking_channel": CHANNEL_NAME,
                    "embedding_rank": rank,
                    "embedding_similarity": round(score, 4),
                })
                rank += 1

    res_df = pd.DataFrame(all_results)
    if res_df.empty:
        res_df = pd.DataFrame(columns=[
            "s1_entity_id", "candidate_entity_id", "candidate_source",
            "country", "blocking_channel", "embedding_rank", "embedding_similarity",
        ])
    else:
        res_df = res_df.drop_duplicates(subset=["s1_entity_id", "candidate_entity_id"]).reset_index(drop=True)

    logger.info("Channel [Dense Embedding Retrieval] generated %d candidate pairs for S1 -> %s", len(res_df), cand_source)
    return res_df
