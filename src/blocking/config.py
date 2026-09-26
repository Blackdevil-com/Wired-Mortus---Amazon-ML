"""
src/blocking/config.py

Configuration dataclass for Stage 1: Wide Blocking.
Centralizes all hyperparameters, channel toggles, and retrieval thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BlockingConfig:
    # ── Channel Toggles ───────────────────────────────────────────────────────
    use_exact_name_block: bool = True
    use_core_name_block: bool = True
    use_house_street_block: bool = True
    use_postal_block: bool = True
    use_tfidf_block: bool = True
    use_embedding_block: bool = True

    # ── Country Partitioning ──────────────────────────────────────────────────
    country_aware: bool = True

    # ── High-Frequency Block Explosion Protection ─────────────────────────────
    max_candidates_per_key: int = 100  # Max candidates retrieved per query in coarse block channels
    max_block_size: int = 1000         # Skip mega-blocks with >1000 candidates (e.g. generic zip/name)

    # ── TF-IDF Retrieval Settings ─────────────────────────────────────────────
    tfidf_ngram_range: tuple[int, int] = (3, 5)
    tfidf_top_k: int = 30
    tfidf_min_sim: float = 0.15

    # ── Dense Embedding / FAISS Settings ──────────────────────────────────────
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_top_k: int = 20
    embedding_batch_size: int = 512
    embedding_min_sim: float = 0.20
    device: str | None = None  # None = auto-detect ('cuda' if available else 'cpu')

    # ── Output & Checkpointing Settings ───────────────────────────────────────
    save_debug_channels: bool = False
    output_dir: str = "data/candidates"
    cache_dir: str = "data/cache"
    cache_embeddings: bool = True

    def get_output_path(self) -> Path:
        out_p = Path(self.output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        return out_p / "stage1_candidate_union.tsv"

    def get_cache_path(self) -> Path:
        cache_p = Path(self.cache_dir)
        cache_p.mkdir(parents=True, exist_ok=True)
        return cache_p
