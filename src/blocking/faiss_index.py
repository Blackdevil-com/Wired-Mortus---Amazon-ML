"""
src/blocking/faiss_index.py

FAISS vector index wrapper for dense nearest-neighbor retrieval.
Supports IndexFlatIP for normalized cosine similarity search with automatic
CPU / GPU and fallback support.
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("faiss package not found; falling back to numpy cosine retrieval.")


class VectorIndex:
    """Wrapper around FAISS / Numpy for dense similarity search."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.cand_ids: list[str] = []
        self._index = None

        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(dimension)
        else:
            self._vectors: list[np.ndarray] = []

    def add(self, vectors: np.ndarray, entity_ids: list[str]) -> None:
        """Add normalized vectors and corresponding entity_ids to index."""
        if len(vectors) == 0:
            return

        self.cand_ids.extend(entity_ids)

        if _FAISS_AVAILABLE and self._index is not None:
            # Ensure float32 contiguous array
            vecs_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
            self._index.add(vecs_f32)
        else:
            if not hasattr(self, "_vectors") or not self._vectors:
                self._vectors = [vectors]
            else:
                self._vectors.append(vectors)

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for top-K nearest neighbors.

        Args:
            query_vectors: (N_query, dim) float32 array of normalized query vectors.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            scores: (N_query, K) similarity scores.
            indices: (N_query, K) index positions in cand_ids.
        """
        k = min(top_k, len(self.cand_ids))
        if k == 0:
            return np.empty((len(query_vectors), 0)), np.empty((len(query_vectors), 0), dtype=int)

        if _FAISS_AVAILABLE and self._index is not None:
            queries_f32 = np.ascontiguousarray(query_vectors, dtype=np.float32)
            scores, indices = self._index.search(queries_f32, k)
            return scores, indices
        else:
            # Numpy matrix multiplication fallback
            matrix = np.vstack(self._vectors) if isinstance(self._vectors, list) else self._vectors
            sims = np.dot(query_vectors, matrix.T)  # (N_q, N_cand)

            indices = []
            scores = []
            for i in range(len(query_vectors)):
                row = sims[i]
                if k == len(self.cand_ids):
                    top_idx = np.argsort(-row)[:k]
                else:
                    part = np.argpartition(-row, k)[:k]
                    top_idx = part[np.argsort(-row[part])]
                indices.append(top_idx)
                scores.append(row[top_idx])

            return np.array(scores), np.array(indices)
