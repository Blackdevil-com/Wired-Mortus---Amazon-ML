"""
src/blocking/__init__.py

Public API for Stage 1: Wide Blocking.
"""
from .candidate_union import FINAL_COLUMNS, union_candidates
from .config import BlockingConfig
from .core_name import core_name_block, extract_core_name
from .embedding_blocking import embedding_block, encode_texts
from .exact_name import exact_name_block
from .faiss_index import VectorIndex
from .house_street import extract_house_and_street, house_street_block
from .pipeline import run_stage1_blocking
from .postal import extract_postal_code, postal_block
from .tfidf_blocking import tfidf_block

__all__ = [
    "BlockingConfig",
    "FINAL_COLUMNS",
    "run_stage1_blocking",
    "union_candidates",
    "exact_name_block",
    "core_name_block",
    "extract_core_name",
    "house_street_block",
    "extract_house_and_street",
    "postal_block",
    "extract_postal_code",
    "tfidf_block",
    "embedding_block",
    "encode_texts",
    "VectorIndex",
]
