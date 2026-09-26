"""
src/preprocessing/__init__.py

Public API for the Stage 1 preprocessing package.
"""
from .pipeline import PREPROCESSING_COLS, process_file
from .text_normalizer import nfkd_lowercase, normalize, normalize_text
from .tokenizer import extract_numbers, extract_numbers_to_json, tokenize, tokenize_to_json
from .unicode_normalizer import apply_nfkd, nfkd_normalize, safe_str, strip_combining_marks

__all__ = [
    # Pipeline
    "process_file",
    "PREPROCESSING_COLS",
    # Text normalization
    "nfkd_lowercase",
    "normalize",
    "normalize_text",
    # Tokenization & number extraction
    "tokenize",
    "tokenize_to_json",
    "extract_numbers",
    "extract_numbers_to_json",
    # Unicode utilities
    "safe_str",
    "nfkd_normalize",
    "apply_nfkd",
    "strip_combining_marks",
]


