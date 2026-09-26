"""
src/evaluation/__init__.py

Public API for Stage 1 Evaluation.
"""
from .blocking_recall import evaluate_blocking_recall, parse_ground_truth_pairs

__all__ = [
    "evaluate_blocking_recall",
    "parse_ground_truth_pairs",
]
