"""
tests/test_blocking_recall.py

Unit tests for src.evaluation.blocking_recall.
"""
import pandas as pd
import pytest

from src.evaluation.blocking_recall import evaluate_blocking_recall, parse_ground_truth_pairs


def test_parse_ground_truth_pairs():
    # JSON list format
    df1 = pd.DataFrame([
        {"source1_entity_id": "S1-001", "matched_entity_ids": '["S2-101", "S3-201"]'},
        {"source1_entity_id": "S1-002", "matched_entity_ids": "S2-102, S3-202"},
    ])
    pairs = parse_ground_truth_pairs(df1)
    assert ("S1-001", "S2-101") in pairs
    assert ("S1-001", "S3-201") in pairs
    assert ("S1-002", "S2-102") in pairs
    assert ("S1-002", "S3-202") in pairs


def test_evaluate_blocking_recall():
    gt_df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "matched_entity_ids": '["S2-101", "S3-201"]'},
        {"source1_entity_id": "S1-002", "matched_entity_ids": '["S2-102"]'},
    ])  # 3 true pairs total

    union_df = pd.DataFrame([
        {"s1_entity_id": "S1-001", "candidate_entity_id": "S2-101"},
        {"s1_entity_id": "S1-001", "candidate_entity_id": "S3-201"},
        {"s1_entity_id": "S1-001", "candidate_entity_id": "S2-999"}, # false candidate
        # S1-002 -> S2-102 is missing
    ])

    report = evaluate_blocking_recall(union_df, gt_df)
    assert report["total_true_matches"] == 3
    assert report["union_recovered"] == 2
    assert report["union_recall"] == round((2 / 3) * 100, 2)
