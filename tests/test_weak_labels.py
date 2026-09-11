"""Unit tests for src/reports/weak_labels.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.reports.label_mapping import TARGETS
from src.reports.weak_labels import batch_extract_weak_labels, extract_weak_labels


def _idx(target: str) -> int:
    return TARGETS.index(target)


def test_no_mention_is_unknown_not_negative():
    labels, confidence = extract_weak_labels("The scan shows no acute abnormality otherwise.")
    assert confidence[_idx("Fracture")] == 0.0
    assert labels[_idx("Fracture")] == 0.0  # value is a placeholder; confidence=0 says "ignore"


def test_none_and_empty_text_is_all_unknown():
    for text in (None, "", "   "):
        labels, confidence = extract_weak_labels(text)
        assert labels.shape == (12,) and confidence.shape == (12,)
        assert np.all(confidence == 0.0)


def test_simple_positive_mention():
    labels, confidence = extract_weak_labels("Findings: rupture of the ACL is present.")
    assert labels[_idx("ACL")] == 1.0
    assert confidence[_idx("ACL")] == 1.0


def test_simple_negated_mention():
    labels, confidence = extract_weak_labels("No effusion.")
    assert labels[_idx("Effusion")] == 0.0
    assert confidence[_idx("Effusion")] == 1.0


def test_uncertain_language_lowers_confidence():
    labels, confidence = extract_weak_labels("Possible meniscal tear, medial meniscus.")
    assert labels[_idx("Medial Meniscus")] == 1.0
    assert confidence[_idx("Medial Meniscus")] == 0.5


def test_historical_finding_lowers_confidence_more():
    labels, confidence = extract_weak_labels("History of fracture of the femur.")
    assert labels[_idx("Fracture")] == 1.0
    assert confidence[_idx("Fracture")] == 0.3


def test_negation_does_not_bleed_across_findings():
    labels, confidence = extract_weak_labels("No ACL tear. Effusion is present.")
    assert labels[_idx("ACL")] == 0.0 and confidence[_idx("ACL")] == 1.0
    assert labels[_idx("Effusion")] == 1.0 and confidence[_idx("Effusion")] == 1.0


def test_affirmed_mention_anywhere_wins_over_earlier_negation():
    text = "No history of ACL injury. New complete tear of the ACL is identified."
    labels, confidence = extract_weak_labels(text)
    assert labels[_idx("ACL")] == 1.0
    assert confidence[_idx("ACL")] == 1.0


def test_batch_extract_shapes():
    texts = ["No effusion.", "ACL tear present.", None]
    labels, confidence = batch_extract_weak_labels(texts)
    assert labels.shape == (3, 12)
    assert confidence.shape == (3, 12)
    assert np.all(confidence[2] == 0.0)


def test_batch_extract_empty_list():
    labels, confidence = batch_extract_weak_labels([])
    assert labels.shape == (0, 12)
    assert confidence.shape == (0, 12)


def test_real_multilingual_reports_do_not_crash(train_csv_path):
    """Documents current recall limitation rather than asserting correctness
    for languages beyond the partial Spanish/Dutch coverage: the extractor
    must run cleanly over the actual multilingual report text in the
    dataset, without raising, even where it fails to find every mention."""
    df = pd.read_csv(train_csv_path, usecols=["Report"])
    sample = df["Report"].dropna().head(50).tolist()
    labels, confidence = batch_extract_weak_labels(sample)
    assert labels.shape == (len(sample), 12)
    assert np.all((labels == 0.0) | (labels == 1.0))
    assert np.all((confidence >= 0.0) & (confidence <= 1.0))


def test_spanish_report_sample_is_recognized():
    # Adapted from an actual data/train.csv row (medial meniscus tear + medial OA + effusion).
    text = (
        "Resultados: Rotura de menisco interno. "
        "Artrosis femorotibial medial. Derrame."
    )
    labels, confidence = extract_weak_labels(text)
    assert labels[_idx("Medial Meniscus")] == 1.0
    assert labels[_idx("Medial OA")] == 1.0
    assert labels[_idx("Effusion")] == 1.0
    assert confidence[_idx("Medial Meniscus")] == 1.0
