"""Unit tests for src/training/metrics.py."""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from src.training.metrics import (
    calibration_curve_data,
    compute_all_metrics,
    confusion_matrices,
    macro_roc_auc,
    per_target_roc_auc,
    sensitivity_specificity_at_threshold,
)

LABELS = ["A", "B", "C"]


def test_per_target_auc_matches_sklearn_directly():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=(40, 3)).astype(float)
    y_prob = rng.rand(40, 3)
    label_mask = np.ones_like(y_true)

    result = per_target_roc_auc(y_true, y_prob, label_mask, LABELS)
    for i, name in enumerate(LABELS):
        expected = roc_auc_score(y_true[:, i], y_prob[:, i])
        assert result[name] == expected


def test_macro_auc_is_nanmean():
    per_target = {"A": 0.8, "B": 0.6, "C": float("nan")}
    assert macro_roc_auc(per_target) == 0.7


def test_macro_auc_all_nan_returns_nan():
    assert np.isnan(macro_roc_auc({"A": float("nan"), "B": float("nan")}))


def test_single_class_target_returns_nan_without_raising():
    y_true = np.zeros((10, 1))  # only class 0 present
    y_prob = np.random.rand(10, 1)
    label_mask = np.ones_like(y_true)
    result = per_target_roc_auc(y_true, y_prob, label_mask, ["OnlyZero"])
    assert np.isnan(result["OnlyZero"])


def test_label_mask_excludes_rows_from_auc():
    y_true = np.array([[1.0], [0.0], [1.0], [0.0]])
    y_prob = np.array([[0.9], [0.8], [0.6], [0.1]])  # row 1 (0.8) would hurt AUC if included
    label_mask_full = np.ones_like(y_true)
    label_mask_excl = np.array([[1.0], [0.0], [1.0], [1.0]])

    full = per_target_roc_auc(y_true, y_prob, label_mask_full, ["X"])["X"]
    excluded = per_target_roc_auc(y_true, y_prob, label_mask_excl, ["X"])["X"]
    assert full != excluded

    expected_excluded = roc_auc_score([1.0, 1.0, 0.0], [0.9, 0.6, 0.1])
    assert excluded == expected_excluded


def test_sensitivity_specificity_hand_computed():
    y_true = np.array([[1.0], [1.0], [0.0], [0.0]])
    y_prob = np.array([[0.9], [0.3], [0.2], [0.8]])  # 1 TP, 1 FN, 1 TN, 1 FP at threshold 0.5
    label_mask = np.ones_like(y_true)
    result = sensitivity_specificity_at_threshold(y_true, y_prob, label_mask, ["X"], threshold=0.5)["X"]
    assert result["tp"] == 1 and result["fn"] == 1 and result["tn"] == 1 and result["fp"] == 1
    assert result["sensitivity"] == 0.5
    assert result["specificity"] == 0.5


def test_sensitivity_specificity_empty_target_is_nan():
    y_true = np.zeros((3, 1))
    y_prob = np.zeros((3, 1))
    label_mask = np.zeros((3, 1))
    result = sensitivity_specificity_at_threshold(y_true, y_prob, label_mask, ["X"])["X"]
    assert np.isnan(result["sensitivity"]) and np.isnan(result["specificity"])


def test_confusion_matrix_shape_and_totals():
    y_true = np.array([[1.0], [1.0], [0.0], [0.0]])
    y_prob = np.array([[0.9], [0.3], [0.2], [0.8]])
    label_mask = np.ones_like(y_true)
    cm = confusion_matrices(y_true, y_prob, label_mask, ["X"], threshold=0.5)["X"]
    assert cm.shape == (2, 2)
    assert cm.sum() == 4


def test_calibration_curve_bins_and_range():
    rng = np.random.RandomState(1)
    y_true = rng.randint(0, 2, size=(100, 1)).astype(float)
    y_prob = rng.rand(100, 1)
    label_mask = np.ones_like(y_true)
    out = calibration_curve_data(y_true, y_prob, label_mask, n_bins=5)
    assert out["bin_edges"].shape == (6,)
    assert out["bin_confidence"].shape == (5,)
    valid = ~np.isnan(out["bin_confidence"])
    assert np.all(out["bin_confidence"][valid] >= 0) and np.all(out["bin_confidence"][valid] <= 1)


def test_compute_all_metrics_is_json_serializable():
    rng = np.random.RandomState(2)
    y_true = rng.randint(0, 2, size=(30, 3)).astype(float)
    y_prob = rng.rand(30, 3)
    label_mask = np.ones_like(y_true)
    metrics = compute_all_metrics(y_true, y_prob, label_mask, LABELS)

    assert "macro_auc" in metrics
    assert set(metrics["per_target_auc"].keys()) == set(LABELS)
    json.dumps(metrics)  # must not raise
