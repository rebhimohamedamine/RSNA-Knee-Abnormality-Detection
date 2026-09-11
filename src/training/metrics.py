"""Evaluation metrics. Macro ROC-AUC is the primary competition metric;
everything else here is for secondary analysis.

Every per-target function is mask-aware: a row with no ground truth for a
given target (`label_mask == 0`) is excluded from that target's metric
entirely, not counted as a negative. A target with fewer than 2 labeled rows,
or only one class present among its labeled rows, has an undefined ROC-AUC
(sklearn would raise) -- it's returned as `NaN` with a logged warning rather
than crashing the whole evaluation.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _valid_mask_for_target(label_mask: np.ndarray, i: int) -> np.ndarray:
    return label_mask[:, i].astype(bool)


def per_target_roc_auc(y_true: np.ndarray, y_prob: np.ndarray, label_mask: np.ndarray, label_names: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for i, name in enumerate(label_names):
        mask = _valid_mask_for_target(label_mask, i)
        y, p = y_true[mask, i], y_prob[mask, i]
        if mask.sum() < 2 or len(np.unique(y)) < 2:
            logger.warning("Target %r has an undefined ROC-AUC in this split (%d labeled rows, %d class(es)).", name, int(mask.sum()), len(np.unique(y)) if mask.sum() else 0)
            result[name] = float("nan")
            continue
        result[name] = float(roc_auc_score(y, p))
    return result


def macro_roc_auc(per_target: dict[str, float]) -> float:
    values = np.array(list(per_target.values()), dtype=float)
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def per_target_pr_auc(y_true: np.ndarray, y_prob: np.ndarray, label_mask: np.ndarray, label_names: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for i, name in enumerate(label_names):
        mask = _valid_mask_for_target(label_mask, i)
        y, p = y_true[mask, i], y_prob[mask, i]
        if mask.sum() < 2 or len(np.unique(y)) < 2:
            result[name] = float("nan")
            continue
        result[name] = float(average_precision_score(y, p))
    return result


def sensitivity_specificity_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, label_mask: np.ndarray, label_names: list[str], threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for i, name in enumerate(label_names):
        mask = _valid_mask_for_target(label_mask, i)
        y, p = y_true[mask, i], y_prob[mask, i]
        if mask.sum() == 0:
            result[name] = {"sensitivity": float("nan"), "specificity": float("nan"), "tp": 0, "fn": 0, "tn": 0, "fp": 0}
            continue
        pred = (p >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        result[name] = {
            "sensitivity": (tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
            "specificity": (tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        }
    return result


def calibration_curve_data(y_true: np.ndarray, y_prob: np.ndarray, label_mask: np.ndarray, n_bins: int = 10) -> dict[str, np.ndarray]:
    """Reliability-diagram data, pooled across every (row, target) pair with
    a valid label. `bin_confidence`/`bin_accuracy` are NaN for an empty bin
    (no predictions fell in that probability range)."""
    mask = label_mask.astype(bool).ravel()
    y, p = y_true.ravel()[mask], y_prob.ravel()[mask]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidence = np.full(n_bins, np.nan)
    bin_accuracy = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = (p >= lo) & (p <= hi if b == n_bins - 1 else p < hi)
        if in_bin.any():
            bin_confidence[b] = float(p[in_bin].mean())
            bin_accuracy[b] = float(y[in_bin].mean())
            bin_counts[b] = int(in_bin.sum())

    return {"bin_edges": bin_edges, "bin_confidence": bin_confidence, "bin_accuracy": bin_accuracy, "bin_counts": bin_counts}


def confusion_matrices(y_true: np.ndarray, y_prob: np.ndarray, label_mask: np.ndarray, label_names: list[str], threshold: float = 0.5) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for i, name in enumerate(label_names):
        mask = _valid_mask_for_target(label_mask, i)
        if mask.sum() == 0:
            result[name] = np.zeros((2, 2), dtype=int)
            continue
        y, pred = y_true[mask, i], (y_prob[mask, i] >= threshold).astype(int)
        result[name] = confusion_matrix(y, pred, labels=[0, 1])
    return result


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def compute_all_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, label_mask: np.ndarray, label_names: list[str],
    threshold: float = 0.5, calibration_bins: int = 10,
) -> dict:
    """Bundles every metric above into one JSON-serializable dict, suitable
    for `json.dump`-ing straight to `outputs/metrics/<run>/metrics.json`."""
    per_auc = per_target_roc_auc(y_true, y_prob, label_mask, label_names)
    result = {
        "macro_auc": macro_roc_auc(per_auc),
        "per_target_auc": per_auc,
        "per_target_pr_auc": per_target_pr_auc(y_true, y_prob, label_mask, label_names),
        "sensitivity_specificity": sensitivity_specificity_at_threshold(y_true, y_prob, label_mask, label_names, threshold),
        "calibration": calibration_curve_data(y_true, y_prob, label_mask, calibration_bins),
        "confusion_matrices": confusion_matrices(y_true, y_prob, label_mask, label_names, threshold),
    }
    return _to_jsonable(result)
