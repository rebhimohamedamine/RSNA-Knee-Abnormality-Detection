"""Plotting helpers for error analysis (spec section 33).

These are diagnostic tools, not clinical explanations -- learned attention
weights show what the model leaned on, not a verified radiological finding.
All functions save a PNG to `out_path` and also return the created
`matplotlib.figure.Figure` (mainly for interactive/notebook use); none of
them are on the training or inference hot path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: no display required (CI, scripts, Kaggle)
import matplotlib.pyplot as plt
import numpy as np


def _save(fig: "plt.Figure", out_path: str | Path | None) -> "plt.Figure":
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    return fig


def plot_slice_attention(
    slices: np.ndarray,
    weights: np.ndarray,
    slice_mask: np.ndarray,
    out_path: str | Path | None = None,
    title: str | None = None,
    max_thumbnails: int = 8,
) -> "plt.Figure":
    """Show the highest-weighted real slices of one series alongside a bar
    chart of the full attention distribution.

    slices: (K, H, W); weights: (K,); slice_mask: (K,) bool -- padded slots
    are excluded from both the thumbnails and the bar chart's highlighted set.
    """
    valid = np.where(slice_mask.astype(bool))[0]
    order = valid[np.argsort(-weights[valid])][:max_thumbnails]

    fig, axes = plt.subplots(2, max(len(order), 1), figsize=(2.2 * max(len(order), 1), 5))
    if len(order) == 0:
        axes = np.array([[axes[0]], [axes[1]]]) if axes.ndim == 1 else axes

    for col in range(max(len(order), 1)):
        top_row = axes[0, col] if axes.ndim == 2 else axes[0]
        bot_row = axes[1, col] if axes.ndim == 2 else axes[1]
        if col < len(order):
            idx = order[col]
            top_row.imshow(slices[idx], cmap="gray")
            top_row.set_title(f"slice {idx}\nw={weights[idx]:.3f}", fontsize=8)
        top_row.axis("off")
        bot_row.axis("off")

    fig.suptitle(title or "Top attended slices")
    _save(fig, out_path)
    return fig


def plot_series_attention(
    series_labels: list[str],
    weights: np.ndarray,
    series_mask: np.ndarray,
    out_path: str | Path | None = None,
    title: str | None = None,
) -> "plt.Figure":
    """Bar chart of which series (e.g. "Sagittal fluid-sensitive") the
    cross-series attention / study pooling weighted most heavily. Padded
    series are dropped, not merely shown at zero, to avoid implying they were
    considered and down-weighted."""
    valid = np.where(series_mask.astype(bool))[0]
    labels = [series_labels[i] for i in valid]
    vals = weights[valid]

    fig, ax = plt.subplots(figsize=(max(4, 0.8 * len(valid)), 3))
    ax.bar(range(len(valid)), vals, color="#4C72B0")
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("weight")
    ax.set_title(title or "Series contribution")
    _save(fig, out_path)
    return fig


def plot_per_target_auc(
    per_target_auc: dict[str, float],
    out_path: str | Path | None = None,
    title: str = "Per-target ROC-AUC",
) -> "plt.Figure":
    """Horizontal bar chart of each target's AUC, with the macro average as
    a reference line. NaN entries (undefined AUC -- a target with no or
    single-class labels in this split) are plotted as zero-height bars with a
    hatch pattern rather than silently dropped, so missing coverage stays
    visible."""
    names = list(per_target_auc.keys())
    values = np.array([per_target_auc[n] for n in names], dtype=float)
    macro = np.nanmean(values) if np.any(~np.isnan(values)) else float("nan")

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(names) + 1))
    plot_vals = np.nan_to_num(values, nan=0.0)
    colors = ["#C44E52" if np.isnan(v) else "#55A868" for v in values]
    hatches = ["//" if np.isnan(v) else None for v in values]
    bars = ax.barh(names, plot_vals, color=colors)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
    if not np.isnan(macro):
        ax.axvline(macro, color="black", linestyle="--", linewidth=1, label=f"macro AUC={macro:.3f}")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("ROC-AUC")
    ax.set_title(title)
    _save(fig, out_path)
    return fig


def plot_calibration_curve(
    bin_confidence: np.ndarray,
    bin_accuracy: np.ndarray,
    out_path: str | Path | None = None,
    title: str = "Calibration",
) -> "plt.Figure":
    """Reliability diagram: predicted-probability bin means vs. observed
    positive rate in that bin, against the y=x perfectly-calibrated line."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    valid = ~np.isnan(bin_confidence) & ~np.isnan(bin_accuracy)
    ax.plot(bin_confidence[valid], bin_accuracy[valid], marker="o", color="#4C72B0")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, out_path)
    return fig
