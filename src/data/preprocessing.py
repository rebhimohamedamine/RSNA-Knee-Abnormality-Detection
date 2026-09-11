"""MRI intensity normalization and resizing.

Kept independent of DICOM/pydicom -- these functions operate on plain numpy
arrays (already rescale-slope/intercept corrected by `src.data.dicom`), so
they're testable with synthetic arrays and reusable if a future data source
isn't DICOM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

_INTERPOLATIONS = {
    "bilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
}


@dataclass
class PreprocessConfig:
    image_size: int = 256
    clip_percentile: tuple[float, float] = field(default_factory=lambda: (0.5, 99.5))
    normalize: str = "per_slice"       # "per_slice" | "per_series" | "none"
    resize_interpolation: str = "bilinear"

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        return cls(
            image_size=d.get("image_size", 256),
            clip_percentile=tuple(d.get("clip_percentile", (0.5, 99.5))),
            normalize=d.get("normalize", "per_slice"),
            resize_interpolation=d.get("resize_interpolation", "bilinear"),
        )


def clip_and_scale(img: np.ndarray, clip_percentile: tuple[float, float]) -> np.ndarray:
    """Clip to the given percentile range, then min-max scale that clipped
    range to [0, 1]. A degenerate (constant-intensity) image maps to all
    zeros rather than dividing by zero."""
    lo_pct, hi_pct = clip_percentile
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    clipped = np.clip(img, lo, hi).astype(np.float64)
    span = hi - lo
    if span < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    scaled = (clipped - lo) / span
    # Clamp defensively: float32 round-trip of values arbitrarily close to
    # the clip bounds can otherwise land a hair outside [0, 1].
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def normalize_series(slices: list[np.ndarray], mode: str, clip_percentile: tuple[float, float]) -> list[np.ndarray]:
    """Normalize every slice of one series.

    - "per_slice": clip/scale each slice independently -- maximizes
      per-image contrast, at the cost of relative brightness across slices.
    - "per_series": compute the clip percentiles once across all slices
      concatenated, apply the same (lo, hi) to every slice -- preserves
      relative brightness across the series (e.g. a genuinely darker slice
      stays darker), at the cost of per-slice contrast.
    - "none": cast to float32 only, no clipping/scaling (caller's
      responsibility to ensure a sane input range).
    """
    if mode == "none":
        return [s.astype(np.float32) for s in slices]

    if mode == "per_slice":
        return [clip_and_scale(s, clip_percentile) for s in slices]

    if mode == "per_series":
        if len(slices) == 0:
            return []
        stacked = np.concatenate([s.ravel() for s in slices])
        lo_pct, hi_pct = clip_percentile
        lo, hi = np.percentile(stacked, [lo_pct, hi_pct])
        span = hi - lo
        if span < 1e-6:
            return [np.zeros_like(s, dtype=np.float32) for s in slices]
        return [
            np.clip((np.clip(s, lo, hi).astype(np.float64) - lo) / span, 0.0, 1.0).astype(np.float32)
            for s in slices
        ]

    raise ValueError(f"Unknown normalize mode: {mode!r} (expected per_slice | per_series | none)")


def resize_slice(img: np.ndarray, size: int, interpolation: str = "bilinear") -> np.ndarray:
    if interpolation not in _INTERPOLATIONS:
        raise ValueError(f"Unknown interpolation: {interpolation!r} (expected one of {list(_INTERPOLATIONS)})")
    resized = cv2.resize(img.astype(np.float32), (size, size), interpolation=_INTERPOLATIONS[interpolation])
    return resized.astype(np.float32)


def preprocess_series(slices: list[np.ndarray], cfg: PreprocessConfig) -> np.ndarray:
    """Normalize then resize every slice of a series.

    Returns `(N, image_size, image_size)` float32, `N = len(slices)` -- this
    runs BEFORE slice-count selection (`src.data.sampler`), so every
    available slice is preprocessed and none are discarded here.
    """
    normalized = normalize_series(slices, cfg.normalize, cfg.clip_percentile)
    resized = [resize_slice(s, cfg.image_size, cfg.resize_interpolation) for s in normalized]
    if not resized:
        return np.zeros((0, cfg.image_size, cfg.image_size), dtype=np.float32)
    return np.stack(resized, axis=0)
