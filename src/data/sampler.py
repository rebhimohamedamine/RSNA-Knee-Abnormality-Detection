"""Selects which K of a series' N available slices go to the model.

Only ever returns real, sorted, unique indices into `[0, n_available)` --
padding a selection out to a configured `max_slices` is the Dataset's job
(a zero tensor + `mask=False`), never this module's, so that a "padded slot"
stays provably distinguishable from a "real slot" in the masking tests.
"""

from __future__ import annotations

import numpy as np


def _uniform_indices(start: int, end_inclusive: int, k: int) -> list[int]:
    """`k` indices spread evenly across `[start, end_inclusive]`, including
    both endpoints when `k >= 2`; the midpoint when `k == 1`."""
    if k <= 0:
        return []
    if k == 1:
        return [start + (end_inclusive - start) // 2]
    raw = np.linspace(start, end_inclusive, num=k)
    indices = sorted(set(int(round(x)) for x in raw))
    return indices


def select_slice_indices(n_available: int, k: int, strategy: str) -> list[int]:
    """Select up to `k` representative slice indices out of `n_available`.

    `n_available < k`: returns `list(range(n_available))` -- every available
    slice, no duplication, no padding (the caller pads).
    `k <= 0`: returns `[]`.

    Strategies:
      - "uniform": evenly spread across the whole series, endpoints included.
      - "center": a contiguous block of `k` indices centered on the middle
        of the series, clipped at the bounds.
      - "uniform_center": evenly spread within the series' middle 60%
        (`[round(0.2n), round(0.8n))`), which is where knee anatomy of
        interest is most consistently in-frame; falls back to full-range
        "uniform" if that sub-range has fewer than `k` slices available.
    """
    if k <= 0 or n_available <= 0:
        return []
    if n_available <= k:
        return list(range(n_available))

    if strategy == "uniform":
        return _uniform_indices(0, n_available - 1, k)

    if strategy == "center":
        start = (n_available // 2) - (k // 2)
        start = max(0, min(start, n_available - k))
        return list(range(start, start + k))

    if strategy == "uniform_center":
        sub_start = round(0.2 * n_available)
        sub_end = round(0.8 * n_available) - 1  # inclusive upper bound
        sub_end = max(sub_start, sub_end)
        if (sub_end - sub_start + 1) < k:
            return _uniform_indices(0, n_available - 1, k)
        return _uniform_indices(sub_start, sub_end, k)

    raise ValueError(f"Unknown slice_sampling_strategy: {strategy!r} (expected uniform | center | uniform_center)")
