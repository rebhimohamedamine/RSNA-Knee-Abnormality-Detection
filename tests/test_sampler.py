"""Unit tests for src/data/sampler.py."""

from __future__ import annotations

import pytest

from src.data.sampler import select_slice_indices


@pytest.mark.parametrize("strategy", ["uniform", "center", "uniform_center"])
def test_indices_are_sorted_and_unique(strategy):
    indices = select_slice_indices(n_available=40, k=10, strategy=strategy)
    assert indices == sorted(set(indices))
    assert len(indices) == 10
    assert all(0 <= i < 40 for i in indices)


def test_uniform_includes_both_endpoints():
    indices = select_slice_indices(n_available=30, k=5, strategy="uniform")
    assert indices[0] == 0
    assert indices[-1] == 29


def test_center_returns_contiguous_middle_block():
    indices = select_slice_indices(n_available=30, k=6, strategy="center")
    assert indices == list(range(indices[0], indices[0] + 6))
    center = 30 // 2
    assert indices[0] <= center <= indices[-1]


def test_uniform_center_falls_back_when_middle_range_too_small():
    # middle 60% of 10 slices is indices [2, 7] inclusive (6 slices) -- asking
    # for more than that forces the full-range fallback.
    n = 10
    indices = select_slice_indices(n_available=n, k=9, strategy="uniform_center")
    assert len(indices) == 9
    assert indices[0] == 0
    assert indices[-1] == n - 1


def test_uniform_center_stays_within_middle_range_when_it_fits():
    n = 40
    k = 5
    indices = select_slice_indices(n_available=n, k=k, strategy="uniform_center")
    sub_start = round(0.2 * n)
    sub_end = round(0.8 * n) - 1
    assert indices[0] >= sub_start
    assert indices[-1] <= sub_end


def test_k_greater_than_n_available_returns_all_no_duplicates():
    indices = select_slice_indices(n_available=5, k=20, strategy="uniform")
    assert indices == [0, 1, 2, 3, 4]


def test_k_equals_n_available_returns_all():
    for strategy in ("uniform", "center", "uniform_center"):
        indices = select_slice_indices(n_available=7, k=7, strategy=strategy)
        assert indices == list(range(7))


def test_k_zero_or_negative_returns_empty():
    assert select_slice_indices(n_available=10, k=0, strategy="uniform") == []
    assert select_slice_indices(n_available=10, k=-3, strategy="center") == []


def test_n_available_zero_returns_empty():
    assert select_slice_indices(n_available=0, k=5, strategy="uniform") == []


def test_k_one_returns_single_representative_index():
    for strategy in ("uniform", "center", "uniform_center"):
        indices = select_slice_indices(n_available=21, k=1, strategy=strategy)
        assert len(indices) == 1
        assert 0 <= indices[0] < 21


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        select_slice_indices(n_available=10, k=3, strategy="bogus")
