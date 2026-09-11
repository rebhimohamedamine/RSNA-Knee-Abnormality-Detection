"""Unit tests for src/data/cache.py and src/data/series.py."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.data.cache import (
    cache_key,
    get_or_build_series_tensor,
    load_series_cache,
    save_series_cache,
    series_cache_path,
)
from src.data.series import SeriesMetadata, build_series_tensor
from tests.fixtures.synthetic_dicom import make_synthetic_series

BASE_DATA_CFG = {
    "image_size": 32,
    "clip_percentile": (0.5, 99.5),
    "normalize": "per_slice",
    "max_slices": 4,
    "slice_sampling_strategy": "uniform",
}


def _metadata() -> SeriesMetadata:
    return SeriesMetadata(
        series_instance_uid="series-1",
        study_instance_uid="study-1",
        anatomical_plane="Sagittal",
        fluid_sensitive=1,
        fat_suppression=0,
    )


def test_build_series_tensor_shape(tmp_path):
    make_synthetic_series(tmp_path, n_slices=10, rows=64, cols=64, plane="Sagittal", seed=0)
    result = build_series_tensor(tmp_path, _metadata(), BASE_DATA_CFG)
    assert result["pixels"].shape == (4, 32, 32)
    assert result["n_real_slices"] == 4
    assert result["plane_id"] == 0  # Sagittal
    assert result["fluid_sensitive"] == 1
    assert result["fat_suppression"] == 0


def test_cache_roundtrip_is_bit_identical(tmp_path):
    payload = {"pixels": np.random.RandomState(0).rand(4, 8, 8).astype(np.float32), "plane_id": 1, "series_uid": "abc"}
    path = tmp_path / "cache_key" / "study" / "series.pt"
    save_series_cache(path, payload)

    assert path.exists()
    assert path.with_suffix(".metadata.json").exists()

    loaded = load_series_cache(path)
    assert loaded is not None
    assert np.array_equal(loaded["pixels"], payload["pixels"])
    assert loaded["plane_id"] == 1
    assert loaded["cache_version"] == 1


def test_cache_key_changes_when_relevant_setting_changes():
    key_a = cache_key(BASE_DATA_CFG)
    key_b = cache_key({**BASE_DATA_CFG, "image_size": 64})
    assert key_a != key_b


def test_cache_key_stable_for_unrelated_settings():
    key_a = cache_key(BASE_DATA_CFG)
    key_b = cache_key({**BASE_DATA_CFG, "val_fraction": 0.5, "num_workers": 8})
    assert key_a == key_b


def test_series_cache_path_layout(tmp_path):
    path = series_cache_path(tmp_path, "abc123", "study-1", "series-1")
    assert path == tmp_path / "abc123" / "study-1" / "series-1.pt"


def test_load_missing_cache_returns_none(tmp_path):
    assert load_series_cache(tmp_path / "does_not_exist.pt") is None


def test_load_cache_wrong_version_returns_none(tmp_path):
    path = tmp_path / "series.pt"
    save_series_cache(path, {"pixels": np.zeros((1, 2, 2), dtype=np.float32)})
    import torch

    stale = torch.load(path, weights_only=False)
    stale["cache_version"] = 999
    torch.save(stale, path)
    assert load_series_cache(path) is None


def test_get_or_build_series_tensor_builds_once_then_caches(tmp_path):
    dicom_dir = tmp_path / "dicom"
    make_synthetic_series(dicom_dir, n_slices=6, rows=32, cols=32, plane="Axial", seed=1)
    cache_root = tmp_path / "cache"

    first = get_or_build_series_tensor(
        study_uid="study-1", series_uid="series-1", series_dir=dicom_dir,
        metadata=_metadata(), data_cfg=BASE_DATA_CFG, cache_root=cache_root,
    )
    assert first["pixels"].shape[0] == min(6, BASE_DATA_CFG["max_slices"])

    with patch("src.data.cache.build_series_tensor", side_effect=AssertionError("should not rebuild")):
        second = get_or_build_series_tensor(
            study_uid="study-1", series_uid="series-1", series_dir=dicom_dir,
            metadata=_metadata(), data_cfg=BASE_DATA_CFG, cache_root=cache_root,
        )
    assert np.array_equal(first["pixels"], second["pixels"])
