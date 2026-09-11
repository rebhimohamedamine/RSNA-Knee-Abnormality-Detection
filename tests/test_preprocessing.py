"""Unit tests for src/data/preprocessing.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.preprocessing import PreprocessConfig, clip_and_scale, normalize_series, preprocess_series, resize_slice


def test_clip_and_scale_output_in_unit_range():
    img = np.random.RandomState(0).normal(1000, 200, size=(32, 32)).astype(np.float32)
    scaled = clip_and_scale(img, (0.5, 99.5))
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0
    assert scaled.dtype == np.float32


def test_clip_and_scale_constant_image_is_all_zero():
    img = np.full((8, 8), 500.0, dtype=np.float32)
    scaled = clip_and_scale(img, (0.5, 99.5))
    assert np.all(scaled == 0.0)


def test_normalize_per_slice_vs_per_series_differ_on_uneven_brightness():
    rng = np.random.RandomState(1)
    dark = rng.normal(100, 10, size=(16, 16)).astype(np.float32)
    bright = rng.normal(900, 10, size=(16, 16)).astype(np.float32)
    slices = [dark, bright]

    per_slice = normalize_series(slices, "per_slice", (0.5, 99.5))
    per_series = normalize_series(slices, "per_series", (0.5, 99.5))

    # per_slice independently stretches each slice to [0,1], so both slices'
    # means end up close together despite very different raw brightness.
    assert abs(per_slice[0].mean() - per_slice[1].mean()) < 0.2
    # per_series preserves relative brightness: the originally-dark slice
    # stays much darker than the originally-bright one.
    assert per_series[0].mean() < per_series[1].mean() - 0.3


def test_normalize_none_is_passthrough_cast():
    slices = [np.full((4, 4), 7, dtype=np.uint16)]
    out = normalize_series(slices, "none", (0.5, 99.5))
    assert out[0].dtype == np.float32
    assert np.all(out[0] == 7.0)


def test_normalize_unknown_mode_raises():
    with pytest.raises(ValueError):
        normalize_series([np.zeros((2, 2))], "bogus", (0.5, 99.5))


def test_resize_slice_output_shape():
    img = np.random.RandomState(0).rand(20, 30).astype(np.float32)
    resized = resize_slice(img, size=64, interpolation="bilinear")
    assert resized.shape == (64, 64)
    assert resized.dtype == np.float32


def test_resize_slice_unknown_interpolation_raises():
    with pytest.raises(ValueError):
        resize_slice(np.zeros((4, 4), dtype=np.float32), size=8, interpolation="bogus")


def test_preprocess_series_shape_and_range():
    rng = np.random.RandomState(0)
    slices = [rng.normal(500, 100, size=(40, 50)).astype(np.float32) for _ in range(5)]
    cfg = PreprocessConfig(image_size=32, clip_percentile=(1.0, 99.0), normalize="per_slice")
    out = preprocess_series(slices, cfg)
    assert out.shape == (5, 32, 32)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_preprocess_series_empty_input():
    cfg = PreprocessConfig(image_size=16)
    out = preprocess_series([], cfg)
    assert out.shape == (0, 16, 16)


def test_preprocess_config_from_dict_defaults():
    cfg = PreprocessConfig.from_dict({"image_size": 128})
    assert cfg.image_size == 128
    assert cfg.normalize == "per_slice"
    assert cfg.clip_percentile == (0.5, 99.5)
