"""Unit tests for src/data/dicom.py, against synthetic DICOM fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.dicom import compute_slice_normal, load_dicom_series, order_slices, read_pixel_array
from tests.fixtures.synthetic_dicom import make_synthetic_series, make_synthetic_series_missing_position


def _spatial_projections(ordered):
    normal = compute_slice_normal(ordered[0].image_orientation_patient)
    return [float(np.dot(r.image_position_patient, normal)) for r in ordered]


@pytest.mark.parametrize("plane", ["Sagittal", "Axial", "Coronal"])
def test_load_and_order_by_position(tmp_path, plane):
    n = 12
    make_synthetic_series(tmp_path, n_slices=n, rows=32, cols=32, plane=plane, seed=7, step_mm=2.5)

    records = load_dicom_series(tmp_path)
    assert len(records) == n

    ordered = order_slices(records)
    projections = _spatial_projections(ordered)
    assert projections == sorted(projections)
    assert projections[-1] - projections[0] == pytest.approx((n - 1) * 2.5, abs=1e-3)


def test_ordering_does_not_depend_on_filename_order(tmp_path):
    make_synthetic_series(tmp_path, n_slices=10, rows=32, cols=32, plane="Axial", seed=3)
    records = load_dicom_series(tmp_path)
    filename_order_projections = _spatial_projections(records)
    # Filenames are UID-derived and unrelated to slice index, so loading in
    # filesystem order should generally NOT already be spatially sorted.
    assert filename_order_projections != sorted(filename_order_projections)

    ordered = order_slices(records)
    assert _spatial_projections(ordered) == sorted(filename_order_projections)


def test_order_does_not_mutate_input(tmp_path):
    make_synthetic_series(tmp_path, n_slices=6, rows=16, cols=16, plane="Sagittal", seed=1)
    records = load_dicom_series(tmp_path)
    original_order = [r.sop_instance_uid for r in records]
    order_slices(records)
    assert [r.sop_instance_uid for r in records] == original_order


def test_fallback_to_instance_number_when_position_missing(tmp_path):
    n = 8
    make_synthetic_series_missing_position(tmp_path, n_slices=n, rows=16, cols=16, seed=5)
    records = load_dicom_series(tmp_path)
    assert all(r.image_position_patient is None for r in records)

    ordered = order_slices(records)
    assert [r.instance_number for r in ordered] == list(range(1, n + 1))


def test_fallback_to_filename_when_all_metadata_missing(tmp_path, caplog):
    import logging

    make_synthetic_series_missing_position(tmp_path, n_slices=5, rows=16, cols=16, seed=9)
    records = load_dicom_series(tmp_path)
    for r in records:
        r.instance_number = None  # strip the remaining fallback signal too

    with caplog.at_level(logging.WARNING, logger="knee_mri"):
        ordered = order_slices(records)
    assert len(ordered) == 5
    assert any("filename order" in message for message in caplog.messages)


def test_single_slice_series_orders_trivially(tmp_path):
    make_synthetic_series(tmp_path, n_slices=1, rows=16, cols=16, plane="Sagittal", seed=0)
    records = load_dicom_series(tmp_path)
    ordered = order_slices(records)
    assert len(ordered) == 1


def test_rescale_slope_and_intercept_applied(tmp_path):
    make_synthetic_series(
        tmp_path, n_slices=3, rows=16, cols=16, plane="Axial", seed=2,
        rescale_slope=1.5, rescale_intercept=-50.0,
    )
    records = load_dicom_series(tmp_path)
    record = records[0]
    rescaled = read_pixel_array(record)
    expected = record.pixel_array.astype(np.float32) * 1.5 - 50.0
    assert np.allclose(rescaled, expected)
    assert rescaled.dtype == np.float32


def test_default_rescale_is_identity(tmp_path):
    make_synthetic_series(tmp_path, n_slices=2, rows=16, cols=16, plane="Coronal", seed=4)
    records = load_dicom_series(tmp_path)
    rescaled = read_pixel_array(records[0])
    assert np.allclose(rescaled, records[0].pixel_array.astype(np.float32))
