"""Unit tests for src/data/dataset.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import KneeStudyDataset, split_studies
from tests.fixtures.synthetic_dicom import make_synthetic_series

LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

DATA_CFG = {
    "image_size": 16,
    "clip_percentile": (0.5, 99.5),
    "normalize": "per_slice",
    "max_slices": 4,
    "max_series": 3,
    "slice_sampling_strategy": "uniform",
}


def _make_study_dicom(dicom_root, study_uid, series_specs, seed=0):
    """series_specs: list of (series_uid, plane, n_slices)."""
    for i, (series_uid, plane, n_slices) in enumerate(series_specs):
        make_synthetic_series(
            dicom_root / study_uid / series_uid, n_slices=n_slices, rows=32, cols=32,
            plane=plane, seed=seed + i,
        )


def _series_row(study_uid, series_uid, plane, fluid=1, fat=0):
    return {
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": series_uid,
        "Fluid_Sensitive": fluid,
        "Fat_Suppression": fat,
        "Anatomical_Plane": plane,
    }


def _study_row(study_uid, labels=None, report=""):
    row = {"StudyInstanceUID": study_uid, "Report": report}
    row.update({col: (labels.get(col) if labels else np.nan) for col in LABEL_COLS})
    return row


def test_single_study_single_series_shapes(tmp_path):
    dicom_root = tmp_path / "dicom"
    study_uid = "study-1"
    _make_study_dicom(dicom_root, study_uid, [("series-1", "Sagittal", 6)])

    studies_df = pd.DataFrame([_study_row(study_uid, labels={"ACL": 1.0})])
    series_df = pd.DataFrame([_series_row(study_uid, "series-1", "Sagittal")])

    ds = KneeStudyDataset(studies_df, series_df, dicom_root, tmp_path / "cache", DATA_CFG, LABEL_COLS)
    item = ds[0]

    assert item["pixel_values"].shape == (3, 4, 1, 16, 16)
    assert item["slice_mask"].shape == (3, 4)
    assert item["series_mask"].shape == (3,)
    assert item["series_mask"].sum().item() == 1
    assert item["slice_mask"][0].sum().item() == 4  # min(max_slices=4, n_real=6)
    assert item["slice_mask"][1:].sum().item() == 0  # padded series entirely unmarked
    assert item["labels"].shape == (12,)
    assert item["label_mask"][0].item() == 1.0
    assert item["label_mask"].sum().item() == 1.0  # only ACL was labeled


def test_multi_series_padding_and_masks(tmp_path):
    dicom_root = tmp_path / "dicom"
    study_uid = "study-1"
    specs = [("s1", "Sagittal", 5), ("s2", "Axial", 3)]
    _make_study_dicom(dicom_root, study_uid, specs)

    studies_df = pd.DataFrame([_study_row(study_uid)])
    series_df = pd.DataFrame([_series_row(study_uid, u, p) for u, p, _ in specs])

    ds = KneeStudyDataset(studies_df, series_df, dicom_root, tmp_path / "cache", DATA_CFG, LABEL_COLS)
    item = ds[0]

    assert item["series_mask"].sum().item() == 2
    assert item["series_mask"][2].item() is False or item["series_mask"][2].item() == 0
    # variable real slice counts within each series, correctly reflected
    assert item["slice_mask"][0].sum().item() == min(5, DATA_CFG["max_slices"])
    assert item["slice_mask"][1].sum().item() == min(3, DATA_CFG["max_slices"])


def test_variable_slice_count_under_max_is_not_padded_as_full(tmp_path):
    dicom_root = tmp_path / "dicom"
    study_uid = "study-1"
    _make_study_dicom(dicom_root, study_uid, [("s1", "Sagittal", 2)])  # fewer than max_slices=4

    studies_df = pd.DataFrame([_study_row(study_uid)])
    series_df = pd.DataFrame([_series_row(study_uid, "s1", "Sagittal")])

    ds = KneeStudyDataset(studies_df, series_df, dicom_root, tmp_path / "cache", DATA_CFG, LABEL_COLS)
    item = ds[0]
    assert item["slice_mask"][0].sum().item() == 2
    assert item["slice_mask"][0, 2:].sum().item() == 0


def test_series_overflow_is_truncated_deterministically(tmp_path):
    dicom_root = tmp_path / "dicom"
    study_uid = "study-1"
    # 5 series, max_series=3 -> overflow must be handled deterministically.
    specs = [("s1", "Sagittal", 4), ("s2", "Sagittal", 8), ("s3", "Axial", 2), ("s4", "Coronal", 6), ("s5", "Coronal", 1)]
    _make_study_dicom(dicom_root, study_uid, specs)

    studies_df = pd.DataFrame([_study_row(study_uid)])
    series_df = pd.DataFrame([_series_row(study_uid, u, p) for u, p, _ in specs])

    ds1 = KneeStudyDataset(studies_df, series_df, dicom_root, tmp_path / "cache", DATA_CFG, LABEL_COLS)
    ds2 = KneeStudyDataset(studies_df, series_df, dicom_root, tmp_path / "cache", DATA_CFG, LABEL_COLS)

    item1, item2 = ds1[0], ds2[0]
    assert item1["series_mask"].sum().item() == 3
    assert bool((item1["series_mask"] == item2["series_mask"]).all())
    assert bool((item1["plane_id"] == item2["plane_id"]).all())
    assert bool((item1["pixel_values"] == item2["pixel_values"]).all())


def test_all_nan_label_row_has_empty_label_mask(tmp_path):
    dicom_root = tmp_path / "dicom"
    study_uid = "study-1"
    _make_study_dicom(dicom_root, study_uid, [("s1", "Sagittal", 3)])

    studies_df = pd.DataFrame([_study_row(study_uid, labels=None)])  # all-NaN labels
    series_df = pd.DataFrame([_series_row(study_uid, "s1", "Sagittal")])

    ds = KneeStudyDataset(studies_df, series_df, dicom_root, tmp_path / "cache", DATA_CFG, LABEL_COLS)
    item = ds[0]
    assert item["label_mask"].sum().item() == 0.0
    assert bool((item["labels"] == 0.0).all())


def test_report_text_empty_when_flag_off(tmp_path):
    dicom_root = tmp_path / "dicom"
    study_uid = "study-1"
    _make_study_dicom(dicom_root, study_uid, [("s1", "Sagittal", 3)])

    studies_df = pd.DataFrame([_study_row(study_uid, report="ACL tear present.")])
    series_df = pd.DataFrame([_series_row(study_uid, "s1", "Sagittal")])

    ds_off = KneeStudyDataset(studies_df, series_df, dicom_root, tmp_path / "cache", DATA_CFG, LABEL_COLS, include_report_text=False)
    assert ds_off[0]["report_text"] == ""

    ds_on = KneeStudyDataset(studies_df, series_df, dicom_root, tmp_path / "cache", DATA_CFG, LABEL_COLS, include_report_text=True, split="train")
    assert ds_on[0]["report_text"] == "ACL tear present."

    ds_test_split = KneeStudyDataset(studies_df, series_df, dicom_root, tmp_path / "cache", DATA_CFG, LABEL_COLS, include_report_text=True, split="test")
    assert ds_test_split[0]["report_text"] == ""


def test_study_with_no_series_rows_returns_all_padded(tmp_path):
    dicom_root = tmp_path / "dicom"
    studies_df = pd.DataFrame([_study_row("study-orphan")])
    series_df = pd.DataFrame(columns=["StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"])

    ds = KneeStudyDataset(studies_df, series_df, dicom_root, tmp_path / "cache", DATA_CFG, LABEL_COLS)
    item = ds[0]
    assert item["series_mask"].sum().item() == 0
    assert item["slice_mask"].sum().item() == 0


# -- split_studies -----------------------------------------------------------

def test_split_studies_is_study_level_and_exhaustive():
    studies_df = pd.DataFrame({"StudyInstanceUID": [f"study-{i}" for i in range(20)]})
    train_df, val_df = split_studies(studies_df, val_fraction=0.25, seed=42)

    assert len(train_df) + len(val_df) == 20
    assert len(val_df) == 5
    assert set(train_df["StudyInstanceUID"]).isdisjoint(set(val_df["StudyInstanceUID"]))


def test_split_studies_is_reproducible_given_seed():
    studies_df = pd.DataFrame({"StudyInstanceUID": [f"study-{i}" for i in range(30)]})
    train1, val1 = split_studies(studies_df, val_fraction=0.2, seed=7)
    train2, val2 = split_studies(studies_df, val_fraction=0.2, seed=7)
    assert list(train1["StudyInstanceUID"]) == list(train2["StudyInstanceUID"])
    assert list(val1["StudyInstanceUID"]) == list(val2["StudyInstanceUID"])


def test_split_studies_raises_on_duplicate_study_uid():
    studies_df = pd.DataFrame({"StudyInstanceUID": ["a", "b", "a"]})
    with pytest.raises(ValueError, match="Duplicate"):
        split_studies(studies_df, val_fraction=0.2, seed=0)
