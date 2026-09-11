"""Masking correctness tests (spec: "padded slices -> ignored, padded series
-> ignored", checked directly rather than assumed).

This file has two parts: dataset-level (padded pixel slots are provably
zero-filled and correctly unmarked) and model-level (padded content never
changes a masked module's output, added once src/models/* exists).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dataset import KneeStudyDataset
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


def _study_row(study_uid):
    row = {"StudyInstanceUID": study_uid, "Report": ""}
    row.update({col: np.nan for col in LABEL_COLS})
    return row


def _series_row(study_uid, series_uid, plane):
    return {
        "StudyInstanceUID": study_uid, "SeriesInstanceUID": series_uid,
        "Fluid_Sensitive": 1, "Fat_Suppression": 0, "Anatomical_Plane": plane,
    }


def test_padded_slice_slots_are_exactly_zero(tmp_path):
    dicom_root = tmp_path / "dicom"
    study_uid = "study-1"
    make_synthetic_series(dicom_root / study_uid / "s1", n_slices=2, rows=32, cols=32, plane="Sagittal", seed=0)

    studies_df = pd.DataFrame([_study_row(study_uid)])
    series_df = pd.DataFrame([_series_row(study_uid, "s1", "Sagittal")])
    ds = KneeStudyDataset(studies_df, series_df, dicom_root, tmp_path / "cache", DATA_CFG, LABEL_COLS)
    item = ds[0]

    # max_slices=4, only 2 real slices -> slots [2, 4) of series 0 are padding.
    assert bool((item["pixel_values"][0, 2:] == 0).all())
    assert not bool(item["slice_mask"][0, 2:].any())


def test_padded_series_slots_are_exactly_zero(tmp_path):
    dicom_root = tmp_path / "dicom"
    study_uid = "study-1"
    make_synthetic_series(dicom_root / study_uid / "s1", n_slices=3, rows=32, cols=32, plane="Axial", seed=1)

    studies_df = pd.DataFrame([_study_row(study_uid)])
    series_df = pd.DataFrame([_series_row(study_uid, "s1", "Axial")])
    ds = KneeStudyDataset(studies_df, series_df, dicom_root, tmp_path / "cache", DATA_CFG, LABEL_COLS)
    item = ds[0]

    # max_series=3, only 1 real series -> series slots [1, 3) are padding.
    assert bool((item["pixel_values"][1:] == 0).all())
    assert not bool(item["series_mask"][1:].any())
    assert not bool(item["slice_mask"][1:].any())
