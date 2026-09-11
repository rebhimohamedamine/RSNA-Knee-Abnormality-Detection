"""Unit tests for src/inference/predict.py and submission.py."""

from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import KneeStudyDataset
from src.inference.predict import run_inference
from src.inference.submission import write_submission
from src.models.knee_model import KneeModel
from src.reports.label_mapping import TARGETS as LABEL_ORDER
from src.utils.config import load_config
from tests.fixtures.synthetic_dicom import make_synthetic_series

LABEL_COLS = LABEL_ORDER


def _study_row(study_uid):
    row = {"StudyInstanceUID": study_uid, "Report": ""}
    row.update({col: np.nan for col in LABEL_COLS})
    return row


def _series_row(study_uid, series_uid, plane):
    return {
        "StudyInstanceUID": study_uid, "SeriesInstanceUID": series_uid,
        "Fluid_Sensitive": 1, "Fat_Suppression": 0, "Anatomical_Plane": plane,
    }


def _minimal_model_and_dataset(tmp_path, n_studies=2):
    from pathlib import Path
    cfg = copy.deepcopy(load_config(Path(__file__).resolve().parent.parent / "configs" / "tiny_cpu_smoke.yaml"))
    for flag in ("slice_attention", "metadata", "cross_series_attention", "pathology_attention", "report_weak_supervision", "distillation"):
        cfg["model"][flag]["enabled"] = False

    dicom_root = tmp_path / "dicom"
    study_rows, series_rows = [], []
    for i in range(n_studies):
        study_uid = f"study-{i}"
        series_uid = f"series-{i}"
        make_synthetic_series(dicom_root / study_uid / series_uid, n_slices=3, rows=32, cols=32, plane="Sagittal", seed=i)
        study_rows.append(_study_row(study_uid))
        series_rows.append(_series_row(study_uid, series_uid, "Sagittal"))

    studies_df = pd.DataFrame(study_rows)
    series_df = pd.DataFrame(series_rows)
    dataset = KneeStudyDataset(studies_df, series_df, dicom_root, tmp_path / "cache", cfg["data"], LABEL_COLS, include_report_text=False, split="test")

    model = KneeModel(cfg)
    return model, dataset


def test_run_inference_end_to_end_without_report_text(tmp_path):
    model, dataset = _minimal_model_and_dataset(tmp_path)
    assert model.report_teacher is None  # no report branch built at all -- flags off

    df = run_inference(model, dataset, device=torch.device("cpu"), batch_size=2)
    assert list(df.columns) == ["StudyInstanceUID"] + list(LABEL_ORDER)
    assert len(df) == 2
    assert set(df["StudyInstanceUID"]) == {"study-0", "study-1"}
    probs = df[LABEL_ORDER].to_numpy()
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_submission_columns_match_sample_submission_exactly(tmp_path, sample_submission_path):
    pred_df = pd.DataFrame([{"StudyInstanceUID": "unrelated-study", **{c: 0.42 for c in LABEL_ORDER}}])
    out_path = tmp_path / "submission.csv"
    write_submission(pred_df, sample_submission_path, out_path)

    written = pd.read_csv(out_path)
    expected_columns = list(pd.read_csv(sample_submission_path).columns)
    assert list(written.columns) == expected_columns


def test_missing_study_filled_with_0_5_and_warns(tmp_path, caplog):
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame([
        {"StudyInstanceUID": "present-study", **{c: 0.5 for c in LABEL_ORDER}},
        {"StudyInstanceUID": "missing-study", **{c: 0.5 for c in LABEL_ORDER}},
    ]).to_csv(sample_path, index=False)

    pred_df = pd.DataFrame([{"StudyInstanceUID": "present-study", **{c: 0.9 for c in LABEL_ORDER}}])
    out_path = tmp_path / "submission.csv"

    with caplog.at_level(logging.WARNING, logger="knee_mri"):
        write_submission(pred_df, sample_path, out_path)

    written = pd.read_csv(out_path).set_index("StudyInstanceUID")
    assert np.allclose(written.loc["present-study", LABEL_ORDER].to_numpy(dtype=float), 0.9)
    assert np.allclose(written.loc["missing-study", LABEL_ORDER].to_numpy(dtype=float), 0.5)
    assert any("missing-study" in m or "no prediction" in m for m in caplog.messages)


def test_write_submission_does_not_crash_on_all_missing(tmp_path):
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame([{"StudyInstanceUID": "study-a", **{c: 0.5 for c in LABEL_ORDER}}]).to_csv(sample_path, index=False)
    empty_pred = pd.DataFrame(columns=["StudyInstanceUID", *LABEL_ORDER])

    out_path = tmp_path / "submission.csv"
    write_submission(empty_pred, sample_path, out_path)  # must not raise
    written = pd.read_csv(out_path)
    assert len(written) == 1
    assert np.allclose(written.loc[0, LABEL_ORDER].to_numpy(dtype=float), 0.5)
