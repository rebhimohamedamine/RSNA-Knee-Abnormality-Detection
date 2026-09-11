"""Unit tests for scripts/preprocess.py's disk-safety and error-resilience
logic. scripts/ is deliberately not a package (it's a thin CLI layer, not a
library), so the module is loaded dynamically, matching
tests/test_end_to_end_smoke.py's approach.
"""

from __future__ import annotations

import importlib.util
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.series import SeriesMetadata
from tests.fixtures.synthetic_dicom import make_synthetic_series

RESEARCH_ROOT = Path(__file__).resolve().parent.parent
_DiskUsage = namedtuple("_DiskUsage", ["total", "used", "free"])


def _load_preprocess_module():
    spec = importlib.util.spec_from_file_location("_test_preprocess_script", RESEARCH_ROOT / "scripts" / "preprocess.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_disk_space_raises_when_low(tmp_path):
    mod = _load_preprocess_module()
    with patch.object(mod.shutil, "disk_usage", return_value=_DiskUsage(100, 99, 1 * 1024**2)):
        with pytest.raises(RuntimeError, match="free"):
            mod.check_disk_space(tmp_path, min_free_bytes=2 * 1024**3)


def test_check_disk_space_passes_with_headroom(tmp_path):
    mod = _load_preprocess_module()
    mod.check_disk_space(tmp_path, min_free_bytes=1)  # trivially satisfied on any real filesystem


def test_process_one_series_returns_error_tuple_on_exception(tmp_path):
    mod = _load_preprocess_module()
    series_dir = tmp_path / "study" / "series"
    series_dir.mkdir(parents=True)
    metadata = SeriesMetadata("series", "study", "Sagittal", 1, 0)

    with patch.object(mod, "get_or_build_series_tensor", side_effect=RuntimeError("disk full")):
        processed, error = mod._process_one_series(("study", "series", series_dir, metadata, {}, tmp_path))
    assert processed is False
    assert error is not None and "disk full" in error


def test_process_one_series_missing_dir_is_not_an_error():
    mod = _load_preprocess_module()
    metadata = SeriesMetadata("series", "study", "Sagittal", 1, 0)
    processed, error = mod._process_one_series(("study", "series", Path("/does/not/exist"), metadata, {}, Path(".")))
    assert processed is False
    assert error is None


def test_preprocess_split_circuit_breaker_aborts_after_repeated_errors(tmp_path):
    mod = _load_preprocess_module()
    mod.MAX_CONSECUTIVE_ERRORS = 3  # keep the test fast

    dicom_root = tmp_path / "data" / "train_series"
    (tmp_path / "data").mkdir(parents=True)
    study_rows, series_rows = [], []
    for i in range(10):
        study_uid, series_uid = f"study-{i}", f"series-{i}"
        make_synthetic_series(dicom_root / study_uid / series_uid, n_slices=2, rows=16, cols=16, plane="Sagittal", seed=i)
        series_rows.append({"StudyInstanceUID": study_uid, "SeriesInstanceUID": series_uid, "Fluid_Sensitive": 1, "Fat_Suppression": 0, "Anatomical_Plane": "Sagittal"})
        study_rows.append({"StudyInstanceUID": study_uid, "Report": ""})

    pd.DataFrame(study_rows).to_csv(tmp_path / "data" / "train.csv", index=False)
    pd.DataFrame(series_rows).to_csv(tmp_path / "data" / "train_series.csv", index=False)

    cfg = {
        "paths": {
            "train_csv": str(tmp_path / "data" / "train.csv"),
            "train_series_csv": str(tmp_path / "data" / "train_series.csv"),
            "dicom_root_train": str(dicom_root),
            "cache_root": str(tmp_path / "cache"),
        },
        "data": {
            "max_studies": None, "split_seed": 0, "num_workers": 0,
            "image_size": 16, "max_slices": 4, "slice_sampling_strategy": "uniform",
            "clip_percentile": (0.5, 99.5), "normalize": "per_slice",
        },
    }

    with patch.object(mod, "get_or_build_series_tensor", side_effect=RuntimeError("simulated disk full")):
        with pytest.raises(RuntimeError, match="in a row failed"):
            mod.preprocess_split(cfg, "train", research_root=tmp_path)
