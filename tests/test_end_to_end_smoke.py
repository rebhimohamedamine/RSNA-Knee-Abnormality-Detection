"""The pass-closing smoke test: builds a handful of synthetic studies (2-3
series each, real DICOM files) and an in-memory train/series CSV, then runs
preprocess -> train (2 epochs) -> evaluate -> predict against
`configs/tiny_cpu_smoke.yaml` (every architectural flag ON, tiny dims)
entirely inside `tmp_path`, asserting the full chain produces a valid
`submission.csv`. CPU-only, no real Kaggle data, done in well under a minute.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.reports.label_mapping import TARGETS as LABEL_COLS
from src.utils.config import load_config
from tests.fixtures.synthetic_dicom import make_synthetic_series

RESEARCH_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = RESEARCH_ROOT / "scripts"


def _load_script(name: str):
    """Imports scripts/<name>.py as a module without relying on scripts/
    being a package (it deliberately isn't -- it's a thin CLI layer, not a
    library)."""
    spec = importlib.util.spec_from_file_location(f"_smoke_test_script_{name}", SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPORTS_BY_LANGUAGE = [
    "Findings: rupture of the ACL. No effusion.",
    "Possible medial meniscus tear. History of fracture of the tibia.",
    "Artrosis femorotibial medial. Derrame. No acl tear.",
    "Impression: no abnormality identified.",
    "Baker's cyst noted. Synovitis present.",
]


def _build_synthetic_world(root: Path, n_studies: int = 5, seed: int = 0):
    """Writes DICOM under <root>/train_series/<StudyUID>/<SeriesUID>/*.dcm
    (and a matching test_series/ tree) plus train.csv/train_series.csv/
    test.csv/test_series.csv/sample_submission.csv -- everything
    `configs/tiny_cpu_smoke.yaml`'s `paths.*` expects, resolved against
    `root` as the sandboxed research_root."""
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_dicom_root = root / "data" / "train_series"
    test_dicom_root = root / "data" / "test_series"

    planes = ["Sagittal", "Axial", "Coronal"]
    study_rows, series_rows = [], []

    for i in range(n_studies):
        study_uid = f"study-{i}"
        n_series = 2 + (i % 2)  # 2 or 3 series per study
        for s in range(n_series):
            series_uid = f"study-{i}-series-{s}"
            plane = planes[s % len(planes)]
            n_slices = 3 + ((i + s) % 4)  # variable slice counts, 3-6
            make_synthetic_series(
                train_dicom_root / study_uid / series_uid, n_slices=n_slices, rows=48, cols=48,
                plane=plane, seed=seed + i * 10 + s,
            )
            series_rows.append({
                "StudyInstanceUID": study_uid, "SeriesInstanceUID": series_uid,
                "Fluid_Sensitive": s % 2, "Fat_Suppression": (s + 1) % 2, "Anatomical_Plane": plane,
            })

        row = {"StudyInstanceUID": study_uid, "Report": REPORTS_BY_LANGUAGE[i % len(REPORTS_BY_LANGUAGE)]}
        row.update({col: np.nan for col in LABEL_COLS})
        if i % 2 == 0:  # most (but not all) studies get real ground-truth labels,
            row["ACL"] = float(i % 3 == 0)  # so this exercises the mask-aware path
            row["Effusion"] = float(i % 3 != 0)  # regardless of how the random val split falls
        study_rows.append(row)

    pd.DataFrame(study_rows).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(series_rows).to_csv(data_dir / "train_series.csv", index=False)

    # A small, separate "test" split, reusing one of the same studies' DICOM
    # so inference has something real to run against.
    test_study_uid = "study-0"
    test_series_rows = [r for r in series_rows if r["StudyInstanceUID"] == test_study_uid]
    for r in test_series_rows:
        src = train_dicom_root / test_study_uid / r["SeriesInstanceUID"]
        dst = test_dicom_root / test_study_uid / r["SeriesInstanceUID"]
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.dcm"):
            (dst / f.name).write_bytes(f.read_bytes())

    pd.DataFrame([{"StudyInstanceUID": test_study_uid}]).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame(test_series_rows).to_csv(data_dir / "test_series.csv", index=False)
    pd.DataFrame([{"StudyInstanceUID": test_study_uid, **{c: 0.5 for c in LABEL_COLS}}]).to_csv(
        data_dir / "sample_submission.csv", index=False
    )


def test_full_pipeline_preprocess_train_evaluate_predict(tmp_path):
    _build_synthetic_world(tmp_path, n_studies=5)

    cfg = copy.deepcopy(load_config(RESEARCH_ROOT / "configs" / "tiny_cpu_smoke.yaml"))
    cfg["run_name"] = "smoke_test_run"
    cfg["data"]["num_workers"] = 0
    device = torch.device("cpu")

    preprocess = _load_script("preprocess")
    preprocess.preprocess_split(cfg, "train", research_root=tmp_path)
    preprocess.preprocess_split(cfg, "test", research_root=tmp_path)

    # cache was actually populated (not silently skipped for missing dirs)
    cache_root = tmp_path / "cache"
    assert any(cache_root.rglob("*.pt"))
    assert (cache_root / "weak_labels.parquet").exists()

    train_module = _load_script("train")
    train_result = train_module.run_training(cfg, tmp_path, run_name=cfg["run_name"], device=device)
    assert len(train_result["history"]) >= 1
    assert np.isfinite(train_result["history"][0]["train_losses"]["total"])

    best_ckpt = train_result["checkpoint_dir"] / "best.pt"
    assert best_ckpt.exists()

    evaluate_module = _load_script("evaluate")
    eval_result = evaluate_module.evaluate_and_save(cfg, str(best_ckpt), "val", device, research_root=tmp_path)
    assert eval_result["out_path"].exists()
    assert "macro_auc" in eval_result["metrics"]
    ablation_path = tmp_path / "outputs" / "metrics" / "ablation_table.csv"
    assert ablation_path.exists()
    ablation_df = pd.read_csv(ablation_path)
    assert ablation_df.iloc[-1]["Model"] == cfg["run_name"]

    predict_module = _load_script("predict")
    submission_path = predict_module.run_prediction(cfg, str(best_ckpt), device, research_root=tmp_path)

    assert submission_path.exists()
    submission_df = pd.read_csv(submission_path)
    expected_columns = list(pd.read_csv(tmp_path / "data" / "sample_submission.csv").columns)
    assert list(submission_df.columns) == expected_columns
    assert len(submission_df) == 1

    probs = submission_df[LABEL_COLS].to_numpy()
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0.0) & (probs <= 1.0))
