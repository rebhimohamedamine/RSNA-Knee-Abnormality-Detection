#!/usr/bin/env python
"""Warm the series-tensor cache and precompute report weak-labels for one
data split. Thin orchestration only -- all logic lives in src/.

    python scripts/preprocess.py --config configs/baseline.yaml --split train
    python scripts/preprocess.py --config configs/baseline.yaml --split both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parent.parent
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))  # so `python scripts/preprocess.py` finds src/ regardless of cwd

import pandas as pd
from tqdm import tqdm

from src.data.cache import get_or_build_series_tensor
from src.data.series import load_series_metadata
from src.reports.label_mapping import TARGETS
from src.reports.weak_labels import extract_weak_labels
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger("scripts.preprocess")


def resolve_path(research_root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else research_root / p


def _precompute_weak_labels(studies_csv: Path, cache_root: Path) -> None:
    studies_df = pd.read_csv(studies_csv, usecols=["StudyInstanceUID", "Report"])
    records = []
    for row in tqdm(studies_df.to_dict("records"), desc="weak-labels"):
        labels, confidence = extract_weak_labels(row.get("Report"))
        record = {"StudyInstanceUID": row["StudyInstanceUID"]}
        record.update({f"label__{t}": float(labels[i]) for i, t in enumerate(TARGETS)})
        record.update({f"confidence__{t}": float(confidence[i]) for i, t in enumerate(TARGETS)})
        records.append(record)

    out_path = cache_root / "weak_labels.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(out_path, index=False)
    logger.info("Wrote precomputed weak labels for %d studies to %s", len(records), out_path)


def preprocess_split(cfg: dict, split: str, research_root: Path = RESEARCH_ROOT) -> None:
    paths = cfg["paths"]
    if split == "train":
        series_csv, dicom_root, studies_csv = paths["train_series_csv"], paths["dicom_root_train"], paths["train_csv"]
    elif split == "test":
        series_csv, dicom_root, studies_csv = paths["test_series_csv"], paths["dicom_root_test"], paths["test_csv"]
    else:
        raise ValueError(f"Unknown split: {split!r} (expected train | test)")

    series_csv, dicom_root, studies_csv = resolve_path(research_root, series_csv), resolve_path(research_root, dicom_root), resolve_path(research_root, studies_csv)
    cache_root = resolve_path(research_root, paths["cache_root"])
    data_cfg = cfg["data"]

    series_df = pd.read_csv(series_csv)
    logger.info("Preprocessing %d series (%s split) into cache at %s", len(series_df), split, cache_root)

    n_missing_dirs = 0
    for row in tqdm(series_df.to_dict("records"), desc=f"preprocess:{split}"):
        metadata = load_series_metadata(pd.Series(row))
        series_dir = dicom_root / metadata.study_instance_uid / metadata.series_instance_uid
        if not series_dir.exists():
            n_missing_dirs += 1
            continue
        get_or_build_series_tensor(
            study_uid=metadata.study_instance_uid, series_uid=metadata.series_instance_uid,
            series_dir=series_dir, metadata=metadata, data_cfg=data_cfg, cache_root=cache_root,
        )

    if n_missing_dirs:
        logger.warning(
            "%d/%d series had no DICOM directory on disk under %s -- skipped "
            "(expected if only the CSV stub is available locally; run on Kaggle for the real data).",
            n_missing_dirs, len(series_df), dicom_root,
        )

    if split == "train":
        header = pd.read_csv(studies_csv, nrows=0)
        if "Report" in header.columns:
            _precompute_weak_labels(studies_csv, cache_root)
        else:
            logger.info("No Report column in %s -- skipping weak-label precomputation.", studies_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a configs/*.yaml file.")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Currently a no-op: the cache is content-addressed by the config's data.* settings "
             "(src.data.cache.cache_key), so a real config change already invalidates it automatically.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        preprocess_split(cfg, split)


if __name__ == "__main__":
    main()
