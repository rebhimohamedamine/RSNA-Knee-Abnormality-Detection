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

import shutil
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

from src.data.cache import get_or_build_series_tensor
from src.data.dataset import _subsample_studies
from src.data.series import SeriesMetadata, load_series_metadata
from src.reports.label_mapping import TARGETS
from src.reports.weak_labels import extract_weak_labels
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger("scripts.preprocess")

MIN_FREE_BYTES = 2 * 1024**3       # refuse to keep writing cache below this much free disk
MAX_CONSECUTIVE_ERRORS = 20        # abort early rather than grinding through thousands of doomed writes


def resolve_path(research_root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else research_root / p


def check_disk_space(path: Path, min_free_bytes: int = MIN_FREE_BYTES) -> None:
    """Raises a clear, actionable error if `path`'s filesystem is nearly
    full, instead of letting `torch.save` fail deep inside a worker process
    with an opaque `RuntimeError: unexpected pos ... vs 0` -- which is what
    a genuinely full disk looks like, and which otherwise only surfaces
    after minutes-to-hours of wasted work."""
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < min_free_bytes:
        raise RuntimeError(
            f"Only {free / 1024**3:.1f} GiB free at {path} (need at least "
            f"{min_free_bytes / 1024**3:.0f} GiB headroom to keep writing cache safely). "
            f"Reduce data.max_studies / data.image_size / data.max_slices, or clear the "
            f"existing cache (`rm -rf {path}`), then retry. Check real free space with "
            f"`!df -h {path}` before picking new numbers -- don't just guess."
        )


def _process_one_series(task: tuple[str, str, Path, SeriesMetadata, dict, Path]) -> tuple[bool, str | None]:
    """Builds/caches one series' tensor. Runs directly (num_workers<=1) or in
    a worker process (num_workers>1, via ProcessPoolExecutor below) -- must
    therefore be a plain module-level function, not a closure, so it can be
    pickled across the process boundary.

    Returns `(processed, error)`: `(False, None)` if the DICOM directory
    doesn't exist (expected/benign); `(False, "...")` if it existed but
    something raised while processing it (e.g. disk full, a corrupt DICOM
    file) -- callers log these and keep going rather than letting one bad
    series abort an hours-long run; `(True, None)` on success."""
    study_uid, series_uid, series_dir, metadata, data_cfg, cache_root = task
    if not series_dir.exists():
        return False, None
    try:
        get_or_build_series_tensor(
            study_uid=study_uid, series_uid=series_uid, series_dir=series_dir,
            metadata=metadata, data_cfg=data_cfg, cache_root=cache_root,
        )
        return True, None
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: log and move on, never abort the whole run on one series
        return False, f"{study_uid}/{series_uid}: {exc!r}"


def _precompute_weak_labels(studies_df: pd.DataFrame, cache_root: Path) -> None:
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

    # Apply the SAME data.max_studies subsample that build_train_val_datasets/
    # build_test_dataset (used by train.py/evaluate.py/predict.py) apply --
    # otherwise this warms the cache for every series in the whole CSV
    # (24,371 for the real training set) regardless of the config's intended
    # subset, which is both pointless work and why a "smoke" config could
    # still take hours.
    studies_df = pd.read_csv(studies_csv)
    studies_df = _subsample_studies(studies_df, data_cfg.get("max_studies"), data_cfg["split_seed"])
    series_df = pd.read_csv(series_csv)
    if data_cfg.get("max_studies"):
        series_df = series_df[series_df["StudyInstanceUID"].isin(set(studies_df["StudyInstanceUID"]))].reset_index(drop=True)

    num_workers = int(data_cfg.get("num_workers", 0) or 0)
    check_disk_space(cache_root)
    logger.info(
        "Preprocessing %d series (%s split, max_studies=%s) into cache at %s using %d worker process(es); "
        "%.1f GiB free at start.",
        len(series_df), split, data_cfg.get("max_studies"), cache_root, num_workers,
        shutil.disk_usage(cache_root).free / 1024**3,
    )

    tasks = []
    for row in series_df.to_dict("records"):
        metadata = load_series_metadata(pd.Series(row))
        series_dir = dicom_root / metadata.study_instance_uid / metadata.series_instance_uid
        tasks.append((metadata.study_instance_uid, metadata.series_instance_uid, series_dir, metadata, data_cfg, cache_root))

    n_missing_dirs = 0
    n_errors = 0
    consecutive_errors = 0
    error_examples: list[str] = []
    DISK_CHECK_EVERY = 200

    def _handle_result(i: int, processed: bool, error: str | None) -> None:
        nonlocal n_missing_dirs, n_errors, consecutive_errors
        if error is not None:
            n_errors += 1
            consecutive_errors += 1
            if len(error_examples) < 5:
                error_examples.append(error)
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                raise RuntimeError(
                    f"{consecutive_errors} series in a row failed to cache (most recent: {error}). "
                    f"Stopping early rather than repeating a likely-systemic failure (disk full is the "
                    f"most common cause -- check `!df -h {cache_root}`) across the remaining series."
                )
        else:
            consecutive_errors = 0
            if not processed:
                n_missing_dirs += 1
        if i % DISK_CHECK_EVERY == 0:
            check_disk_space(cache_root)

    if num_workers > 1:
        # DICOM decode + resize is CPU/IO-bound (no GPU involved at all), and
        # each series is independent, so this parallelizes cleanly across
        # processes -- a real wall-clock win on Kaggle's multi-core CPU.
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i, (processed, error) in enumerate(tqdm(executor.map(_process_one_series, tasks, chunksize=4), total=len(tasks), desc=f"preprocess:{split}")):
                _handle_result(i, processed, error)
    else:
        for i, task in enumerate(tqdm(tasks, desc=f"preprocess:{split}")):
            processed, error = _process_one_series(task)
            _handle_result(i, processed, error)

    if n_missing_dirs:
        logger.warning(
            "%d/%d series had no DICOM directory on disk under %s -- skipped "
            "(expected if only the CSV stub is available locally; run on Kaggle for the real data).",
            n_missing_dirs, len(series_df), dicom_root,
        )
    if n_errors:
        logger.warning("%d/%d series raised an error while caching (examples: %s).", n_errors, len(series_df), error_examples)

    if split == "train":
        if "Report" in studies_df.columns:
            _precompute_weak_labels(studies_df[["StudyInstanceUID", "Report"]], cache_root)
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
