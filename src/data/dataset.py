"""Study-level PyTorch Dataset: one `__getitem__` call returns one complete
MRI study (all its series, padded to fixed shapes) plus labels and metadata.

Padding is FIXED to the configured `max_series`/`max_slices` (not dynamic
per-batch padding) so `torch.utils.data.default_collate` works unmodified
and shapes stay deterministic across batches -- see the project plan for the
justification. A "padded slot" (`series_mask`/`slice_mask` False) is always
literally zero-filled and provably excluded downstream by the model's
masking (see `tests/test_masking.py`), never merely assumed harmless.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.cache import get_or_build_series_tensor
from src.data.series import PLANE_TO_ID, load_series_metadata
from src.utils.logging import get_logger

logger = get_logger(__name__)


def split_studies(studies_df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Study-level train/val split (spec section 7: never split a single
    study's series across train and validation). Raises if `studies_df`
    contains a duplicate `StudyInstanceUID` -- that would silently leak a
    study across both splits."""
    uids = studies_df["StudyInstanceUID"]
    if uids.duplicated().any():
        dupes = uids[uids.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate StudyInstanceUID in studies_df, would leak across splits: {dupes[:5]}")

    rng = np.random.RandomState(seed)
    order = np.arange(len(studies_df))
    rng.shuffle(order)
    n_val = int(round(len(order) * val_fraction))
    val_idx, train_idx = order[:n_val], order[n_val:]
    train_df = studies_df.iloc[train_idx].reset_index(drop=True)
    val_df = studies_df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def _count_series_slices(dicom_root: Path, study_uid: str, series_uid: str) -> int:
    series_dir = dicom_root / study_uid / series_uid
    if not series_dir.exists():
        return 0
    return sum(1 for _ in series_dir.glob("*.dcm"))


def _select_series_rows(series_rows: pd.DataFrame, max_series: int, dicom_root: Path, study_uid: str) -> pd.DataFrame:
    """Deterministically choose at most `max_series` of a study's series:
    when there's no overflow, keep them all; otherwise keep one series per
    distinct anatomical plane (the one with the most slices in that plane),
    then fill any remaining slots by slice count descending. Ties are broken
    by `SeriesInstanceUID` so the choice is 100% reproducible across runs --
    never random."""
    if len(series_rows) <= max_series:
        return series_rows.reset_index(drop=True)

    rows = series_rows.copy()
    rows["_n_slices"] = rows["SeriesInstanceUID"].map(lambda uid: _count_series_slices(dicom_root, study_uid, uid))

    selected_index: list = []
    for _plane, group in rows.groupby("Anatomical_Plane", sort=True):
        best = group.sort_values(["_n_slices", "SeriesInstanceUID"], ascending=[False, True]).iloc[0]
        selected_index.append(best.name)
        if len(selected_index) >= max_series:
            break

    if len(selected_index) < max_series:
        remaining = rows.drop(index=selected_index).sort_values(["_n_slices", "SeriesInstanceUID"], ascending=[False, True])
        for idx in remaining.index:
            if len(selected_index) >= max_series:
                break
            selected_index.append(idx)

    result = rows.loc[selected_index].drop(columns=["_n_slices"])
    return result.sort_values("SeriesInstanceUID").reset_index(drop=True)


def _resolve_path(research_root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else research_root / p


def _subsample_studies(studies_df: pd.DataFrame, max_studies: int | None, seed: int) -> pd.DataFrame:
    """Optional, config-driven (`data.max_studies`) deterministic subsample
    of the full study list -- for a fast sanity run against real Kaggle data
    (hundreds of GB, hundreds of thousands of DICOM files) before committing
    to a full-scale pass. `None`/unset means "use every study", the default."""
    if not max_studies or len(studies_df) <= max_studies:
        return studies_df
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(studies_df), size=max_studies, replace=False)
    return studies_df.iloc[np.sort(idx)].reset_index(drop=True)


def build_train_val_datasets(cfg: dict, research_root: Path) -> tuple["KneeStudyDataset", "KneeStudyDataset"]:
    """Reads `paths.train_csv`/`train_series_csv`, applies the study-level
    split, and returns `(train_dataset, val_dataset)`. Shared by
    `scripts/train.py` and `scripts/evaluate.py` so the two scripts can never
    disagree about which studies ended up in which split."""
    paths = cfg["paths"]
    studies_df = pd.read_csv(_resolve_path(research_root, paths["train_csv"]))
    series_df = pd.read_csv(_resolve_path(research_root, paths["train_series_csv"]))
    studies_df = _subsample_studies(studies_df, cfg["data"].get("max_studies"), cfg["data"]["split_seed"])
    train_df, val_df = split_studies(studies_df, cfg["data"]["val_fraction"], cfg["data"]["split_seed"])

    use_report = cfg["model"]["report_weak_supervision"]["enabled"] or cfg["model"]["distillation"]["enabled"]
    common = dict(
        series_df=series_df,
        dicom_root=_resolve_path(research_root, paths["dicom_root_train"]),
        cache_root=_resolve_path(research_root, paths["cache_root"]),
        data_cfg=cfg["data"],
        label_cols=cfg["labels"]["columns"],
    )
    train_ds = KneeStudyDataset(train_df, include_report_text=use_report, split="train", **common)
    val_ds = KneeStudyDataset(val_df, include_report_text=False, split="val", **common)
    return train_ds, val_ds


def build_test_dataset(
    cfg: dict, research_root: Path, studies_csv: str | Path | None = None, series_csv: str | Path | None = None,
) -> "KneeStudyDataset":
    """Reads `paths.test_csv`/`test_series_csv` (or the given overrides, for
    `scripts/predict.py --input-csv/--series-csv`). The real Kaggle test set
    carries no ground-truth label columns at all -- any missing label column
    is injected as all-NaN so the same `KneeStudyDataset` contract (a
    `labels`/`label_mask` pair) still holds, with `label_mask` correctly all
    zero rather than the Dataset raising a `KeyError`."""
    paths = cfg["paths"]
    studies_df = pd.read_csv(studies_csv if studies_csv is not None else _resolve_path(research_root, paths["test_csv"]))
    series_df = pd.read_csv(series_csv if series_csv is not None else _resolve_path(research_root, paths["test_series_csv"]))
    studies_df = _subsample_studies(studies_df, cfg["data"].get("max_studies"), cfg["data"]["split_seed"])
    label_cols = cfg["labels"]["columns"]
    for col in label_cols:
        if col not in studies_df.columns:
            studies_df[col] = float("nan")
    return KneeStudyDataset(
        studies_df, series_df,
        _resolve_path(research_root, paths["dicom_root_test"]),
        _resolve_path(research_root, paths["cache_root"]),
        cfg["data"], label_cols, include_report_text=False, split="test",
    )


class KneeStudyDataset(Dataset):
    def __init__(
        self,
        studies_df: pd.DataFrame,
        series_df: pd.DataFrame,
        dicom_root: str | Path,
        cache_root: str | Path,
        data_cfg: dict,
        label_cols: list[str],
        include_report_text: bool = False,
        split: str = "train",
    ):
        self.studies_df = studies_df.reset_index(drop=True)
        self.series_df = series_df
        self.dicom_root = Path(dicom_root)
        self.cache_root = Path(cache_root)
        self.data_cfg = data_cfg
        self.label_cols = list(label_cols)
        self.include_report_text = include_report_text
        self.split = split

        self.max_series = int(data_cfg["max_series"])
        self.max_slices = int(data_cfg["max_slices"])
        self.image_size = int(data_cfg["image_size"])

        self._series_by_study = {uid: df for uid, df in series_df.groupby("StudyInstanceUID")}

    def __len__(self) -> int:
        return len(self.studies_df)

    def __getitem__(self, idx: int) -> dict:
        row = self.studies_df.iloc[idx]
        study_uid = str(row["StudyInstanceUID"])

        series_rows = self._series_by_study.get(study_uid)
        if series_rows is None or len(series_rows) == 0:
            logger.warning("Study %s has no rows in series_df -- returning an all-padded study.", study_uid)
            series_rows = self.series_df.iloc[0:0]
        else:
            series_rows = _select_series_rows(series_rows, self.max_series, self.dicom_root, study_uid)

        pixel_values = np.zeros((self.max_series, self.max_slices, 1, self.image_size, self.image_size), dtype=np.float32)
        slice_mask = np.zeros((self.max_series, self.max_slices), dtype=bool)
        series_mask = np.zeros((self.max_series,), dtype=bool)
        plane_id = np.full((self.max_series,), PLANE_TO_ID["Unknown"], dtype=np.int64)
        fluid_sensitive = np.zeros((self.max_series,), dtype=np.int64)
        fat_suppression = np.zeros((self.max_series,), dtype=np.int64)

        for s, series_row in enumerate(series_rows.to_dict("records")):
            if s >= self.max_series:
                break
            metadata = load_series_metadata(pd.Series(series_row))
            series_dir = self.dicom_root / study_uid / metadata.series_instance_uid
            tensor = get_or_build_series_tensor(
                study_uid=study_uid,
                series_uid=metadata.series_instance_uid,
                series_dir=series_dir,
                metadata=metadata,
                data_cfg=self.data_cfg,
                cache_root=self.cache_root,
            )
            pixels = tensor["pixels"]  # (n_selected, H, W), n_selected <= max_slices
            n = pixels.shape[0]
            if n > 0:
                pixel_values[s, :n, 0] = pixels
                slice_mask[s, :n] = True
            series_mask[s] = True
            plane_id[s] = tensor["plane_id"]
            fluid_sensitive[s] = tensor["fluid_sensitive"]
            fat_suppression[s] = tensor["fat_suppression"]

        labels_raw = row[self.label_cols].to_numpy(dtype=np.float32)
        label_mask = (~np.isnan(labels_raw)).astype(np.float32)
        labels = np.nan_to_num(labels_raw, nan=0.0).astype(np.float32)

        report_text = ""
        if self.include_report_text and self.split == "train":
            value = row.get("Report", "") if hasattr(row, "get") else ""
            if isinstance(value, str):
                report_text = value

        return {
            "pixel_values": torch.from_numpy(pixel_values),
            "slice_mask": torch.from_numpy(slice_mask),
            "series_mask": torch.from_numpy(series_mask),
            "plane_id": torch.from_numpy(plane_id),
            "fluid_sensitive": torch.from_numpy(fluid_sensitive),
            "fat_suppression": torch.from_numpy(fat_suppression),
            "labels": torch.from_numpy(labels),
            "label_mask": torch.from_numpy(label_mask),
            "report_text": report_text,
            "study_uid": study_uid,
        }
