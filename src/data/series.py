"""Builds one series' preprocessed slice tensor from its DICOM files and
metadata row.

`build_series_tensor` chains the three previous layers (load -> order ->
preprocess -> select) and returns only the REAL, unpadded, selected slices --
padding out to a configured `max_slices` is the Dataset's responsibility, not
this module's (see `src/data/dataset.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dicom import load_dicom_series, order_slices, read_pixel_array
from src.data.preprocessing import PreprocessConfig, preprocess_series
from src.data.sampler import select_slice_indices

PLANE_TO_ID: dict[str, int] = {"Sagittal": 0, "Axial": 1, "Coronal": 2, "Unknown": 3}
NUM_PLANES = len(PLANE_TO_ID)


@dataclass
class SeriesMetadata:
    series_instance_uid: str
    study_instance_uid: str
    anatomical_plane: str
    fluid_sensitive: int
    fat_suppression: int

    @property
    def plane_id(self) -> int:
        return PLANE_TO_ID.get(self.anatomical_plane, PLANE_TO_ID["Unknown"])


def load_series_metadata(row: pd.Series) -> SeriesMetadata:
    """Build a `SeriesMetadata` from one row of `train_series.csv` /
    `test_series.csv`."""
    return SeriesMetadata(
        series_instance_uid=str(row["SeriesInstanceUID"]),
        study_instance_uid=str(row["StudyInstanceUID"]),
        anatomical_plane=str(row["Anatomical_Plane"]),
        fluid_sensitive=int(row["Fluid_Sensitive"]),
        fat_suppression=int(row["Fat_Suppression"]),
    )


def build_series_tensor(series_dir: str | Path, metadata: SeriesMetadata, data_cfg: dict) -> dict:
    """Load, spatially order, preprocess, and select representative slices
    for one series.

    `data_cfg` is the `data:` section of the run config (`image_size`,
    `clip_percentile`, `normalize`, `max_slices`, `slice_sampling_strategy`).

    Returns a dict with `pixels` shaped `(n_selected, H, W)` float32 where
    `n_selected = min(max_slices, n_real_slices_found)` -- never more than
    what was actually available, and never padded here.
    """
    preprocess_cfg = PreprocessConfig.from_dict(data_cfg)

    records = load_dicom_series(series_dir)
    ordered = order_slices(records)
    raw_slices = [read_pixel_array(r) for r in ordered]
    processed = preprocess_series(raw_slices, preprocess_cfg)  # (n_real, H, W)

    n_real = processed.shape[0]
    indices = select_slice_indices(n_real, data_cfg["max_slices"], data_cfg["slice_sampling_strategy"])
    selected = processed[indices] if indices else np.zeros((0, preprocess_cfg.image_size, preprocess_cfg.image_size), dtype=np.float32)

    return {
        "pixels": selected.astype(np.float32),
        "n_real_slices": len(indices),
        "plane_id": metadata.plane_id,
        "fluid_sensitive": int(metadata.fluid_sensitive),
        "fat_suppression": int(metadata.fat_suppression),
        "series_uid": metadata.series_instance_uid,
        "study_uid": metadata.study_instance_uid,
    }
