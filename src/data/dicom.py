"""DICOM loading and slice ordering.

Slice order must never be inferred from filename when spatial metadata is
available -- `order_slices` implements the fallback chain described in the
project spec: ImagePositionPatient projected onto the series' slice normal,
then InstanceNumber, then SliceLocation, then (last resort, with a logged
warning) natural filename order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SliceRecord:
    path: Path
    sop_instance_uid: str
    pixel_array: np.ndarray                        # raw, as stored (not rescaled)
    image_position_patient: np.ndarray | None       # (3,) float64, or None if absent
    image_orientation_patient: np.ndarray | None    # (6,) float64, or None if absent
    pixel_spacing: tuple[float, float] | None
    slice_thickness: float | None
    slice_location: float | None
    rescale_slope: float
    rescale_intercept: float
    instance_number: int | None


def _to_float_array(value, expected_len: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.array([float(v) for v in value], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape != (expected_len,):
        return None
    return arr


def load_dicom_series(series_dir: str | Path) -> list[SliceRecord]:
    """Read every `*.dcm` file directly under `series_dir` into a
    `SliceRecord`, in whatever order the filesystem returns them (callers
    must call `order_slices` before relying on ordering -- this function
    intentionally does not sort)."""
    series_dir = Path(series_dir)
    records: list[SliceRecord] = []
    for path in series_dir.glob("*.dcm"):
        ds = pydicom.dcmread(path)

        ipp = _to_float_array(getattr(ds, "ImagePositionPatient", None), 3)
        iop = _to_float_array(getattr(ds, "ImageOrientationPatient", None), 6)

        pixel_spacing_raw = getattr(ds, "PixelSpacing", None)
        pixel_spacing = None
        if pixel_spacing_raw is not None and len(pixel_spacing_raw) == 2:
            pixel_spacing = (float(pixel_spacing_raw[0]), float(pixel_spacing_raw[1]))

        slice_location = getattr(ds, "SliceLocation", None)
        slice_location = float(slice_location) if slice_location is not None else None

        instance_number = getattr(ds, "InstanceNumber", None)
        instance_number = int(instance_number) if instance_number is not None else None

        records.append(
            SliceRecord(
                path=path,
                sop_instance_uid=str(getattr(ds, "SOPInstanceUID", path.stem)),
                pixel_array=ds.pixel_array,
                image_position_patient=ipp,
                image_orientation_patient=iop,
                pixel_spacing=pixel_spacing,
                slice_thickness=float(ds.SliceThickness) if getattr(ds, "SliceThickness", None) is not None else None,
                slice_location=slice_location,
                rescale_slope=float(getattr(ds, "RescaleSlope", 1.0)),
                rescale_intercept=float(getattr(ds, "RescaleIntercept", 0.0)),
                instance_number=instance_number,
            )
        )
    return records


def compute_slice_normal(iop: np.ndarray) -> np.ndarray:
    """Unit normal to the imaging plane, from the row/column direction
    cosines in ImageOrientationPatient (tag 0020,0037)."""
    row, col = iop[0:3], iop[3:6]
    normal = np.cross(row, col)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        raise ValueError("Degenerate ImageOrientationPatient: row/col vectors are parallel")
    return normal / norm


def _natural_sort_key(name: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def order_slices(records: list[SliceRecord]) -> list[SliceRecord]:
    """Return a new, correctly z-ordered list (does not mutate `records`),
    using the first strategy below whose required metadata is present on
    every record:

    1. ImagePositionPatient projected onto the normal derived from the
       first record's ImageOrientationPatient (orientation is assumed
       consistent within one series), sorted ascending.
    2. InstanceNumber, sorted ascending.
    3. SliceLocation, sorted ascending.
    4. Natural (numeric-aware) filename sort -- last resort, logged as a
       warning since it may not reflect true spatial order.
    """
    if len(records) <= 1:
        return list(records)

    has_ipp_iop = all(r.image_position_patient is not None and r.image_orientation_patient is not None for r in records)
    if has_ipp_iop:
        normal = compute_slice_normal(records[0].image_orientation_patient)
        return sorted(records, key=lambda r: float(np.dot(r.image_position_patient, normal)))

    if all(r.instance_number is not None for r in records):
        return sorted(records, key=lambda r: r.instance_number)

    if all(r.slice_location is not None for r in records):
        return sorted(records, key=lambda r: r.slice_location)

    logger.warning(
        "No ImagePositionPatient/ImageOrientationPatient, InstanceNumber, or "
        "SliceLocation available for all %d slices in %s -- falling back to "
        "filename order, which may not reflect true spatial order.",
        len(records),
        records[0].path.parent,
    )
    return sorted(records, key=lambda r: _natural_sort_key(r.path.name))


def read_pixel_array(record: SliceRecord) -> np.ndarray:
    """Apply RescaleSlope/RescaleIntercept to a record's raw pixel array,
    returning float32."""
    return record.pixel_array.astype(np.float32) * record.rescale_slope + record.rescale_intercept
