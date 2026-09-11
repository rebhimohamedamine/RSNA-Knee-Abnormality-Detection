"""Generates small, valid, synthetic DICOM series on disk for tests -- no
real Kaggle data is ever required to exercise `src/data/dicom.py`.

Files are written under filenames unrelated to spatial order (a UID-derived
name uncorrelated with slice index), so a test asserting correct ordering is
actually exercising `order_slices()`'s use of spatial/instance metadata,
not accidentally passing because the filesystem happened to return files in
the "right" order.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

# Canonical (Rows, Columns) direction-cosine pairs -- not clinically exact,
# just distinct and consistent per plane so compute_slice_normal gives a
# clean, predictable stacking axis for each.
PLANE_ORIENTATIONS: dict[str, list[float]] = {
    "Sagittal": [0.0, 1.0, 0.0, 0.0, 0.0, -1.0],   # normal ~ (-1, 0, 0)
    "Axial": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],        # normal ~ (0, 0, 1)
    "Coronal": [1.0, 0.0, 0.0, 0.0, 0.0, -1.0],     # normal ~ (0, 1, 0)
}


def _new_uid() -> str:
    # Not a spec-conformant DICOM UID, just unique and dot-separated enough
    # for pydicom to accept and for filenames to be unrelated to slice index.
    return "1.2.826.0.1.3680043.8.498." + str(uuid.uuid4().int)[:24]


def _make_slice_dataset(
    *,
    sop_instance_uid: str,
    rows: int,
    cols: int,
    pixel_array: np.ndarray,
    image_position_patient: list[float] | None,
    image_orientation_patient: list[float] | None,
    instance_number: int | None,
    slice_location: float | None,
    rescale_slope: float,
    rescale_intercept: float,
) -> FileDataset:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)

    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = sop_instance_uid
    ds.Modality = "MR"
    ds.SeriesInstanceUID = _new_uid()
    ds.StudyInstanceUID = _new_uid()

    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixel_array.astype(np.uint16).tobytes()

    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 3.0
    ds.RescaleSlope = rescale_slope
    ds.RescaleIntercept = rescale_intercept

    if image_position_patient is not None:
        ds.ImagePositionPatient = image_position_patient
    if image_orientation_patient is not None:
        ds.ImageOrientationPatient = image_orientation_patient
    if instance_number is not None:
        ds.InstanceNumber = instance_number
    if slice_location is not None:
        ds.SliceLocation = slice_location

    return ds


def make_synthetic_series(
    out_dir: str | Path,
    n_slices: int = 20,
    rows: int = 64,
    cols: int = 64,
    plane: str = "Sagittal",
    seed: int = 0,
    step_mm: float = 3.0,
    rescale_slope: float = 1.0,
    rescale_intercept: float = 0.0,
) -> Path:
    """Write `n_slices` synthetic DICOM files (full IPP/IOP/InstanceNumber
    metadata) into `out_dir`, in filename order unrelated to true slice
    index `i`. True spatial order is `i = 0 .. n_slices-1` along the plane's
    normal, `ImagePositionPatient = normal * i * step_mm`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    row = np.array(PLANE_ORIENTATIONS[plane][0:3])
    col = np.array(PLANE_ORIENTATIONS[plane][3:6])
    normal = np.cross(row, col)
    normal = normal / np.linalg.norm(normal)

    for i in range(n_slices):
        pixel_array = rng.randint(0, 4096, size=(rows, cols), dtype=np.uint16)
        position = (normal * i * step_mm).tolist()
        ds = _make_slice_dataset(
            sop_instance_uid=_new_uid(),
            rows=rows,
            cols=cols,
            pixel_array=pixel_array,
            image_position_patient=position,
            image_orientation_patient=PLANE_ORIENTATIONS[plane],
            instance_number=i + 1,
            slice_location=float(i * step_mm),
            rescale_slope=rescale_slope,
            rescale_intercept=rescale_intercept,
        )
        # Filename is the SOPInstanceUID (as in the real dataset layout),
        # which is unrelated to `i` -- filename order != spatial order.
        ds.save_as(out_dir / f"{ds.SOPInstanceUID}.dcm", enforce_file_format=True)

    return out_dir


def make_synthetic_series_missing_position(
    out_dir: str | Path,
    n_slices: int = 20,
    rows: int = 64,
    cols: int = 64,
    seed: int = 0,
) -> Path:
    """Like `make_synthetic_series`, but omits ImagePositionPatient and
    ImageOrientationPatient entirely -- forces `order_slices` down to its
    InstanceNumber fallback. InstanceNumber is set to the true spatial order
    `i + 1` so the fallback's correctness is checkable."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    for i in range(n_slices):
        pixel_array = rng.randint(0, 4096, size=(rows, cols), dtype=np.uint16)
        ds = _make_slice_dataset(
            sop_instance_uid=_new_uid(),
            rows=rows,
            cols=cols,
            pixel_array=pixel_array,
            image_position_patient=None,
            image_orientation_patient=None,
            instance_number=i + 1,
            slice_location=None,
            rescale_slope=1.0,
            rescale_intercept=0.0,
        )
        ds.save_as(out_dir / f"{ds.SOPInstanceUID}.dcm", enforce_file_format=True)

    return out_dir
