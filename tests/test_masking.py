"""Masking correctness tests (spec: "padded slices -> ignored, padded series
-> ignored", checked directly rather than assumed).

This file has two parts: dataset-level (padded pixel slots are provably
zero-filled and correctly unmarked) and model-level (padded content never
changes a masked module's output -- proven by corrupting exactly the padded
positions with large noise and checking the output doesn't move).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import KneeStudyDataset
from src.models.cross_series_attention import CrossSeriesTransformer
from src.models.knee_model import KneeModel
from src.models.slice_attention import MaskedAttentionPool
from src.models.study_encoder import StudyEncoder
from src.utils.config import load_config
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


# -- model-level: padded content must never move a masked module's output ------------

D, B, K, S = 16, 2, 5, 4


def _mask_with_padding(shape, n_real):
    """A (B, N) bool mask with the first `n_real` positions True (real) and
    the rest False (padded), for every row in `shape`."""
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[:, :n_real] = True
    return mask


def test_masked_attention_pool_ignores_padded_slice_content():
    n_real = 3
    slice_mask = _mask_with_padding((B, K), n_real)
    clean = torch.randn(B, K, D)
    corrupted = clean.clone()
    corrupted[:, n_real:] = torch.randn(B, K - n_real, D) * 1000.0  # huge noise, padded slots only

    for enabled in (True, False):
        pool = MaskedAttentionPool(D, enabled=enabled)
        pooled_clean, _ = pool(clean, slice_mask)
        pooled_corrupted, _ = pool(corrupted, slice_mask)
        assert torch.allclose(pooled_clean, pooled_corrupted, atol=1e-5)


def test_cross_series_transformer_ignores_padded_series_content():
    n_real = 2
    series_mask = _mask_with_padding((B, S), n_real)
    clean = torch.randn(B, S, D)
    corrupted = clean.clone()
    corrupted[:, n_real:] = torch.randn(B, S - n_real, D) * 1000.0

    xf = CrossSeriesTransformer(D, enabled=True, num_heads=2, num_layers=1, dropout=0.0)
    xf.eval()
    with torch.no_grad():
        out_clean = xf(clean, series_mask)
        out_corrupted = xf(corrupted, series_mask)
    # Only the REAL positions' output is used downstream (study_encoder pools
    # by series_mask) -- those must be unaffected by padded-position content.
    assert torch.allclose(out_clean[:, :n_real], out_corrupted[:, :n_real], atol=1e-4)


def test_study_encoder_ignores_padded_series_content():
    n_real = 2
    series_mask = _mask_with_padding((B, S), n_real)
    clean = torch.randn(B, S, D)
    corrupted = clean.clone()
    corrupted[:, n_real:] = torch.randn(B, S - n_real, D) * 1000.0

    enc = StudyEncoder(D, pooling="masked_mean")
    assert torch.allclose(enc(clean, series_mask), enc(corrupted, series_mask), atol=1e-5)


def test_knee_model_full_pipeline_ignores_padded_content():
    """End-to-end: corrupting only the zero-padded slice/series slots of the
    input (never the real ones) must not change the model's output logits,
    for the fully-flagged (M7) architecture."""
    research_root = Path(__file__).resolve().parent.parent
    cfg = copy.deepcopy(load_config(research_root / "configs" / "tiny_cpu_smoke.yaml"))
    cfg["model"]["report_weak_supervision"]["enabled"] = False
    cfg["model"]["distillation"]["enabled"] = False  # skip building the text branch: irrelevant here, faster

    model = KneeModel(cfg)
    model.eval()

    d = cfg["data"]
    Sd, Kd, IMG = d["max_series"], d["max_slices"], d["image_size"]
    n_real_series, n_real_slices = 2, 2

    slice_mask = torch.zeros(1, Sd, Kd, dtype=torch.bool)
    slice_mask[:, :n_real_series, :n_real_slices] = True
    series_mask = torch.zeros(1, Sd, dtype=torch.bool)
    series_mask[:, :n_real_series] = True
    plane_id = torch.randint(0, 4, (1, Sd))
    fluid = torch.randint(0, 2, (1, Sd))
    fat = torch.randint(0, 2, (1, Sd))

    pixel_values = torch.zeros(1, Sd, Kd, 1, IMG, IMG)
    pixel_values[:, :n_real_series, :n_real_slices] = torch.randn(1, n_real_series, n_real_slices, 1, IMG, IMG)

    corrupted = pixel_values.clone()
    # Corrupt only padded slots: padded slices within real series, and every
    # slot of padded series.
    corrupted[:, :n_real_series, n_real_slices:] = torch.randn_like(corrupted[:, :n_real_series, n_real_slices:]) * 1000.0
    corrupted[:, n_real_series:] = torch.randn_like(corrupted[:, n_real_series:]) * 1000.0

    batch_clean = {
        "pixel_values": pixel_values, "slice_mask": slice_mask, "series_mask": series_mask,
        "plane_id": plane_id, "fluid_sensitive": fluid, "fat_suppression": fat,
    }
    batch_corrupted = {**batch_clean, "pixel_values": corrupted}

    with torch.no_grad():
        out_clean = model(batch_clean)
        out_corrupted = model(batch_corrupted)

    assert torch.allclose(out_clean["logits"], out_corrupted["logits"], atol=1e-4)
