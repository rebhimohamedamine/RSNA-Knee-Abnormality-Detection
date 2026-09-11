"""Shape/behavior tests for src/models/*, run against a tiny fake config so
the whole pipeline is checkable in seconds on CPU. Each module's test is
added right after that module is implemented, per the project's build order.
"""

from __future__ import annotations

import torch

import copy

from src.models.cross_series_attention import CrossSeriesTransformer
from src.models.knee_model import KneeModel
from src.models.metadata_encoder import SeriesMetadataEncoder
from src.models.pathology_attention import PATHOLOGY_GROUPS, PathologyAttentionModule
from src.models.report_teacher import ReportTeacher
from src.models.series_encoder import SeriesEncoder
from src.models.slice_attention import MaskedAttentionPool
from src.models.slice_encoder import SliceEncoder
from src.models.study_encoder import StudyEncoder
from src.models.task_heads import LABEL_ORDER, TaskHeads
from src.models.text_encoder import ReportTextEncoder
from src.utils.config import load_config

# Tiny fake dims, per the build plan: fast enough to run every test on CPU.
D = 16       # embedding_dim
B = 2        # batch
K = 4        # max_slices
S = 3        # max_series
IMG = 32     # image_size


# -- slice_encoder ------------------------------------------------------------

def test_slice_encoder_output_shape():
    enc = SliceEncoder(backbone="resnet18", embedding_dim=D, pretrained=False, in_channels=1)
    x = torch.randn(B * K, 1, IMG, IMG)
    out = enc(x)
    assert out.shape == (B * K, D)


def test_slice_encoder_three_channel_input_uses_backbone_conv_unchanged():
    enc = SliceEncoder(backbone="resnet18", embedding_dim=D, pretrained=False, in_channels=3)
    x = torch.randn(2, 3, IMG, IMG)
    assert enc(x).shape == (2, D)


# -- slice_attention -----------------------------------------------------------

def test_masked_attention_pool_shapes_enabled_and_disabled():
    feats = torch.randn(B, K, D)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

    for enabled in (True, False):
        pool = MaskedAttentionPool(D, enabled=enabled)
        pooled, weights = pool(feats, mask)
        assert pooled.shape == (B, D)
        assert weights.shape == (B, K)
        # weights over real slots sum to 1 (padded slots always weight 0)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(B), atol=1e-5)
        assert torch.allclose(weights[~mask], torch.zeros_like(weights[~mask]))


def test_masked_attention_pool_handles_all_padded_row_without_nan():
    feats = torch.randn(1, K, D)
    mask = torch.zeros(1, K, dtype=torch.bool)
    for enabled in (True, False):
        pool = MaskedAttentionPool(D, enabled=enabled)
        pooled, weights = pool(feats, mask)
        assert not torch.isnan(pooled).any()
        assert not torch.isnan(weights).any()


# -- metadata_encoder -----------------------------------------------------------

def test_metadata_encoder_shapes():
    enc = SeriesMetadataEncoder(embedding_dim=8, output_dim=D)
    plane_id = torch.randint(0, 4, (B, S))
    fluid = torch.randint(0, 2, (B, S))
    fat = torch.randint(0, 2, (B, S))
    out = enc(plane_id, fluid, fat)
    assert out.shape == (B, S, D)


# -- series_encoder -------------------------------------------------------------

def _make_series_encoder(with_metadata: bool):
    slice_enc = SliceEncoder(backbone="resnet18", embedding_dim=D, pretrained=False, in_channels=1)
    slice_attn = MaskedAttentionPool(D, enabled=True)
    meta_enc = SeriesMetadataEncoder(embedding_dim=8, output_dim=D) if with_metadata else None
    return SeriesEncoder(slice_enc, slice_attn, meta_enc, embedding_dim=D)


def _fake_batch():
    pixel_values = torch.randn(B, S, K, 1, IMG, IMG)
    slice_mask = torch.ones(B, S, K, dtype=torch.bool)
    slice_mask[:, :, -1] = False  # last slot of every series is padding
    plane_id = torch.randint(0, 4, (B, S))
    fluid = torch.randint(0, 2, (B, S))
    fat = torch.randint(0, 2, (B, S))
    series_mask = torch.ones(B, S, dtype=torch.bool)
    series_mask[:, -1] = False  # last series slot is padding
    return pixel_values, slice_mask, plane_id, fluid, fat, series_mask


def test_series_encoder_shapes_with_and_without_metadata():
    pixel_values, slice_mask, plane_id, fluid, fat, _series_mask = _fake_batch()
    for with_metadata in (True, False):
        enc = _make_series_encoder(with_metadata)
        out = enc(pixel_values, slice_mask, plane_id, fluid, fat)
        assert out.shape == (B, S, D)


def test_series_encoder_return_attention_shape():
    pixel_values, slice_mask, plane_id, fluid, fat, _series_mask = _fake_batch()
    enc = _make_series_encoder(with_metadata=False)
    out, attn = enc(pixel_values, slice_mask, plane_id, fluid, fat, return_attention=True)
    assert out.shape == (B, S, D)
    assert attn.shape == (B, S, K)


# -- cross_series_attention ------------------------------------------------------

def test_cross_series_transformer_shapes_enabled_and_disabled():
    series_emb = torch.randn(B, S, D)
    series_mask = torch.ones(B, S, dtype=torch.bool)
    series_mask[:, -1] = False
    for enabled in (True, False):
        xf = CrossSeriesTransformer(D, enabled=enabled, num_heads=2, num_layers=1)
        out = xf(series_emb, series_mask)
        assert out.shape == (B, S, D)
        assert not torch.isnan(out).any()


def test_cross_series_transformer_handles_all_padded_row_without_nan():
    series_emb = torch.randn(1, S, D)
    series_mask = torch.zeros(1, S, dtype=torch.bool)
    xf = CrossSeriesTransformer(D, enabled=True, num_heads=2, num_layers=1)
    out = xf(series_emb, series_mask)
    assert not torch.isnan(out).any()


# -- study_encoder ----------------------------------------------------------------

def test_study_encoder_masked_mean_shape():
    series_emb = torch.randn(B, S, D)
    series_mask = torch.ones(B, S, dtype=torch.bool)
    series_mask[:, -1] = False
    enc = StudyEncoder(D, pooling="masked_mean")
    out = enc(series_emb, series_mask)
    assert out.shape == (B, D)


def test_study_encoder_cls_token_not_implemented():
    import pytest

    with pytest.raises(NotImplementedError):
        StudyEncoder(D, pooling="cls_token")


# -- pathology_attention -----------------------------------------------------------

def test_pathology_attention_enabled_returns_all_groups_and_shapes():
    module = PathologyAttentionModule(D, enabled=True, bottleneck_ratio=2)
    study_emb = torch.randn(B, D)
    out = module(study_emb)
    assert set(out.keys()) == set(PATHOLOGY_GROUPS.keys())
    for group_emb in out.values():
        assert group_emb.shape == (B, D)


def test_pathology_attention_disabled_is_identity_for_every_group():
    module = PathologyAttentionModule(D, enabled=False)
    study_emb = torch.randn(B, D)
    out = module(study_emb)
    assert set(out.keys()) == set(PATHOLOGY_GROUPS.keys())
    for group_emb in out.values():
        assert torch.equal(group_emb, study_emb)


# -- task_heads ---------------------------------------------------------------------

def test_task_heads_output_shape_and_label_order():
    heads = TaskHeads(D)
    study_emb = torch.randn(B, D)
    group_embeddings = {group: study_emb for group in PATHOLOGY_GROUPS}
    logits = heads(group_embeddings)
    assert logits.shape == (B, len(LABEL_ORDER))
    assert len(LABEL_ORDER) == 12


# -- text_encoder / report_teacher ---------------------------------------------------

TINY_TEXT_MODEL = "hf-internal-testing/tiny-random-BertModel"


def test_report_text_encoder_output_shape():
    enc = ReportTextEncoder(model_name=TINY_TEXT_MODEL, embedding_dim=D, freeze_backbone=True, max_length=16)
    out = enc(["ACL tear present.", "No effusion.", ""])
    assert out.shape == (3, D)


def test_report_teacher_output_shape_and_label_count():
    text_encoder = ReportTextEncoder(model_name=TINY_TEXT_MODEL, embedding_dim=D, freeze_backbone=True, max_length=16)
    teacher = ReportTeacher(text_encoder, embedding_dim=D)
    logits = teacher(["Possible ACL tear.", "Fracture ruled out."])
    assert logits.shape == (2, len(LABEL_ORDER))


# -- knee_model (full pipeline) ------------------------------------------------------

def research_root_path():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent


def _full_cfg():
    """tiny_cpu_smoke.yaml as-is: every model.* flag on, tiny dims."""
    return load_config(research_root_path() / "configs" / "tiny_cpu_smoke.yaml")


def _minimal_cfg():
    """Same tiny dims, but every flag off (M1 shape) and no text branch, so
    it never needs to build/download a text encoder -- fast."""
    cfg = copy.deepcopy(_full_cfg())
    m = cfg["model"]
    m["slice_attention"]["enabled"] = False
    m["metadata"]["enabled"] = False
    m["cross_series_attention"]["enabled"] = False
    m["pathology_attention"]["enabled"] = False
    m["report_weak_supervision"]["enabled"] = False
    m["distillation"]["enabled"] = False
    return cfg


def _fake_dataset_batch(cfg, batch_size=2):
    d = cfg["data"]
    S, K, IMG_SIZE = d["max_series"], d["max_slices"], d["image_size"]
    pixel_values = torch.randn(batch_size, S, K, 1, IMG_SIZE, IMG_SIZE)
    slice_mask = torch.ones(batch_size, S, K, dtype=torch.bool)
    slice_mask[:, :, -1] = False
    series_mask = torch.ones(batch_size, S, dtype=torch.bool)
    series_mask[:, -1] = False
    plane_id = torch.randint(0, 4, (batch_size, S))
    fluid = torch.randint(0, 2, (batch_size, S))
    fat = torch.randint(0, 2, (batch_size, S))
    return {
        "pixel_values": pixel_values, "slice_mask": slice_mask, "series_mask": series_mask,
        "plane_id": plane_id, "fluid_sensitive": fluid, "fat_suppression": fat,
    }


def test_knee_model_m1_minimal_flags_forward_shape():
    cfg = _minimal_cfg()
    model = KneeModel(cfg)
    assert model.report_teacher is None
    batch = _fake_dataset_batch(cfg)
    out = model(batch)
    assert out["logits"].shape == (2, 12)
    assert out["teacher_logits"] is None
    assert not torch.isnan(out["logits"]).any()


def test_knee_model_m7_full_flags_forward_shape_with_report_texts():
    cfg = _full_cfg()
    model = KneeModel(cfg)
    assert model.report_teacher is not None
    batch = _fake_dataset_batch(cfg)
    out = model(batch, report_texts=["ACL tear.", "No effusion."])
    assert out["logits"].shape == (2, 12)
    assert out["teacher_logits"].shape == (2, 12)


def test_knee_model_report_texts_none_skips_teacher_output():
    cfg = _full_cfg()
    model = KneeModel(cfg)
    batch = _fake_dataset_batch(cfg)
    out = model(batch, report_texts=None)
    assert out["teacher_logits"] is None
    assert out["logits"].shape == (2, 12)


def test_report_teacher_never_called_when_report_texts_is_none():
    """The inference contract: even a checkpoint trained with M6/M7 flags on
    must never invoke the report branch when `report_texts=None`."""
    cfg = _full_cfg()
    model = KneeModel(cfg)
    batch = _fake_dataset_batch(cfg)

    def _raise(*args, **kwargs):
        raise AssertionError("report_teacher.forward must not be called when report_texts=None")

    model.report_teacher.forward = _raise
    out = model(batch, report_texts=None)  # must not raise
    assert out["logits"].shape == (2, 12)
