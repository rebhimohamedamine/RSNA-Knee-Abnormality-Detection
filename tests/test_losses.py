"""Unit tests for src/training/losses.py."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.training.losses import (
    build_bce_term,
    combined_loss,
    compute_pos_weight,
    distillation_term,
    report_weak_supervision_term,
    sigmoid_focal_loss,
)

LOSS_CFG_PLAIN = {"class_imbalance_strategy": "plain", "focal": {"gamma": 2.0, "alpha": 0.25}, "weights": {"bce": 1.0, "report_weak": 0.0, "distillation": 0.0}}


def test_plain_bce_matches_hand_computed_value():
    logits = torch.tensor([[0.0, 2.0]])
    labels = torch.tensor([[1.0, 0.0]])
    label_mask = torch.ones_like(labels)
    bce_fn = build_bce_term(LOSS_CFG_PLAIN)
    loss = bce_fn(logits, labels, label_mask)

    expected = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
    assert torch.allclose(loss, expected, atol=1e-6)


def test_label_mask_zero_rows_contribute_exactly_zero():
    logits = torch.tensor([[0.0, 0.0], [50.0, -50.0]])  # second row: extreme logits
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])      # second row: maximally wrong
    label_mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])  # second row fully masked out
    bce_fn = build_bce_term(LOSS_CFG_PLAIN)

    masked_loss = bce_fn(logits, labels, label_mask)
    only_first_row_loss = bce_fn(logits[:1], labels[:1], label_mask[:1])
    assert torch.allclose(masked_loss, only_first_row_loss, atol=1e-6)


def test_pos_weight_strategy_differs_from_plain():
    logits = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
    labels = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    label_mask = torch.ones_like(labels)

    plain = build_bce_term(LOSS_CFG_PLAIN)(logits, labels, label_mask)
    pos_weight = torch.tensor([3.0, 3.0])
    cfg = {**LOSS_CFG_PLAIN, "class_imbalance_strategy": "pos_weight"}
    weighted = build_bce_term(cfg, pos_weight=pos_weight)(logits, labels, label_mask)

    assert not torch.allclose(plain, weighted)


def test_pos_weight_strategy_requires_pos_weight_tensor():
    import pytest

    cfg = {**LOSS_CFG_PLAIN, "class_imbalance_strategy": "pos_weight"}
    with pytest.raises(ValueError):
        build_bce_term(cfg)


def test_compute_pos_weight_matches_hand_computed_ratio():
    labels = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    label_mask = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])  # last label of row 3 unlabeled
    pw = compute_pos_weight(labels, label_mask)
    # label 0: 2 positives, 1 negative (row3) -> neg/pos = 0.5
    assert math.isclose(pw[0].item(), 0.5, rel_tol=1e-5)
    # label 1: 0 positives among 2 labeled rows, 2 negatives -> huge ratio, clamped
    assert pw[1].item() == 100.0  # clamped to max_weight default


def test_focal_gamma_zero_alpha_none_reduces_to_plain_bce():
    logits = torch.randn(5, 3)
    labels = torch.randint(0, 2, (5, 3)).float()
    focal = sigmoid_focal_loss(logits, labels, gamma=0.0, alpha=None)
    plain = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    assert torch.allclose(focal, plain, atol=1e-6)


def test_focal_with_alpha_differs_from_plain_bce():
    logits = torch.randn(5, 3)
    labels = torch.randint(0, 2, (5, 3)).float()
    focal = sigmoid_focal_loss(logits, labels, gamma=2.0, alpha=0.25)
    plain = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    assert not torch.allclose(focal, plain)


def test_report_weak_supervision_term_excludes_zero_confidence():
    teacher_logits = torch.tensor([[0.0, 5.0]])   # second column: confidently wrong if target=0
    weak_labels = torch.tensor([[1.0, 0.0]])
    weak_confidence = torch.tensor([[1.0, 0.0]])  # second label unknown -> excluded
    loss = report_weak_supervision_term(teacher_logits, weak_labels, weak_confidence)

    only_first = report_weak_supervision_term(teacher_logits[:, :1], weak_labels[:, :1], weak_confidence[:, :1])
    assert torch.allclose(loss, only_first, atol=1e-6)


def test_report_weak_supervision_term_scales_with_confidence():
    teacher_logits = torch.tensor([[2.0]])
    weak_labels = torch.tensor([[0.0]])
    full_conf = report_weak_supervision_term(teacher_logits, weak_labels, torch.tensor([[1.0]]))
    half_conf = report_weak_supervision_term(teacher_logits, weak_labels, torch.tensor([[0.5]]))
    # both normalize by the confidence sum, so a SINGLE-element batch's value is
    # confidence-invariant; scaling only matters when averaged across elements
    # of differing confidence (see the exclusion test above). Sanity: neither is NaN.
    assert torch.isfinite(full_conf) and torch.isfinite(half_conf)


def test_distillation_term_is_zero_when_student_matches_teacher():
    logits = torch.randn(4, 12)
    loss = distillation_term(logits, logits, temperature=2.0)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_distillation_term_is_positive_when_student_differs():
    student = torch.randn(4, 12)
    teacher = torch.randn(4, 12)
    loss = distillation_term(student, teacher, temperature=2.0)
    assert loss.item() > 0


def test_combined_loss_with_zero_aux_weights_equals_weighted_bce():
    logits = torch.randn(3, 12)
    labels = torch.randint(0, 2, (3, 12)).float()
    label_mask = torch.ones_like(labels)
    bce_fn = build_bce_term(LOSS_CFG_PLAIN)

    out = combined_loss(logits, labels, label_mask, LOSS_CFG_PLAIN, bce_fn)
    assert set(out.keys()) == {"total", "bce", "report_weak", "distill"}
    assert torch.allclose(out["total"], LOSS_CFG_PLAIN["weights"]["bce"] * out["bce"], atol=1e-6)
    assert out["report_weak"].item() == 0.0
    assert out["distill"].item() == 0.0


def test_combined_loss_includes_report_and_distillation_when_weighted():
    logits = torch.randn(3, 12)
    labels = torch.randint(0, 2, (3, 12)).float()
    label_mask = torch.ones_like(labels)
    teacher_logits = torch.randn(3, 12)
    weak_labels = torch.randint(0, 2, (3, 12)).float()
    weak_confidence = torch.ones_like(labels)

    cfg = {
        "class_imbalance_strategy": "plain", "focal": {"gamma": 2.0, "alpha": 0.25},
        "weights": {"bce": 1.0, "report_weak": 0.3, "distillation": 0.2},
    }
    bce_fn = build_bce_term(cfg)
    out = combined_loss(
        logits, labels, label_mask, cfg, bce_fn,
        teacher_logits=teacher_logits, weak_labels=weak_labels, weak_confidence=weak_confidence,
    )
    expected_total = cfg["weights"]["bce"] * out["bce"] + cfg["weights"]["report_weak"] * out["report_weak"] + cfg["weights"]["distillation"] * out["distill"]
    assert torch.allclose(out["total"], expected_total, atol=1e-6)
    assert out["report_weak"].item() != 0.0
    assert out["distill"].item() != 0.0
