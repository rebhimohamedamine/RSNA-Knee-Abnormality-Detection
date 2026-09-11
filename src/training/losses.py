"""Training losses.

All per-label reductions here are mask/confidence-weighted means, never
simple `.mean()` calls -- the dominant case in this dataset is a row with no
ground-truth labels at all (only 58 of 4407 local studies have real labels;
the rest are all-NaN), and those rows must contribute exactly zero to the
ground-truth loss rather than being silently treated as negatives.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _masked_mean(per_element: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted mean of `per_element` by `weight` (a 0/1 mask or a
    continuous confidence in [0, 1]). Rows/labels with weight 0 are fully
    excluded, including from the denominator -- not merely zeroed out and
    still divided by the full element count."""
    return (per_element * weight).sum() / weight.sum().clamp_min(1e-8)


def sigmoid_focal_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0, alpha: float | None = 0.25) -> torch.Tensor:
    """Elementwise sigmoid focal loss (Lin et al., RetinaNet). `alpha=None`
    (or any negative value) disables the class-balance reweighting term, so
    that `gamma=0, alpha=None` reduces exactly to plain BCE-with-logits --
    the property `tests/test_losses.py` checks."""
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p_t = prob * labels + (1 - prob) * (1 - labels)
    loss = ce * (1 - p_t).clamp_min(0.0).pow(gamma)
    if alpha is not None and alpha >= 0:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss
    return loss


def compute_pos_weight(labels: torch.Tensor, label_mask: torch.Tensor, max_weight: float = 100.0) -> torch.Tensor:
    """Per-label `pos_weight` for `BCEWithLogitsLoss`, computed from
    LABELED-only rows (`label_mask == 1`) as `n_negative / n_positive`,
    clamped to `[1/max_weight, max_weight]` to avoid an unstably large weight
    for a target with very few positive examples in the training split."""
    pos = (labels * label_mask).sum(dim=0)
    total = label_mask.sum(dim=0)
    neg = (total - pos).clamp_min(0.0)
    weight = neg / pos.clamp_min(1e-6)
    return weight.clamp(min=1.0 / max_weight, max=max_weight)


def build_bce_term(loss_cfg: dict, pos_weight: torch.Tensor | None = None) -> LossFn:
    """Builds the ground-truth BCE loss term per `loss_cfg["class_imbalance_strategy"]`
    (`"plain"`, `"pos_weight"`, or `"focal"` -- exactly one active, per spec).
    Returns `fn(logits, labels, label_mask) -> scalar`."""
    strategy = loss_cfg["class_imbalance_strategy"]

    if strategy == "plain":
        def bce(logits: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
            per_element = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            return _masked_mean(per_element, label_mask)
        return bce

    if strategy == "pos_weight":
        if pos_weight is None:
            raise ValueError("class_imbalance_strategy='pos_weight' requires a precomputed pos_weight tensor")

        def bce(logits: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
            per_element = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=pos_weight)
            return _masked_mean(per_element, label_mask)
        return bce

    if strategy == "focal":
        gamma = loss_cfg["focal"]["gamma"]
        alpha = loss_cfg["focal"]["alpha"]

        def bce(logits: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
            per_element = sigmoid_focal_loss(logits, labels, gamma=gamma, alpha=alpha)
            return _masked_mean(per_element, label_mask)
        return bce

    raise ValueError(f"Unknown class_imbalance_strategy: {strategy!r} (expected plain | pos_weight | focal)")


def report_weak_supervision_term(teacher_logits: torch.Tensor, weak_labels: torch.Tensor, weak_confidence: torch.Tensor) -> torch.Tensor:
    """BCE between the report teacher's logits and report-derived weak
    labels, weighted by per-label confidence (0 confidence -- no mention
    found in the report -- fully excludes that (row, label) pair)."""
    per_element = F.binary_cross_entropy_with_logits(teacher_logits, weak_labels, reduction="none")
    return _masked_mean(per_element, weak_confidence)


def distillation_term(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Per-label binary KL divergence KL(teacher || student) at temperature
    `T`, teacher detached. Using true KL (rather than plain soft-target
    cross-entropy) means this term is exactly 0 when the student's
    predictions already match the teacher's -- not merely at its minimum --
    which is what `tests/test_losses.py` checks."""
    eps = 1e-7
    student_prob = torch.sigmoid(student_logits / temperature).clamp(eps, 1 - eps)
    teacher_prob = torch.sigmoid(teacher_logits.detach() / temperature).clamp(eps, 1 - eps)
    kl = teacher_prob * torch.log(teacher_prob / student_prob) + (1 - teacher_prob) * torch.log((1 - teacher_prob) / (1 - student_prob))
    return (kl * (temperature**2)).mean()


def combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    loss_cfg: dict,
    bce_fn: LossFn,
    teacher_logits: torch.Tensor | None = None,
    weak_labels: torch.Tensor | None = None,
    weak_confidence: torch.Tensor | None = None,
    distillation_temperature: float = 2.0,
) -> dict[str, torch.Tensor]:
    """`L_total = w1*L_bce + w2*L_report_weak + w3*L_distillation`.

    Always returns all four keys (`total`, `bce`, `report_weak`, `distill`),
    with a `0.0` tensor for any disabled/unavailable term, so logged run
    history has one stable schema across every M1..M7 config."""
    weights = loss_cfg["weights"]
    zero = logits.new_zeros(())

    bce = bce_fn(logits, labels, label_mask)

    report_weak = zero
    if weights.get("report_weak", 0.0) > 0 and teacher_logits is not None and weak_labels is not None and weak_confidence is not None:
        report_weak = report_weak_supervision_term(teacher_logits, weak_labels, weak_confidence)

    distill = zero
    if weights.get("distillation", 0.0) > 0 and teacher_logits is not None:
        distill = distillation_term(logits, teacher_logits, temperature=distillation_temperature)

    total = weights["bce"] * bce + weights.get("report_weak", 0.0) * report_weak + weights.get("distillation", 0.0) * distill
    return {"total": total, "bce": bce, "report_weak": report_weak, "distill": distill}
