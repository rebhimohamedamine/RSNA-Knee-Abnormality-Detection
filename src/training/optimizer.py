"""Optimizer and LR scheduler construction from a run config."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """AdamW, with an optional separate (typically lower) learning rate for
    the pretrained slice-encoder backbone via `optim.backbone_lr_mult`."""
    optim_cfg = cfg["optim"]
    lr = optim_cfg["lr"]
    weight_decay = optim_cfg["weight_decay"]
    backbone_lr_mult = optim_cfg.get("backbone_lr_mult", 1.0)

    if backbone_lr_mult != 1.0 and hasattr(model, "slice_encoder"):
        backbone_param_ids = {id(p) for p in model.slice_encoder.parameters()}
        backbone_params = [p for p in model.parameters() if id(p) in backbone_param_ids]
        other_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
        param_groups = [
            {"params": backbone_params, "lr": lr * backbone_lr_mult},
            {"params": other_params, "lr": lr},
        ]
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, steps_per_epoch: int):
    """`optim.schedule`: `"cosine"` (linear warmup then cosine decay to 0
    over the full run) or `"none"` (constant LR, returns `None`)."""
    optim_cfg = cfg["optim"]
    schedule = optim_cfg.get("schedule", "none")
    if schedule == "none":
        return None

    if schedule == "cosine":
        warmup_steps = max(0, optim_cfg.get("warmup_steps", 0))
        total_steps = max(1, steps_per_epoch * cfg["train"]["epochs"])

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unknown optim.schedule: {schedule!r} (expected cosine | none)")
