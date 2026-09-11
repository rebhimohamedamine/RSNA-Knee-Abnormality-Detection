"""The training loop: one epoch of training, one full-validation pass, and
`fit()` tying epochs together with checkpointing/early-stopping/logging
callbacks.

`validate()` always calls the model with `report_texts=None` -- it reuses
the exact inference contract, so model selection (which checkpoint counts as
"best") never depends on report access the test set won't have.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.reports.weak_labels import batch_extract_weak_labels
from src.training.losses import LossFn, combined_loss
from src.training.metrics import compute_all_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        cfg: dict,
        device: torch.device,
        bce_fn: LossFn,
        callbacks: dict[str, Any] | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device
        self.bce_fn = bce_fn
        self.callbacks = callbacks or {}

        self.label_names = cfg["labels"]["columns"]
        self.use_report = cfg["model"]["report_weak_supervision"]["enabled"] or cfg["model"]["distillation"]["enabled"]
        self.use_amp = bool(cfg["train"]["mixed_precision"]) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)
        self.distillation_temperature = cfg["model"]["distillation"].get("temperature", 2.0)
        self.grad_clip_norm = cfg["train"].get("gradient_clip_norm")

    def _move_batch(self, batch: dict) -> dict:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _weak_labels_for(self, report_texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        labels_np, confidence_np = batch_extract_weak_labels(report_texts)
        labels = torch.from_numpy(labels_np).float().to(self.device)
        confidence = torch.from_numpy(confidence_np).float().to(self.device)
        return labels, confidence

    def train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        totals = {"total": 0.0, "bce": 0.0, "report_weak": 0.0, "distill": 0.0}
        n_batches = 0

        for batch in self.train_loader:
            batch = self._move_batch(batch)
            report_texts = batch["report_text"] if self.use_report else None

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(batch, report_texts=report_texts)

                weak_labels = weak_confidence = None
                if self.use_report and outputs["teacher_logits"] is not None:
                    weak_labels, weak_confidence = self._weak_labels_for(report_texts)

                losses = combined_loss(
                    outputs["logits"], batch["labels"], batch["label_mask"], self.cfg["loss"], self.bce_fn,
                    teacher_logits=outputs["teacher_logits"], weak_labels=weak_labels, weak_confidence=weak_confidence,
                    distillation_temperature=self.distillation_temperature,
                )

            self.scaler.scale(losses["total"]).backward()
            if self.grad_clip_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()

            for key in totals:
                totals[key] += float(losses[key].detach().item())
            n_batches += 1

        return {key: value / max(1, n_batches) for key, value in totals.items()}

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        all_labels, all_probs, all_masks = [], [], []

        for batch in self.val_loader:
            batch = self._move_batch(batch)
            outputs = self.model(batch, report_texts=None)  # inference contract: no report, always
            probs = torch.sigmoid(outputs["logits"])
            all_labels.append(batch["labels"].cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_masks.append(batch["label_mask"].cpu().numpy())

        y_true = np.concatenate(all_labels, axis=0)
        y_prob = np.concatenate(all_probs, axis=0)
        label_mask = np.concatenate(all_masks, axis=0)
        return compute_all_metrics(
            y_true, y_prob, label_mask, self.label_names,
            threshold=self.cfg["eval"]["threshold"], calibration_bins=self.cfg["eval"]["calibration_bins"],
        )

    def fit(self, epochs: int) -> dict:
        history = []
        val_metrics: dict = {}

        for epoch in range(epochs):
            train_losses = self.train_one_epoch()
            val_metrics = self.validate()
            score = val_metrics["macro_auc"]
            if np.isnan(score):
                logger.warning("Epoch %d: validation macro AUC is undefined (too few labeled rows in this split).", epoch)
                score = -float("inf")

            record = {"epoch": epoch, "train_losses": train_losses, "val_macro_auc": val_metrics["macro_auc"]}
            if "logging" in self.callbacks:
                self.callbacks["logging"](record)

            if "checkpoint" in self.callbacks:
                self.callbacks["checkpoint"](epoch, self.model, self.optimizer, self.cfg, val_metrics, score)

            history.append(record)

            if "early_stopping" in self.callbacks and self.callbacks["early_stopping"](score):
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        return {"history": history, "final_val_metrics": val_metrics}
