"""Checkpointing, early stopping, and per-epoch history logging."""

from __future__ import annotations

import json
from pathlib import Path

import torch


class CheckpointCallback:
    """Saves `last.pt` every call, and `best.pt` whenever the monitored
    score improves. Each checkpoint bundles the model/optimizer state with
    the epoch, config, and metrics, so a run is fully reproducible from one
    file (spec section 39)."""

    def __init__(self, checkpoint_dir: str | Path, mode: str = "max"):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best_score = -float("inf") if mode == "max" else float("inf")

    def _is_improvement(self, score: float) -> bool:
        return score > self.best_score if self.mode == "max" else score < self.best_score

    def __call__(self, epoch: int, model, optimizer, cfg: dict, metrics: dict, score: float) -> bool:
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": cfg,
            "metrics": metrics,
        }
        torch.save(payload, self.checkpoint_dir / "last.pt")

        improved = self._is_improvement(score)
        if improved:
            self.best_score = score
            torch.save(payload, self.checkpoint_dir / "best.pt")
        return improved


class EarlyStoppingCallback:
    """Returns `True` (meaning "stop now") once `patience` consecutive
    calls have failed to improve on the best score seen so far."""

    def __init__(self, patience: int, mode: str = "max"):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.num_bad_epochs = 0

    def __call__(self, score: float) -> bool:
        improved = score > self.best_score if self.mode == "max" else score < self.best_score
        if improved:
            self.best_score = score
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


class LoggingCallback:
    """Appends one JSON record per call to `outputs/logs/<run>/history.jsonl`."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, record: dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
