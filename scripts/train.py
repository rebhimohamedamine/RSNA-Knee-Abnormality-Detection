#!/usr/bin/env python
"""Train one experiment from a config. Thin orchestration only -- all logic
lives in src/.

    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/final.yaml --run-name m7_run1 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parent.parent
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))  # so `python scripts/train.py` finds src/ regardless of cwd

import torch
from torch.utils.data import DataLoader

from src.data.dataset import KneeStudyDataset, build_train_val_datasets
from src.models.knee_model import KneeModel
from src.training.callbacks import CheckpointCallback, EarlyStoppingCallback, LoggingCallback
from src.training.losses import build_bce_term, compute_pos_weight
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.utils.config import load_config, save_config
from src.utils.logging import add_file_handler, get_logger
from src.utils.seed import seed_worker, set_seed

logger = get_logger("scripts.train")


def resolve_path(research_root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else research_root / p


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_pos_weight_if_needed(cfg: dict, train_ds: KneeStudyDataset) -> torch.Tensor | None:
    if cfg["loss"]["class_imbalance_strategy"] != "pos_weight":
        return None
    all_labels, all_masks = [], []
    for i in range(len(train_ds)):
        item = train_ds[i]
        all_labels.append(item["labels"])
        all_masks.append(item["label_mask"])
    return compute_pos_weight(torch.stack(all_labels), torch.stack(all_masks))


def run_training(
    cfg: dict,
    research_root: Path,
    run_name: str | None = None,
    device: torch.device | None = None,
    resume: str | Path | None = None,
) -> dict:
    """Runs one full training experiment from an already-loaded config.
    Factored out of `main()` so it can be called directly (e.g. by
    `tests/test_end_to_end_smoke.py`) with a config pointed at a sandboxed
    `research_root`, without going through argparse/sys.argv."""
    run_name = run_name or cfg.get("run_name", "unnamed_run")
    device = device or select_device("auto")
    set_seed(cfg["seed"])

    checkpoint_dir = resolve_path(research_root, cfg["paths"]["checkpoint_root"]) / run_name
    log_path = resolve_path(research_root, cfg["paths"]["output_root"]) / "logs" / run_name / "history.jsonl"
    add_file_handler(logger, log_path.parent / "train.log")
    save_config(cfg, checkpoint_dir / "config.yaml")

    logger.info("Run %r on device %s", run_name, device)

    train_ds, val_ds = build_train_val_datasets(cfg, research_root)
    logger.info("Study-level split: %d train / %d val studies.", len(train_ds), len(val_ds))

    num_workers = cfg["data"].get("num_workers", 0)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=num_workers, worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=num_workers)

    model = KneeModel(cfg)

    pos_weight = build_pos_weight_if_needed(cfg, train_ds)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    bce_fn = build_bce_term(cfg["loss"], pos_weight=pos_weight)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=max(1, len(train_loader)))

    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        logger.info("Resumed from %s (epoch %d).", resume, checkpoint.get("epoch", -1))

    callbacks = {
        "checkpoint": CheckpointCallback(checkpoint_dir, mode="max"),
        "early_stopping": EarlyStoppingCallback(cfg["train"]["early_stopping_patience"], mode="max"),
        "logging": LoggingCallback(log_path),
    }

    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, cfg, device, bce_fn, callbacks=callbacks)
    result = trainer.fit(cfg["train"]["epochs"])
    result["checkpoint_dir"] = checkpoint_dir

    logger.info("Finished %r: final val macro AUC = %s", run_name, result["final_val_metrics"].get("macro_auc"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None, help="Overrides run_name from the config.")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint (.pt) to resume model+optimizer state from.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_training(cfg, RESEARCH_ROOT, run_name=args.run_name, device=select_device(args.device), resume=args.resume)


if __name__ == "__main__":
    main()
