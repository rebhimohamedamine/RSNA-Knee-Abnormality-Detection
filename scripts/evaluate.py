#!/usr/bin/env python
"""Evaluate a checkpoint on the validation (or, once real labels exist, test)
split and save metrics + raw predictions for later analysis. Thin
orchestration only -- all logic lives in src/.

    python scripts/evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/m1_baseline/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parent.parent
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))  # so `python scripts/evaluate.py` finds src/ regardless of cwd

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import build_test_dataset, build_train_val_datasets
from src.models.knee_model import KneeModel
from src.training.metrics import compute_all_metrics
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger("scripts.evaluate")


def resolve_path(research_root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else research_root / p


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _append_ablation_row(cfg: dict, macro_auc: float, out_dir: Path) -> None:
    m = cfg["model"]
    row = {
        "Model": cfg.get("run_name", "unnamed_run"),
        "SliceAttn": m["slice_attention"]["enabled"],
        "Metadata": m["metadata"]["enabled"],
        "CrossSeries": m["cross_series_attention"]["enabled"],
        "Pathology": m["pathology_attention"]["enabled"],
        "Report": m["report_weak_supervision"]["enabled"],
        "MacroAUC": macro_auc,
    }
    table_path = out_dir / "ablation_table.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    combined = pd.concat([pd.read_csv(table_path), new_row], ignore_index=True) if table_path.exists() else new_row
    combined.to_csv(table_path, index=False)
    logger.info("Appended ablation row for %r to %s", row["Model"], table_path)


def run_evaluation(cfg: dict, checkpoint_path: str, split: str, device: torch.device, research_root: Path = RESEARCH_ROOT) -> dict:
    model = KneeModel(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    if split == "val":
        _, dataset = build_train_val_datasets(cfg, research_root)
    elif split == "test":
        dataset = build_test_dataset(cfg, research_root)
    else:
        raise ValueError(f"Unknown split: {split!r} (expected val | test)")

    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False)
    study_uids, all_labels, all_probs, all_masks = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(moved, report_texts=None)  # evaluation always uses the inference contract
            probs = torch.sigmoid(outputs["logits"])
            study_uids.extend(moved["study_uid"])
            all_labels.append(moved["labels"].cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_masks.append(moved["label_mask"].cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    label_mask = np.concatenate(all_masks, axis=0)
    metrics = compute_all_metrics(
        y_true, y_prob, label_mask, cfg["labels"]["columns"],
        threshold=cfg["eval"]["threshold"], calibration_bins=cfg["eval"]["calibration_bins"],
    )
    return {"metrics": metrics, "study_uids": study_uids, "y_true": y_true, "y_prob": y_prob, "label_mask": label_mask}


def evaluate_and_save(
    cfg: dict, checkpoint_path: str, split: str, device: torch.device,
    research_root: Path = RESEARCH_ROOT, out_path: str | Path | None = None,
) -> dict:
    """`run_evaluation` plus saving metrics.json, predictions.npz, and (for
    the val split) an ablation_table.csv row. Factored out of `main()` so
    `tests/test_end_to_end_smoke.py` can call it directly against a
    sandboxed `research_root`."""
    run_name = cfg.get("run_name", "unnamed_run")
    result = run_evaluation(cfg, checkpoint_path, split, device, research_root=research_root)

    metrics_dir = resolve_path(research_root, cfg["paths"]["output_root"]) / "metrics" / run_name
    out_path = Path(out_path) if out_path else metrics_dir / f"metrics_{split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)

    predictions_path = out_path.parent / f"predictions_{split}.npz"
    np.savez(
        predictions_path,
        study_uid=np.array(result["study_uids"]), y_true=result["y_true"],
        y_prob=result["y_prob"], label_mask=result["label_mask"],
    )
    logger.info("Wrote metrics to %s and predictions to %s (macro AUC = %s)", out_path, predictions_path, result["metrics"]["macro_auc"])

    if split == "val":
        _append_ablation_row(cfg, result["metrics"]["macro_auc"], resolve_path(research_root, cfg["paths"]["output_root"]) / "metrics")

    return {"out_path": out_path, "predictions_path": predictions_path, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--out", default=None, help="Metrics JSON output path (default: outputs/metrics/<run_name>/metrics_<split>.json)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _select_device(args.device)
    evaluate_and_save(cfg, args.checkpoint, args.split, device, out_path=args.out)


if __name__ == "__main__":
    main()
