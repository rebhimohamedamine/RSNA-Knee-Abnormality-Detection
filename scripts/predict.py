#!/usr/bin/env python
"""Run MRI+metadata-only inference on the test split and write submission.csv.
Thin orchestration only -- all logic lives in src/.

    python scripts/predict.py --config configs/final.yaml --checkpoint checkpoints/m7_final/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parent.parent
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))  # so `python scripts/predict.py` finds src/ regardless of cwd

import torch

from src.data.dataset import build_test_dataset
from src.inference.predict import run_inference
from src.inference.submission import write_submission
from src.models.knee_model import KneeModel
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger("scripts.predict")


def resolve_path(research_root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else research_root / p


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run_prediction(
    cfg: dict,
    checkpoint_path: str,
    device: torch.device,
    research_root: Path = RESEARCH_ROOT,
    studies_csv: str | Path | None = None,
    series_csv: str | Path | None = None,
    out_path: str | Path | None = None,
    batch_size: int = 8,
) -> Path:
    """Builds the test dataset, loads the checkpoint, runs MRI+metadata-only
    inference, and writes submission.csv. Factored out of `main()` so
    `tests/test_end_to_end_smoke.py` can call it directly against a
    sandboxed `research_root`. Returns the path written."""
    dataset = build_test_dataset(cfg, research_root, studies_csv=studies_csv, series_csv=series_csv)
    logger.info("Running inference on %d test studies.", len(dataset))

    model = KneeModel(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    pred_df = run_inference(model, dataset, device, batch_size=batch_size)

    resolved_out = Path(out_path) if out_path else resolve_path(research_root, cfg["paths"]["output_root"]) / "predictions" / "submission.csv"
    write_submission(pred_df, resolve_path(research_root, cfg["paths"]["sample_submission_csv"]), resolved_out)
    return resolved_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-csv", default=None, help="Defaults to paths.test_csv from the config.")
    parser.add_argument("--series-csv", default=None, help="Defaults to paths.test_series_csv from the config.")
    parser.add_argument("--out", default=None, help="Defaults to outputs/predictions/submission.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = select_device(args.device)
    run_prediction(
        cfg, args.checkpoint, device, studies_csv=args.input_csv, series_csv=args.series_csv,
        out_path=args.out, batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
