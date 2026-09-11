"""MRI+metadata-only inference: runs a trained model over a dataset and
returns per-study probabilities. Never passes report text to the model --
this is the same inference contract `KneeModel.forward` enforces internally,
applied consistently here too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.reports.label_mapping import TARGETS as LABEL_ORDER
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_inference(model: torch.nn.Module, dataset, device: torch.device, batch_size: int = 8) -> pd.DataFrame:
    """Returns a DataFrame with columns `["StudyInstanceUID", *LABEL_ORDER]`,
    one row per study in `dataset`, with continuous sigmoid probabilities."""
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    study_uids: list[str] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(moved, report_texts=None)  # MRI + metadata only, per the inference contract
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            all_probs.append(probs)
            study_uids.extend(moved["study_uid"])

    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(LABEL_ORDER)))
    df = pd.DataFrame(probs, columns=LABEL_ORDER)
    df.insert(0, "StudyInstanceUID", study_uids)
    return df
