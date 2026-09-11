"""Maps a report's text embedding to the same 12 targets the MRI branch
predicts, for weak-supervision loss and/or knowledge distillation. Training
only -- see `src/models/knee_model.py`'s `report_texts=None` inference
contract, which guarantees this module is never invoked at test time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.text_encoder import ReportTextEncoder
from src.reports.label_mapping import TARGETS as LABEL_ORDER


class ReportTeacher(nn.Module):
    def __init__(self, text_encoder: ReportTextEncoder, embedding_dim: int):
        super().__init__()
        self.text_encoder = text_encoder
        self.classifier = nn.Linear(embedding_dim, len(LABEL_ORDER))

    def forward(self, report_texts: list[str]) -> torch.Tensor:
        """`report_texts`: list of `B` strings -> teacher logits `(B, 12)`,
        in the same `LABEL_ORDER` as `src.models.task_heads.TaskHeads`."""
        embedding = self.text_encoder(report_texts)
        return self.classifier(embedding)
