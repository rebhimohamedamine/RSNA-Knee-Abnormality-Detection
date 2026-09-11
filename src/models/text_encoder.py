"""Encodes report free text into a fixed-size embedding for the auxiliary
report branch (weak supervision / distillation teacher). Never used at
inference -- see `src/models/knee_model.py`'s `report_texts=None` contract.

Default model is a small, CPU-tractable, multilingual sentence encoder
(`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`), chosen
because the reports actually present in `data/train.csv` are multilingual
(Spanish, Dutch, etc. observed), not because it's state of the art. Config
tests/dev runs can point `model_name` at a tiny HF test model (e.g.
`hf-internal-testing/tiny-random-BertModel`) to exercise the same code path
without downloading a real multilingual model -- see
`configs/tiny_cpu_smoke.yaml`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ReportTextEncoder(nn.Module):
    def __init__(self, model_name: str, embedding_dim: int, freeze_backbone: bool = True, max_length: int = 256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        hidden_size = self.backbone.config.hidden_size
        self.projection = nn.Linear(hidden_size, embedding_dim)

    @staticmethod
    def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1e-9)
        return summed / counts

    def forward(self, report_texts: list[str]) -> torch.Tensor:
        """`report_texts`: list of `B` strings -> `(B, embedding_dim)`."""
        device = self.projection.weight.device
        encoded = self.tokenizer(
            list(report_texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",
        ).to(device)

        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(**encoded)
        else:
            outputs = self.backbone(**encoded)

        pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        return self.projection(pooled)
