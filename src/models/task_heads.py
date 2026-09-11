"""The 12 independent binary prediction heads (multi-label: 12 sigmoids, not
one softmax -- a study can have several abnormalities at once).

`LABEL_ORDER` is imported from `src.reports.label_mapping.TARGETS` rather
than redefined here, so the model's output order and the report
weak-supervision module's target order can never drift apart. That import
costs nothing extra: `label_mapping.py` depends only on the stdlib `re`
module, not on any of the report branch's heavier text-encoder machinery.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.pathology_attention import PATHOLOGY_GROUPS
from src.reports.label_mapping import TARGETS as LABEL_ORDER

LABEL_TO_GROUP: dict[str, str] = {label: group for group, labels in PATHOLOGY_GROUPS.items() for label in labels}
assert set(LABEL_TO_GROUP.keys()) == set(LABEL_ORDER), "Every label must belong to exactly one pathology group"


class TaskHeads(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.heads = nn.ModuleDict({label: nn.Linear(embedding_dim, 1) for label in LABEL_ORDER})

    def forward(self, group_embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """`group_embeddings`: `{group_name: (B, D)}` (from
        `PathologyAttentionModule`) -> raw logits `(B, 12)` in `LABEL_ORDER`.
        Sigmoid is applied by the caller (metrics/inference), never here."""
        logits = [self.heads[label](group_embeddings[LABEL_TO_GROUP[label]]) for label in LABEL_ORDER]
        return torch.cat(logits, dim=-1)
