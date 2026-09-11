"""Lightweight pathology-group specialization on top of the shared study
embedding (spec section 18): six small residual-bottleneck adapters, one per
clinically related group, NOT twelve independent deep networks -- most of the
representation (everything up through `study_encoder`) stays shared.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PATHOLOGY_GROUPS: dict[str, list[str]] = {
    "ligament": ["ACL", "MCL"],
    "meniscus": ["Medial Meniscus", "Lateral Meniscus"],
    "oa": ["Medial OA", "Lateral OA", "PF OA"],
    "inflammation": ["Effusion", "Synovitis"],
    "cyst": ["Baker's"],
    "bone": ["Contusion", "Fracture"],
}


class PathologyAdapter(nn.Module):
    """A residual bottleneck MLP: cheap per-group specialization without
    discarding the shared study representation (the input always survives
    via the residual connection)."""

    def __init__(self, embedding_dim: int, bottleneck_ratio: int = 4):
        super().__init__()
        bottleneck_dim = max(1, embedding_dim // bottleneck_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class PathologyAttentionModule(nn.Module):
    def __init__(self, embedding_dim: int, enabled: bool = True, bottleneck_ratio: int = 4):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.adapters = nn.ModuleDict(
                {group: PathologyAdapter(embedding_dim, bottleneck_ratio) for group in PATHOLOGY_GROUPS}
            )

    def forward(self, study_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        """`study_embedding`: (B, D) -> `{group_name: (B, D)}` for every
        group in `PATHOLOGY_GROUPS`. When disabled, every group maps to the
        same (identity) study embedding, so `task_heads` never has to branch
        on this flag."""
        if not self.enabled:
            return {group: study_embedding for group in PATHOLOGY_GROUPS}
        return {group: adapter(study_embedding) for group, adapter in self.adapters.items()}
