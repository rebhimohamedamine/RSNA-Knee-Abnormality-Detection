"""Encodes each series' metadata (anatomical plane, fluid-sensitivity,
fat-suppression) into a vector fused with that series' visual embedding, so
the model can distinguish e.g. "sagittal fluid-sensitive" from "coronal
non-fluid-sensitive" rather than treating all series identically.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.series import NUM_PLANES

NUM_BINARY_STATES = 2  # Fluid_Sensitive and Fat_Suppression are each {0, 1}


class SeriesMetadataEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 16, output_dim: int | None = None):
        super().__init__()
        self.plane_embedding = nn.Embedding(NUM_PLANES, embedding_dim)
        self.fluid_embedding = nn.Embedding(NUM_BINARY_STATES, embedding_dim)
        self.fat_suppression_embedding = nn.Embedding(NUM_BINARY_STATES, embedding_dim)
        output_dim = output_dim or embedding_dim
        self.projection = nn.Linear(3 * embedding_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, plane_id: torch.Tensor, fluid_sensitive: torch.Tensor, fat_suppression: torch.Tensor) -> torch.Tensor:
        """Each input `(...,)` long -> output `(..., output_dim)`, any
        leading shape (e.g. `(B,)` or `(B, S)`)."""
        combined = torch.cat(
            [self.plane_embedding(plane_id), self.fluid_embedding(fluid_sensitive), self.fat_suppression_embedding(fat_suppression)],
            dim=-1,
        )
        return self.projection(combined)
