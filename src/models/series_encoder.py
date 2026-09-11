"""Level 2 of the hierarchy: a study's slices -> one embedding per series.

Processes series **one at a time** (`for s in range(S)`), encoding that
series' slices through the CNN, attention-pooling them, and freeing the
slice-level activations before moving to the next series. This is the
spec's GPU-memory rule -- never build one giant `(B, S, K, ...)` activation
tensor through the CNN at once -- implemented directly in the forward pass.
`S` (max_series, ~6) is small enough that the Python-level loop costs
nothing next to the memory it saves.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.metadata_encoder import SeriesMetadataEncoder
from src.models.slice_attention import MaskedAttentionPool
from src.models.slice_encoder import SliceEncoder


class SeriesEncoder(nn.Module):
    def __init__(
        self,
        slice_encoder: SliceEncoder,
        slice_attention: MaskedAttentionPool,
        metadata_encoder: SeriesMetadataEncoder | None,
        embedding_dim: int,
    ):
        super().__init__()
        self.slice_encoder = slice_encoder
        self.slice_attention = slice_attention
        self.metadata_encoder = metadata_encoder
        self.fusion_proj = (
            nn.Linear(embedding_dim + metadata_encoder.output_dim, embedding_dim)
            if metadata_encoder is not None
            else None
        )

    def forward(
        self,
        pixel_values: torch.Tensor,        # (B, S, K, C, H, W)
        slice_mask: torch.Tensor,          # (B, S, K)
        plane_id: torch.Tensor,            # (B, S)
        fluid_sensitive: torch.Tensor,     # (B, S)
        fat_suppression: torch.Tensor,     # (B, S)
        return_attention: bool = False,
    ):
        B, S, K, C, H, W = pixel_values.shape
        series_embeddings: list[torch.Tensor] = []
        attention_weights: list[torch.Tensor] = [] if return_attention else None

        for s in range(S):
            slices = pixel_values[:, s].reshape(B * K, C, H, W)
            slice_feats = self.slice_encoder(slices).view(B, K, -1)
            pooled, weights = self.slice_attention(slice_feats, slice_mask[:, s])
            del slices, slice_feats  # free before encoding the next series

            if self.metadata_encoder is not None:
                meta = self.metadata_encoder(plane_id[:, s], fluid_sensitive[:, s], fat_suppression[:, s])
                pooled = self.fusion_proj(torch.cat([pooled, meta], dim=-1))

            series_embeddings.append(pooled)
            if return_attention:
                attention_weights.append(weights)

        series_out = torch.stack(series_embeddings, dim=1)  # (B, S, D)
        if return_attention:
            return series_out, torch.stack(attention_weights, dim=1)  # (B, S, K)
        return series_out
