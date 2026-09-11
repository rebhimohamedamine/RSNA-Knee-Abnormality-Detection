"""Level 3 (part 1): lets series attend to each other -- e.g. a sagittal
fluid-sensitive finding informing how a coronal series is read -- before
pooling into one study embedding.

`enabled=False` (M1-M3) is a passthrough. `enabled=True` (M4+) wraps
`nn.TransformerEncoder` with `src_key_padding_mask` so padded series never
attend to, or are attended from, real ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossSeriesTransformer(nn.Module):
    def __init__(self, embedding_dim: int, enabled: bool = True, num_heads: int = 4, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.enabled = enabled
        if enabled:
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, series_embeddings: torch.Tensor, series_mask: torch.Tensor) -> torch.Tensor:
        """`series_embeddings`: (B, S, D); `series_mask`: (B, S) bool -> (B, S, D)."""
        if not self.enabled:
            return series_embeddings

        padding_mask = ~series_mask  # torch convention: True = ignore this position
        # A study with zero real series (shouldn't occur in practice, but
        # guarded so it can't NaN the batch) would give a fully-masked row,
        # which softmaxes -inf/-inf to NaN inside the encoder. Temporarily
        # un-mask such rows -- harmless, since study_encoder's own
        # series_mask-based pooling excludes them from the result regardless.
        all_padded = ~series_mask.any(dim=-1)
        if all_padded.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padded] = False

        return self.encoder(series_embeddings, src_key_padding_mask=padding_mask)
