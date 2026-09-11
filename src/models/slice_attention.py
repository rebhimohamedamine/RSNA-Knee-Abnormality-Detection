"""Pools a series' slice embeddings into one series embedding.

One class implements both M1 (`enabled=False`: masked mean -- equal weight
over real slices) and M2 (`enabled=True`: a learned attention score per
slice) so the M1->M2 ablation is a single config flag, not a different
module. Padded slice positions never influence the pooled result in either
mode: scores are masked to `-inf` before softmax, and the mean's denominator
counts only real slices.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedAttentionPool(nn.Module):
    def __init__(self, embedding_dim: int, enabled: bool = True, hidden_dim: int | None = None):
        super().__init__()
        self.enabled = enabled
        hidden_dim = hidden_dim or embedding_dim
        if enabled:
            self.score_proj = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, slice_embeddings: torch.Tensor, slice_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`slice_embeddings`: (B, K, D); `slice_mask`: (B, K) bool.
        Returns `(pooled (B, D), weights (B, K))`."""
        mask_float = slice_mask.float()
        denom = mask_float.sum(dim=-1, keepdim=True).clamp_min(1.0)

        if not self.enabled:
            weights = mask_float / denom
            pooled = (weights.unsqueeze(-1) * slice_embeddings).sum(dim=1)
            return pooled, weights

        scores = self.score_proj(slice_embeddings).squeeze(-1)  # (B, K)
        scores = scores.masked_fill(~slice_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        # An all-padded row would softmax -inf/-inf to NaN; replace with 0s
        # (its `series_mask` is False anyway, so it never reaches the study
        # embedding -- but NaNs must not appear in gradients either way).
        has_any_real = slice_mask.any(dim=-1, keepdim=True)
        weights = torch.where(has_any_real, weights, torch.zeros_like(weights))
        pooled = (weights.unsqueeze(-1) * slice_embeddings).sum(dim=1)
        return pooled, weights
