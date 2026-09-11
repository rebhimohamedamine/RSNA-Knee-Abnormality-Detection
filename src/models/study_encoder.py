"""Level 3 (part 2): pools per-series embeddings (after cross-series
attention) into one fixed-size study embedding.

`masked_mean` is the only pooling implemented this pass and the default for
M1-M7: mean over real series only, padded series excluded from both the sum
and the denominator. `cls_token` is named as a config option and a documented
future extension point (would need a permanent-True slot prepended to
`series_mask` upstream), but deliberately not implemented here to avoid
adding that plumbing before `masked_mean` is shown to need it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_SUPPORTED_POOLING = ("masked_mean",)


class StudyEncoder(nn.Module):
    def __init__(self, embedding_dim: int, pooling: str = "masked_mean"):
        super().__init__()
        if pooling == "cls_token":
            raise NotImplementedError(
                "study_pooling='cls_token' is a named future extension point, not implemented in "
                "this pass -- use 'masked_mean'."
            )
        if pooling not in _SUPPORTED_POOLING:
            raise ValueError(f"Unknown study_pooling: {pooling!r} (expected one of {_SUPPORTED_POOLING})")
        self.pooling = pooling
        self.embedding_dim = embedding_dim

    def forward(self, series_embeddings: torch.Tensor, series_mask: torch.Tensor) -> torch.Tensor:
        """`series_embeddings`: (B, S, D); `series_mask`: (B, S) bool -> (B, D)."""
        mask = series_mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (series_embeddings * mask).sum(dim=1) / denom
