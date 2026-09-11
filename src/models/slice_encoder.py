"""Level 1 of the hierarchy: 2D CNN slice -> embedding.

Backbone is configurable (`model.backbone` in the run config); MRI slices are
single-channel, so a non-RGB `in_channels` (the default, 1) triggers
replacing the backbone's first conv and, when `pretrained=True`, initializing
it by averaging the pretrained RGB filters across the channel dimension and
repeating that average -- a standard, cheap way to reuse ImageNet features on
grayscale input without discarding them.

`pretrained=True` downloads torchvision's ImageNet weights over the network;
on an internet-disabled Kaggle notebook this needs a pre-cached torch hub
directory or a Kaggle model dataset providing the weights file instead --
out of scope for this pass (local dev runs with `pretrained=False`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models

# name -> (constructor, ImageNet-weights enum, output feature dim before fc)
_BACKBONES = {
    "resnet18": (tv_models.resnet18, tv_models.ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet34": (tv_models.resnet34, tv_models.ResNet34_Weights.IMAGENET1K_V1, 512),
    "resnet50": (tv_models.resnet50, tv_models.ResNet50_Weights.IMAGENET1K_V2, 2048),
}


class SliceEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", embedding_dim: int = 128, pretrained: bool = False, in_channels: int = 1):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone!r} (expected one of {list(_BACKBONES)})")

        ctor, weights_enum, feat_dim = _BACKBONES[backbone]
        net = ctor(weights=weights_enum if pretrained else None)

        if in_channels != 3:
            old_conv = net.conv1
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None,
            )
            if pretrained:
                with torch.no_grad():
                    avg_weight = old_conv.weight.mean(dim=1, keepdim=True)  # (out_ch, 1, k, k)
                    new_conv.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1))
            net.conv1 = new_conv

        net.fc = nn.Identity()
        self.backbone = net
        self.projection = nn.Linear(feat_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, in_channels, H, W) -> (N, embedding_dim)`."""
        return self.projection(self.backbone(x))
