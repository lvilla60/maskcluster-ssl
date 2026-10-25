"""Small Vision Transformer implemented with general PyTorch primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.image_size = tuple(image_size)
        self.patch_size = int(patch_size)
        if any(size % self.patch_size for size in self.image_size):
            raise ValueError("Image dimensions must be divisible by patch size")
        self.base_grid = tuple(size // self.patch_size for size in self.image_size)
        self.projection = nn.Conv2d(in_channels, embed_dim, self.patch_size, self.patch_size)

    def forward(self, images: Tensor) -> tuple[Tensor, tuple[int, int]]:
        features = self.projection(images)
        grid = (features.shape[-2], features.shape[-1])
        return features.flatten(2).transpose(1, 2), grid


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized,
                                     key_padding_mask=key_padding_mask, need_weights=False)
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


@dataclass(frozen=True)
class ViTOutput:
    cls: Tensor
    patches: Tensor
    grid: tuple[int, int]
    intermediate_patches: tuple[Tensor, ...]


class VisionTransformer(nn.Module):
    """ViT with interpolated rectangular positions and selectable intermediates."""

    def __init__(
        self,
        image_size=(128, 64),
        patch_size=16,
        in_channels=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_mode="visible",
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        if attention_mode not in {"visible", "standard"}:
            raise ValueError("attention_mode must be 'visible' or 'standard'")
        self.attention_mode = attention_mode
        patch_count = self.patch_embed.base_grid[0] * self.patch_embed.base_grid[1]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, patch_count + 1, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def patch_size(self) -> int:
        return self.patch_embed.patch_size

    def _position(self, grid: tuple[int, int]) -> Tensor:
        if grid == self.patch_embed.base_grid:
            return self.position
        cls_position, patch_position = self.position[:, :1], self.position[:, 1:]
        base_h, base_w = self.patch_embed.base_grid
        patch_position = patch_position.reshape(1, base_h, base_w, self.embed_dim).permute(0, 3, 1, 2)
        patch_position = F.interpolate(
            patch_position, size=grid, mode="bicubic", align_corners=False, antialias=True
        )
        patch_position = patch_position.permute(0, 2, 3, 1).reshape(1, grid[0] * grid[1], self.embed_dim)
        return torch.cat((cls_position, patch_position), dim=1)

    def forward_features(self, images: Tensor, intermediate_layers=(), patch_mask: Tensor | None = None) -> ViTOutput:
        patches, grid = self.patch_embed(images)
        cls = self.cls_token.expand(len(images), -1, -1)
        tokens = torch.cat((cls, patches), dim=1) + self._position(grid)
        key_padding_mask = None
        if patch_mask is not None:
            if tuple(patch_mask.shape) != (len(images), *grid):
                raise ValueError("Patch mask does not match the token grid")
            if self.attention_mode == "visible":
                key_padding_mask = torch.cat((
                    torch.zeros((len(images), 1), dtype=torch.bool, device=images.device),
                    patch_mask.reshape(len(images), -1).to(device=images.device, dtype=torch.bool),
                ), dim=1)
        requested = set(int(index) for index in intermediate_layers)
        if any(index < 1 or index > self.depth for index in requested):
            raise ValueError(f"Intermediate layers must be in [1, {self.depth}]")
        intermediates = []
        for index, block in enumerate(self.blocks, start=1):
            tokens = block(tokens, key_padding_mask)
            if index in requested:
                intermediates.append(tokens[:, 1:])
        tokens = self.norm(tokens)
        return ViTOutput(tokens[:, 0], tokens[:, 1:], grid, tuple(intermediates))

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_features(images).cls
