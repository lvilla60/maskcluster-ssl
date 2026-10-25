"""MaskCluster adaptation for image-only person representation learning."""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .vit import VisionTransformer


VARIANTS = {
    "baseline": {"patch_clustering": False, "reconstruction": False},
    "no_mim": {"patch_clustering": True, "reconstruction": False},
    "full": {"patch_clustering": True, "reconstruction": True},
}


class ClusteringHead(nn.Module):
    def __init__(self, input_dim, prototypes, hidden_dim=1024, bottleneck_dim=256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.prototypes = nn.Linear(bottleneck_dim, prototypes, bias=False)

    def forward(self, features: Tensor) -> Tensor:
        projected = F.normalize(self.projector(features), dim=-1)
        return self.prototypes(projected)


class ReconstructionHead(nn.Module):
    """Predict RGB patch pixels in [0, 1] from summed intermediate tokens."""

    def __init__(self, embed_dim, patch_size, channels=3):
        super().__init__()
        self.patch_size = int(patch_size)
        self.channels = int(channels)
        patch_values = channels * patch_size * patch_size
        self.layers = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_values),
        )

    def forward(self, patch_tokens: Tensor, grid: tuple[int, int]) -> Tensor:
        batch, token_count, _ = patch_tokens.shape
        rows, cols = grid
        if token_count != rows * cols:
            raise ValueError("Patch tokens and spatial grid are inconsistent")
        patches = torch.sigmoid(self.layers(patch_tokens))
        patches = patches.view(
            batch, rows, cols, self.channels, self.patch_size, self.patch_size
        )
        return patches.permute(0, 3, 1, 4, 2, 5).reshape(
            batch, self.channels, rows * self.patch_size, cols * self.patch_size
        )


class EncoderBranch(nn.Module):
    def __init__(self, backbone, prototypes, patch_clustering):
        super().__init__()
        self.backbone = backbone
        self.global_head = ClusteringHead(backbone.embed_dim, prototypes)
        self.patch_head = (
            ClusteringHead(backbone.embed_dim, prototypes) if patch_clustering else None
        )

    def encode(self, images: Tensor) -> Tensor:
        return self.backbone(images)

    def forward(self, images: Tensor, intermediate_layers=(), patch_mask=None) -> dict:
        features = self.backbone.forward_features(images, intermediate_layers, patch_mask)
        return {
            "features": features.cls,
            "patch_features": features.patches,
            "global_logits": self.global_head(features.cls),
            "patch_logits": self.patch_head(features.patches) if self.patch_head is not None else None,
            "grid": features.grid,
            "intermediate_patches": features.intermediate_patches,
        }


class MaskClusterSSL(nn.Module):
    """Student optimized by gradients and an identically initialized EMA teacher."""

    def __init__(
        self,
        *,
        variant="full",
        image_size=(128, 64),
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        prototypes=1024,
        reconstruction_layers=None,
        attention_mode="visible",
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
        flags = VARIANTS[variant]
        backbone = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attention_mode=attention_mode,
        )
        self.variant = variant
        self.student = EncoderBranch(backbone, prototypes, flags["patch_clustering"])
        self.teacher = copy.deepcopy(self.student)
        self.reconstruction = (
            ReconstructionHead(embed_dim, patch_size) if flags["reconstruction"] else None
        )
        if reconstruction_layers is None:
            reconstruction_layers = (6, 8, 10, 12) if depth >= 12 else tuple(range(max(1, depth - 3), depth + 1))
        self.reconstruction_layers = tuple(int(v) for v in reconstruction_layers)
        if self.reconstruction is not None and max(self.reconstruction_layers) > depth:
            raise ValueError("A reconstruction layer exceeds backbone depth")
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

    @property
    def has_patch_clustering(self) -> bool:
        return self.student.patch_head is not None

    @property
    def has_reconstruction(self) -> bool:
        return self.reconstruction is not None

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be in [0, 1]")
        for teacher, student in zip(self.teacher.parameters(), self.student.parameters()):
            teacher.lerp_(student.detach(), 1.0 - momentum)
        for teacher, student in zip(self.teacher.buffers(), self.student.buffers()):
            teacher.copy_(student)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, teacher_globals, student_globals, student_locals=(), patch_masks=None) -> dict:
        intermediate = self.reconstruction_layers if self.has_reconstruction else ()
        with torch.no_grad():
            teacher_outputs = tuple(self.teacher(view) for view in teacher_globals)
        student_global_outputs = tuple(
            self.student(view, intermediate_layers=intermediate,
                         patch_mask=None if patch_masks is None else patch_masks[index])
            for index, view in enumerate(student_globals)
        )
        student_local_outputs = tuple(self.student(view) for view in student_locals)
        reconstructions = []
        if self.reconstruction is not None:
            for output in student_global_outputs:
                tokens = torch.stack(output["intermediate_patches"], dim=0).sum(dim=0)
                reconstructions.append(self.reconstruction(tokens, output["grid"]))
        return {
            "teacher_globals": teacher_outputs,
            "student_globals": student_global_outputs,
            "student_locals": student_local_outputs,
            "reconstructions": tuple(reconstructions),
        }
