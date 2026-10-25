"""Clustering, ME-MAX, and masked reconstruction objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def distillation_cross_entropy(student: Tensor, teacher: Tensor, student_temperature: float, teacher_temperature: float) -> Tensor:
    target = F.softmax(teacher.detach().float() / teacher_temperature, dim=-1)
    prediction = F.log_softmax(student.float() / student_temperature, dim=-1)
    return -(target * prediction).sum(dim=-1).mean()


def me_max(logits: list[Tensor], temperature: float) -> tuple[Tensor, Tensor]:
    probabilities = torch.cat([
        F.softmax(value.float().reshape(-1, value.shape[-1]) / temperature, dim=-1)
        for value in logits
    ], dim=0)
    mean_probability = probabilities.mean(dim=0)
    entropy = -(mean_probability * mean_probability.clamp_min(1e-8).log()).sum()
    return -entropy, entropy.detach()


def global_me_max(
    global_logits: list[Tensor], local_logits: list[Tensor], temperature: float
) -> tuple[Tensor, Tensor]:
    """Implement Eq. A5, where every local crop is paired with both globals."""
    if len(global_logits) != 2:
        raise ValueError("Global ME-MAX requires exactly two global views")
    paper_weighted = list(global_logits)
    for _ in global_logits:
        paper_weighted.extend(local_logits)
    return me_max(paper_weighted, temperature)


def masked_l1(reconstruction: Tensor, target: Tensor, patch_mask: Tensor, patch_size: int) -> Tensor:
    pixel_mask = patch_mask.repeat_interleave(patch_size, -2).repeat_interleave(patch_size, -1)
    pixel_mask = pixel_mask[:, None].to(dtype=torch.float32)
    denominator = pixel_mask.sum().clamp_min(1.0) * reconstruction.shape[1]
    return ((reconstruction.float() - target.float()).abs() * pixel_mask).sum() / denominator


@dataclass(frozen=True)
class LossWeights:
    global_clustering: float = 1.0
    patch_clustering: float = 1.0
    reconstruction: float = 1.0
    global_me_max: float = 1.0
    patch_me_max: float = 1.0


class MaskClusterLoss(nn.Module):
    def __init__(
        self,
        *,
        student_temperature=0.1,
        teacher_temperature=0.04,
        me_max_temperature=0.1,
        patch_size=16,
        weights=LossWeights(),
    ):
        super().__init__()
        self.student_temperature = float(student_temperature)
        self.teacher_temperature = float(teacher_temperature)
        self.me_max_temperature = float(me_max_temperature)
        self.patch_size = int(patch_size)
        self.weights = weights

    def forward(self, outputs: dict, targets, patch_masks) -> tuple[Tensor, dict[str, Tensor]]:
        teacher = outputs["teacher_globals"]
        student_global = outputs["student_globals"]
        student_local = outputs["student_locals"]
        if len(teacher) != 2 or len(student_global) != 2:
            raise ValueError("MaskCluster requires exactly two global views")

        global_terms = [
            distillation_cross_entropy(student_global[1]["global_logits"], teacher[0]["global_logits"], self.student_temperature, self.teacher_temperature),
            distillation_cross_entropy(student_global[0]["global_logits"], teacher[1]["global_logits"], self.student_temperature, self.teacher_temperature),
        ]
        for local in student_local:
            for teacher_view in teacher:
                global_terms.append(distillation_cross_entropy(
                    local["global_logits"], teacher_view["global_logits"],
                    self.student_temperature, self.teacher_temperature,
                ))
        global_clustering = torch.stack(global_terms).mean()
        global_me, global_entropy = global_me_max(
            [view["global_logits"] for view in student_global],
            [view["global_logits"] for view in student_local],
            self.me_max_temperature,
        )

        zero = global_clustering.new_zeros(())
        patch_clustering = zero
        patch_me = zero
        patch_entropy = zero.detach()
        if student_global[0]["patch_logits"] is not None:
            patch_terms = [
                distillation_cross_entropy(student_global[index]["patch_logits"], teacher[index]["patch_logits"], self.student_temperature, self.teacher_temperature)
                for index in range(2)
            ]
            patch_clustering = torch.stack(patch_terms).mean()
            patch_me, patch_entropy = me_max(
                [view["patch_logits"] for view in student_global], self.me_max_temperature
            )

        reconstruction = zero
        if outputs["reconstructions"]:
            reconstruction = torch.stack([
                masked_l1(predicted, target, mask, self.patch_size)
                for predicted, target, mask in zip(outputs["reconstructions"], targets, patch_masks)
            ]).mean()

        weighted = {
            "global_clustering": global_clustering * self.weights.global_clustering,
            "patch_clustering": patch_clustering * self.weights.patch_clustering,
            "reconstruction": reconstruction * self.weights.reconstruction,
            "global_me_max": global_me * self.weights.global_me_max,
            "patch_me_max": patch_me * self.weights.patch_me_max,
        }
        total = sum(weighted.values())
        student_features = torch.cat([view["features"] for view in student_global], dim=0)
        diagnostics = {
            **{f"loss/{name}": value.detach() for name, value in weighted.items()},
            "loss/total": total.detach(),
            "diagnostic/global_assignment_entropy": global_entropy,
            "diagnostic/patch_assignment_entropy": patch_entropy,
            "diagnostic/embedding_std": student_features.float().std(dim=0).mean().detach(),
        }
        return total, diagnostics
