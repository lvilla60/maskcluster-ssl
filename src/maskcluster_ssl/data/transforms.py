"""Image-only view generation and deterministic evaluation preprocessing."""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def stream_seed(seed: int, epoch: int, index: int, stream: str) -> int:
    """Derive a stable per-sample seed without touching global RNG state."""
    key = f"{seed}:{epoch}:{index}:{stream}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)


def sample_generator(seed: int, epoch: int, index: int, stream: str) -> torch.Generator:
    return torch.Generator().manual_seed(stream_seed(seed, epoch, index, stream))


def as_tensor(image: np.ndarray | Tensor) -> Tensor:
    tensor = torch.from_numpy(image) if isinstance(image, np.ndarray) else image
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW image, found {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.float32)


class EvalTransform:
    """Resize to a fixed rectangle with bicubic interpolation and normalize."""

    def __init__(self, size=(128, 64), mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.size = tuple(size)
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def __call__(self, image: np.ndarray | Tensor) -> Tensor:
        tensor = as_tensor(image).unsqueeze(0)
        tensor = F.interpolate(tensor, self.size, mode="bicubic", align_corners=False, antialias=True)[0]
        return (tensor - self.mean) / self.std


@dataclass(frozen=True)
class ViewBatch:
    global_views: tuple[Tensor, Tensor]
    local_views: tuple[Tensor, ...]
    patch_masks: tuple[Tensor, Tensor]


class MultiCropTransform:
    """Create two global crops, local crops, and aligned global patch masks."""

    def __init__(
        self,
        global_size=(128, 64),
        local_size=(96, 48),
        local_crops=4,
        global_scale=(0.65, 1.0),
        local_scale=(0.35, 0.65),
        patch_size=16,
        mask_ratio=0.5,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        seed=0,
        photometric=True,
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        grayscale_probability=0.05,
    ):
        self.global_size = tuple(global_size)
        self.local_size = tuple(local_size)
        self.local_crops = int(local_crops)
        self.global_scale = tuple(global_scale)
        self.local_scale = tuple(local_scale)
        self.patch_size = int(patch_size)
        self.mask_ratio = float(mask_ratio)
        self.seed = int(seed)
        self.photometric = bool(photometric)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.grayscale_probability = float(grayscale_probability)
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        for size in (self.global_size, self.local_size):
            if size[0] % self.patch_size or size[1] % self.patch_size:
                raise ValueError(f"Crop size {size} must be divisible by patch size {self.patch_size}")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between zero and one")

    def _crop(self, image: Tensor, target, scale, generator=None) -> Tensor:
        _, height, width = image.shape
        area = height * width
        target_ratio = target[1] / target[0]
        selected = None
        for _ in range(10):
            crop_area = area * (scale[0] + (scale[1] - scale[0]) * float(torch.rand((), generator=generator)))
            crop_h = max(1, min(height, int(round(math.sqrt(crop_area / target_ratio)))))
            crop_w = max(1, min(width, int(round(crop_h * target_ratio))))
            if crop_h <= height and crop_w <= width:
                top = int(torch.randint(0, height - crop_h + 1, (), generator=generator).item())
                left = int(torch.randint(0, width - crop_w + 1, (), generator=generator).item())
                selected = image[:, top:top + crop_h, left:left + crop_w]
                break
        if selected is None:
            selected = image
        selected = F.interpolate(
            selected.unsqueeze(0), target, mode="bicubic", align_corners=False, antialias=True
        )[0].clamp(0.0, 1.0)
        if torch.rand((), generator=generator) < 0.5:
            selected = selected.flip(-1)
        return selected

    def _mask(self, size, generator=None) -> Tensor:
        rows, cols = size[0] // self.patch_size, size[1] // self.patch_size
        count = max(1, min(rows * cols - 1, round(rows * cols * self.mask_ratio)))
        mask = torch.zeros(rows, cols, dtype=torch.bool)
        attempts = 0
        while int(mask.sum()) < count and attempts < rows * cols * 4:
            block_h = int(torch.randint(1, min(3, rows) + 1, (), generator=generator).item())
            block_w = int(torch.randint(1, min(2, cols) + 1, (), generator=generator).item())
            top = int(torch.randint(0, rows - block_h + 1, (), generator=generator).item())
            left = int(torch.randint(0, cols - block_w + 1, (), generator=generator).item())
            candidates = (~mask[top:top + block_h, left:left + block_w]).nonzero()
            remaining = count - int(mask.sum())
            for candidate in candidates[:remaining]:
                mask[top + int(candidate[0]), left + int(candidate[1])] = True
            attempts += 1
        if int(mask.sum()) < count:
            available = (~mask).flatten().nonzero().flatten()
            mask.flatten()[available[:count - int(mask.sum())]] = True
        return mask

    def _normalize(self, image: Tensor) -> Tensor:
        return (image - self.mean) / self.std

    def _photometric(self, image: Tensor, generator: torch.Generator) -> Tensor:
        if not self.photometric:
            return image
        def factor(amount):
            return 1.0 + (2.0 * float(torch.rand((), generator=generator)) - 1.0) * amount
        image = image * factor(self.brightness)
        image = (image - image.mean(dim=(-2, -1), keepdim=True)) * factor(self.contrast) + image.mean(dim=(-2, -1), keepdim=True)
        gray = image.mean(dim=0, keepdim=True)
        image = gray + (image - gray) * factor(self.saturation)
        if torch.rand((), generator=generator) < self.grayscale_probability:
            image = gray.expand_as(image)
        return image.clamp(0.0, 1.0)

    def __call__(self, image: np.ndarray | Tensor, *, index=0, epoch=0) -> dict[str, tuple[Tensor, ...]]:
        image = as_tensor(image)
        geometry = sample_generator(self.seed, epoch, index, "geometry")
        masks = sample_generator(self.seed, epoch, index, "masks")
        globals_raw = tuple(self._crop(image, self.global_size, self.global_scale, geometry) for _ in range(2))
        locals_raw = tuple(self._crop(image, self.local_size, self.local_scale, geometry) for _ in range(self.local_crops))
        globals_raw = tuple(self._photometric(view, sample_generator(self.seed, epoch, index, f"global_photo_{i}")) for i, view in enumerate(globals_raw))
        locals_raw = tuple(self._photometric(view, sample_generator(self.seed, epoch, index, f"local_photo_{i}")) for i, view in enumerate(locals_raw))
        return {
            "global_views": tuple(self._normalize(view) for view in globals_raw),
            "local_views": tuple(self._normalize(view) for view in locals_raw),
            "global_targets": globals_raw,
            "patch_masks": tuple(self._mask(self.global_size, masks) for _ in range(2)),
        }


def corrupt_masked_patches(
    images: Tensor,
    patch_masks: Tensor,
    patch_size: int,
    *,
    zero_probability: float = 1 / 3,
    noise_probability: float = 1 / 3,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply GMML-style zero, noise, and batch-alien patch corruption."""
    if images.ndim != 4 or patch_masks.ndim != 3:
        raise ValueError("Expected images BCHW and patch masks BGHxGW")
    batch, _, height, width = images.shape
    rows, cols = patch_masks.shape[-2:]
    if (height, width) != (rows * patch_size, cols * patch_size):
        raise ValueError("Patch masks do not align with image geometry")
    output = images.clone()
    alien = images.roll(1, dims=0) if batch > 1 else images.flip(-1)
    for row in range(rows):
        for col in range(cols):
            active = patch_masks[:, row, col]
            if not active.any():
                continue
            y, x = row * patch_size, col * patch_size
            choice = torch.rand(batch, generator=generator, device="cpu").to(images.device)
            zero = active & (choice < zero_probability)
            noise = active & (choice >= zero_probability) & (choice < zero_probability + noise_probability)
            other = active & ~(zero | noise)
            block = output[:, :, y:y + patch_size, x:x + patch_size]
            block[zero] = 0.0
            block[noise] = torch.rand(block[noise].shape, generator=generator, device="cpu").to(images.device)
            block[other] = alien[other, :, y:y + patch_size, x:x + patch_size]
    return output


def normalize_images(images: Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> Tensor:
    """Normalize a BCHW tensor after pixel-space masking."""
    mean_tensor = images.new_tensor(mean)[None, :, None, None]
    std_tensor = images.new_tensor(std)[None, :, None, None]
    return (images - mean_tensor) / std_tensor
