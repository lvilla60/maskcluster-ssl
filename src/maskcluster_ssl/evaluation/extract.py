"""Batched embedding extraction for student or EMA teacher encoders."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


@torch.inference_mode()
def extract_embeddings(model, loader, device: torch.device | str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Extract L2-normalized CLS embeddings and aligned evaluation metadata."""
    model.eval()
    embeddings: list[torch.Tensor] = []
    metadata = {"person_id": [], "camera_id": [], "image_id": [], "index": []}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        features = model.encode(images)
        embeddings.append(F.normalize(features.float(), dim=-1).cpu())
        for key in metadata:
            values = batch[key]
            if isinstance(values, torch.Tensor):
                values = values.cpu().tolist()
            metadata[key].extend(list(values))
    if not embeddings:
        raise ValueError("Embedding loader produced no batches")
    order = np.argsort(np.asarray(metadata["index"], dtype=np.int64), kind="stable")
    matrix = torch.cat(embeddings).numpy()[order]
    aligned = {key: np.asarray(values, dtype=object)[order] for key, values in metadata.items()}
    return matrix, aligned
