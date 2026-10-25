"""Validation in original- and low-resolution-query conditions."""

from __future__ import annotations

from torch.utils.data import DataLoader

from maskcluster_ssl.data.hdf5 import ReIDH5Dataset
from maskcluster_ssl.data.transforms import EvalTransform

from .extract import extract_embeddings
from .reid import evaluate_reid, summarize_conditions


def evaluate_validation(
    encoder,
    validation_path,
    *,
    device,
    input_size=(128, 64),
    batch_size=128,
    num_workers=0,
    query_block_size=128,
) -> dict:
    """Use one encoder checkpoint for both query resolutions and one gallery."""
    transform = EvalTransform(input_size)
    gallery = ReIDH5Dataset(validation_path, "gallery", transform=transform)
    query_original = ReIDH5Dataset(validation_path, "query", transform=transform)
    query_low = ReIDH5Dataset(validation_path, "query", low_resolution=True, transform=transform)

    def loader(dataset):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=str(device).startswith("cuda"),
        )

    try:
        gallery_embeddings, gallery_meta = extract_embeddings(encoder, loader(gallery), device)
        results = []
        for dataset in (query_original, query_low):
            query_embeddings, query_meta = extract_embeddings(encoder, loader(dataset), device)
            results.append(evaluate_reid(
                query_embeddings,
                gallery_embeddings,
                query_meta["person_id"],
                gallery_meta["person_id"],
                query_meta["camera_id"],
                gallery_meta["camera_id"],
                query_block_size=query_block_size,
            ))
        return summarize_conditions(*results)
    finally:
        gallery.close()
        query_original.close()
        query_low.close()
