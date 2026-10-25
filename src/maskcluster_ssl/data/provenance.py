"""Portable content signatures for training and retrieval datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def file_signature(path: str | Path) -> dict[str, str | int]:
    """Hash file bytes so relocated, identical HDF5 files have the same identity."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def training_signatures(training_path: str | Path, validation_path: str | Path) -> dict:
    return {"training": file_signature(training_path),
            "validation": file_signature(validation_path)}


def _gallery_signature(group: h5py.Group) -> str:
    digest = hashlib.sha256()
    for name in ("images", "person_ids", "camera_ids", "image_ids"):
        dataset = group[name]
        digest.update(json.dumps({"name": name, "shape": dataset.shape,
                                  "dtype": str(dataset.dtype)}, sort_keys=True).encode())
        for start in range(0, len(dataset), 64):
            values = np.asarray(dataset[start:start + 64])
            if values.dtype.hasobject:
                for value in values.flat:
                    encoded = value if isinstance(value, bytes) else str(value).encode("utf-8")
                    digest.update(len(encoded).to_bytes(8, "little"))
                    digest.update(encoded)
            else:
                digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def evaluation_provenance(path: str | Path) -> dict:
    """Describe the actual query/gallery groups and their source file."""
    path = Path(path).resolve(strict=True)
    groups = {}
    with h5py.File(path, "r") as handle:
        for group_name, image_names in (("query", ("images", "images_low_resolution")),
                                        ("gallery", ("images",))):
            group = handle[group_name]
            groups[group_name] = {
                name: {"count": len(group[name]), "shape": list(group[name].shape),
                       "dtype": str(group[name].dtype)}
                for name in (*image_names, "person_ids", "camera_ids", "image_ids")
            }
        gallery_sha256 = _gallery_signature(handle["gallery"])
    return {"path": str(path), "file": file_signature(path),
            "groups": groups, "gallery_sha256": gallery_sha256}
