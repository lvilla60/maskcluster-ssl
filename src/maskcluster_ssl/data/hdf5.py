"""Read-only HDF5 datasets with one lazy handle per process."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable

import h5py
import numpy as np


def _decode_id(value):
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", "backslashreplace")
    if isinstance(value, np.generic):
        return value.item()
    return value


class _LazyH5Dataset:
    """Map-style dataset whose HDF5 handle is never shared across processes."""

    def __init__(self, path: str | Path, image_key: str, transform: Callable | None = None):
        self.path = Path(path).resolve(strict=True)
        self.image_key = image_key.strip("/")
        self.transform = transform
        self._file: h5py.File | None = None
        self._pid: int | None = None
        with h5py.File(self.path, "r") as handle:
            if self.image_key not in handle:
                raise KeyError(f"Missing HDF5 dataset: {self.image_key}")
            dataset = handle[self.image_key]
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 4:
                raise ValueError(f"Expected an NCHW image dataset at {self.image_key}")
            self._length = int(dataset.shape[0])
            self.image_shape = tuple(int(v) for v in dataset.shape[1:])
            layout = dataset.attrs.get("layout", handle.attrs.get("layout", "NCHW"))
            if _decode_id(layout) != "NCHW":
                raise ValueError(f"Only NCHW arrays are supported, found {layout!r}")

    def _handle(self) -> h5py.File:
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            self.close()
            self._file = h5py.File(self.path, "r")
            self._pid = pid
        return self._file

    def _image(self, index: int) -> np.ndarray:
        image = np.asarray(self._handle()[self.image_key][index])
        if image.ndim != 3 or image.shape[0] not in (1, 3, 4):
            raise ValueError(f"Invalid CHW image at index {index}: {image.shape}")
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif np.issubdtype(image.dtype, np.floating):
            image = image.astype(np.float32, copy=False)
        else:
            image = image.astype(np.float32)
        return np.ascontiguousarray(image)

    def __len__(self) -> int:
        return self._length

    def close(self) -> None:
        file = getattr(self, "_file", None)
        if file is not None:
            file.close()
        self._file = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state

    def __del__(self):
        self.close()


class TrainingH5Dataset(_LazyH5Dataset):
    """Training images only; identifiers and camera metadata are inaccessible."""

    def __init__(self, path: str | Path, transform: Callable | None = None):
        super().__init__(path, "images", transform)

    def __getitem__(self, index: int):
        epoch = 0
        if isinstance(index, tuple):
            epoch, index = index
        image = self._image(index)
        return self.transform(image, index=index, epoch=epoch) if self.transform is not None else image


class ReIDH5Dataset(_LazyH5Dataset):
    """Validation/test query or gallery images and evaluation-only metadata."""

    _METADATA = ("person_ids", "camera_ids", "image_ids")

    def __init__(
        self,
        path: str | Path,
        group: str,
        *,
        low_resolution: bool = False,
        transform: Callable | None = None,
    ):
        group = group.strip("/")
        if group not in {"query", "gallery"}:
            raise ValueError("group must be 'query' or 'gallery'")
        if low_resolution and group != "query":
            raise ValueError("Only query has a low-resolution condition")
        name = "images_low_resolution" if low_resolution else "images"
        super().__init__(path, f"{group}/{name}", transform)
        self.group = group
        self.low_resolution = low_resolution
        with h5py.File(self.path, "r") as handle:
            for field in self._METADATA:
                key = f"{group}/{field}"
                if key not in handle or len(handle[key]) != self._length:
                    raise ValueError(f"Missing or misaligned metadata: {key}")

    def __getitem__(self, index: int):
        image = self._image(index)
        if self.transform is not None:
            image = self.transform(image)
        handle = self._handle()
        return {
            "image": image,
            "person_id": _decode_id(handle[f"{self.group}/person_ids"][index]),
            "camera_id": _decode_id(handle[f"{self.group}/camera_ids"][index]),
            "image_id": _decode_id(handle[f"{self.group}/image_ids"][index]),
            "index": int(index),
        }


def collect_metadata(dataset: ReIDH5Dataset, fields: Iterable[str] = ("person_id", "camera_id")):
    """Read evaluation metadata in dataset order without loading image arrays."""
    requested = tuple(fields)
    invalid = set(requested) - {"person_id", "camera_id", "image_id"}
    if invalid:
        raise ValueError(f"Unknown metadata fields: {sorted(invalid)}")
    handle = dataset._handle()
    plural = {"person_id": "person_ids", "camera_id": "camera_ids", "image_id": "image_ids"}
    return {
        field: np.asarray([_decode_id(v) for v in handle[f"{dataset.group}/{plural[field]}"]])
        for field in requested
    }
