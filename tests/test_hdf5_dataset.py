import pickle
import uuid
from pathlib import Path

import h5py
import numpy as np
import pytest

from maskcluster_ssl.data.hdf5 import ReIDH5Dataset, TrainingH5Dataset


@pytest.fixture
def h5_files():
    root = Path(__file__).resolve().parents[1] / ".tmp" / ("loader_test_" + uuid.uuid4().hex)
    root.mkdir(parents=True)
    training = root / "training.h5"
    validation = root / "validation.h5"
    with h5py.File(training, "w") as handle:
        handle.attrs["layout"] = "NCHW"
        handle.create_dataset("images", data=np.arange(2 * 3 * 8 * 4, dtype=np.uint8).reshape(2, 3, 8, 4))
        handle.create_dataset("person_ids", data=[99, 100])
    with h5py.File(validation, "w") as handle:
        handle.attrs["layout"] = "NCHW"
        for group_name, length in (("query", 2), ("gallery", 3)):
            group = handle.create_group(group_name)
            group.create_dataset("images", data=np.ones((length, 3, 8, 4), dtype=np.uint8))
            if group_name == "query":
                group.create_dataset("images_low_resolution", data=np.ones((length, 3, 2, 1), dtype=np.uint8))
            group.create_dataset("person_ids", data=np.arange(length))
            group.create_dataset("camera_ids", data=np.arange(length) + 1)
            group.create_dataset("image_ids", data=np.asarray([f"im-{i}".encode() for i in range(length)]))
    return training, validation


def test_training_returns_images_only_and_pickle_drops_handle(h5_files):
    training, _ = h5_files
    dataset = TrainingH5Dataset(training)
    image = dataset[0]
    assert isinstance(image, np.ndarray)
    assert image.shape == (3, 8, 4)
    assert image.dtype == np.float32
    assert 0.0 <= image.min() <= image.max() <= 1.0
    state = pickle.loads(pickle.dumps(dataset))
    assert state._file is None
    np.testing.assert_allclose(state[1], dataset[1])


def test_evaluation_metadata_and_low_resolution_order(h5_files):
    _, validation = h5_files
    original = ReIDH5Dataset(validation, "query")
    low = ReIDH5Dataset(validation, "query", low_resolution=True)
    assert original[1]["image"].shape == (3, 8, 4)
    assert low[1]["image"].shape == (3, 2, 1)
    assert low[1]["person_id"] == original[1]["person_id"] == 1
    assert low[1]["image_id"] == original[1]["image_id"] == "im-1"


def test_invalid_group_and_gallery_low_resolution(h5_files):
    _, validation = h5_files
    with pytest.raises(ValueError, match="group"):
        ReIDH5Dataset(validation, "training")
    with pytest.raises(ValueError, match="Only query"):
        ReIDH5Dataset(validation, "gallery", low_resolution=True)
