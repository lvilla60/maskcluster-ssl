import uuid
import multiprocessing
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from maskcluster_ssl.data.hdf5 import TrainingH5Dataset
from maskcluster_ssl.data.transforms import MultiCropTransform
from maskcluster_ssl.losses.maskcluster import masked_l1
from maskcluster_ssl.training.checkpoint import load_checkpoint, save_checkpoint


def test_training_loader_persistent_workers_or_documented_permission_block():
    try:
        reader, writer = multiprocessing.Pipe()
        reader.close()
        writer.close()
    except PermissionError as exc:
        pytest.skip(f"Worker IPC blocked before DataLoader creation: {exc}")
    root = Path(__file__).resolve().parents[1] / ".tmp"
    root.mkdir(exist_ok=True)
    path = root / f"workers_{uuid.uuid4().hex}.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((4, 3, 32, 16), dtype=np.uint8))
    dataset = TrainingH5Dataset(path, transform=MultiCropTransform(
        global_size=(32, 16), local_size=(16, 8), local_crops=1, patch_size=8, seed=4))
    try:
        loader = DataLoader(dataset, batch_size=2, num_workers=2, persistent_workers=True)
        try:
            assert next(iter(loader))["global_views"][0].shape == (2, 3, 32, 16)
        except PermissionError as exc:
            pytest.skip(f"Worker process creation blocked by permissions: {exc}")
        finally:
            del loader
    finally:
        dataset.close()
        path.unlink()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA build/device unavailable here")
def test_cpu_to_cuda_amp_checkpoint_and_cuda_continuation():
    root = Path(__file__).resolve().parents[1] / ".tmp"
    root.mkdir(exist_ok=True)
    path = root / f"cuda_{uuid.uuid4().hex}.pt"
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = {"runtime": {"device": "cpu"}, "training": {"epochs": 2}}
    x = torch.ones(2, 4)
    optimizer.zero_grad()
    model(x).square().mean().backward()
    optimizer.step()
    save_checkpoint(path, model=model, optimizer=optimizer, scaler=None, epoch=0,
                    global_step=1, config=config, best_scores={})
    cuda_model = torch.nn.Linear(4, 4)
    cuda_optimizer = torch.optim.AdamW(cuda_model.parameters(), lr=1e-3)
    load_checkpoint(path, model=cuda_model, optimizer=cuda_optimizer, map_location="cuda",
                    config={"runtime": {"device": "cuda"}, "training": {"epochs": 2}})
    assert torch.get_rng_state().device.type == "cpu"
    assert all(value["exp_avg"].device.type == "cuda" for value in cuda_optimizer.state.values())
    scaler = torch.amp.GradScaler("cuda")
    prediction = torch.full((8, 3, 128, 64), 0.5, device="cuda", dtype=torch.float16, requires_grad=True)
    with torch.autocast("cuda"):
        reconstruction_loss = masked_l1(prediction, torch.zeros_like(prediction),
                                        torch.ones((8, 8, 4), device="cuda", dtype=torch.bool), 16)
    assert reconstruction_loss.dtype == torch.float32 and reconstruction_loss.item() == pytest.approx(0.5)
    reconstruction_loss.backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0
    def step(network, opt, amp):
        opt.zero_grad()
        with torch.autocast("cuda"):
            loss = network(x.cuda()).square().mean()
        amp.scale(loss).backward()
        amp.step(opt)
        amp.update()
    save_checkpoint(path, model=cuda_model, optimizer=cuda_optimizer, scaler=scaler,
                    epoch=0, global_step=1, config=config, best_scores={})
    step(cuda_model, cuda_optimizer, scaler)
    expected = {key: value.clone() for key, value in cuda_model.state_dict().items()}
    resumed = torch.nn.Linear(4, 4)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_scaler = torch.amp.GradScaler("cuda")
    load_checkpoint(path, model=resumed, optimizer=resumed_optimizer, scaler=resumed_scaler,
                    map_location="cuda", config={"runtime": {"device": "cuda"}, "training": {"epochs": 2}})
    step(resumed, resumed_optimizer, resumed_scaler)
    assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in expected.items())
    path.unlink()
