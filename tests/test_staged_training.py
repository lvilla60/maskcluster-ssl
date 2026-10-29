"""Bounded synthetic stage, overflow and provenance regressions."""
import copy
import json
import random
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from maskcluster_ssl.data.sampling import EpochSampler
from maskcluster_ssl.data.transforms import MultiCropTransform
from maskcluster_ssl.evaluation.analysis import require_completed_budget
from maskcluster_ssl.losses import MaskClusterLoss
from maskcluster_ssl.models import MaskClusterSSL
from maskcluster_ssl.training.checkpoint import validate_resume_config
from maskcluster_ssl.training.runtime import seed_everything, git_metadata, direct_pins_metadata
from maskcluster_ssl.training.trainer import Trainer


def folder():
    path = Path(__file__).resolve().parents[1] / ".tmp" / ("stages_synthetic_" + uuid.uuid4().hex)
    path.mkdir(parents=True)
    return path


class SyntheticImages:
    def __init__(self):
        self.order = []
        self.transform = MultiCropTransform(global_size=(32, 16), local_size=(16, 8),
                                             local_crops=1, patch_size=8, seed=17)

    def __len__(self):
        return 4

    def __getitem__(self, key):
        epoch, index = key
        rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
        self.order.append((epoch, index, rng))
        image = torch.full((3, 32, 16), (index + sum(rng)) / 7)
        return self.transform(image, index=index, epoch=epoch)


def trainer(device="cpu"):
    seed_everything(17)
    dataset = SyntheticImages()
    loader = DataLoader(dataset, batch_size=2, sampler=EpochSampler(dataset, 17), num_workers=0)
    model = MaskClusterSSL(variant="full", image_size=(32, 16), patch_size=8,
                           embed_dim=16, depth=2, num_heads=2, mlp_ratio=2,
                           prototypes=8, reconstruction_layers=(1, 2))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    config = {"seed": 17, "runtime": {"device": device, "amp": device == "cuda"},
              "training": {"epochs": 2, "warmup_epochs": 0, "learning_rate": 1e-4,
                           "minimum_learning_rate": 1e-6, "ema_momentum": 0.9,
                           "ema_momentum_final": 0.99, "gradient_clip": 3},
              "evaluation": {"every_epochs": 1}}
    result = Trainer(model, MaskClusterLoss(patch_size=8), optimizer, loader, config, folder(), device)
    result.scaler = torch.amp.GradScaler(device, enabled=True, init_scale=128)
    # Fixed validation keeps the first best checkpoint, exercising carry-over.
    result.validate = lambda: {"teacher": {"mean_mAP": 0.3}, "student": {"mean_mAP": 0.2}}
    return result


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        for left, right in zip(a, b):
            equal_state(left, right)
    else:
        assert a == b


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA staged training requires target GPU"))])
def test_staged_matches_continuous(device):
    continuous = trainer(device)
    final = continuous.fit()
    full_rng = (random.random(), float(np.random.rand()), torch.rand(3))
    first = trainer(device)
    paused = first.fit(max_epochs_per_invocation=1)
    assert paused["status"] == "paused" and paused["completed_epochs"] == 1
    assert paused["planned_epochs"] == 2 and (first.run_dir / "last.pt").exists()
    with pytest.raises(ValueError, match="completed common"):
        require_completed_budget({"config": first.config, "run_summary": paused})
    changed = copy.deepcopy(first.config)
    changed["training"]["epochs"] = 3
    with pytest.raises(ValueError, match="training.epochs"):
        validate_resume_config(first.config, changed)
    resumed = trainer(device)
    resumed.resume(first.run_dir / "last.pt")
    assert resumed.start_epoch == 1
    result = resumed.fit(max_epochs_per_invocation=10)
    require_completed_budget({"config": resumed.config, "run_summary": result})
    assert result["status"] == "completed" and result["epochs_this_invocation"] == 1
    assert result["global_step"] == final["global_step"] == 4
    equal_state(continuous.model.state_dict(), resumed.model.state_dict())
    equal_state(continuous.optimizer.state_dict(), resumed.optimizer.state_dict())
    equal_state(continuous.scaler.state_dict(), resumed.scaler.state_dict())
    equal_state(full_rng, (random.random(), float(np.random.rand()), torch.rand(3)))
    assert continuous.train_loader.dataset.order == first.train_loader.dataset.order + resumed.train_loader.dataset.order
    for name in ("teacher", "student"):
        assert (resumed.run_dir / f"best_{name}.pt").read_bytes() == (first.run_dir / f"best_{name}.pt").read_bytes()
    def records(run):
        return [json.loads(line) for line in (run.run_dir / "history.jsonl").read_text().splitlines()]
    left = [r for r in records(continuous) if "learning_rate" in r]
    right = [r for r in records(resumed) if "learning_rate" in r]
    for a, b in zip(left, right):
        for key in ("step", "learning_rate", "ema_momentum", "loss/total", "samples", "optimizer_step_executed"):
            assert a[key] == b[key]
    epochs = [r for r in records(resumed) if r.get("event") == "epoch"]
    assert [r["epoch"] for r in epochs] == [0, 1]
    for r in epochs:
        assert r["steps"] == 2 and r["samples"] == 4
        assert r["train_seconds"] > 0 and r["validation_seconds"] >= 0
        assert r["train_images_per_second"] == pytest.approx(4 / r["train_seconds"])
        if device == "cpu":
            assert r["cuda_peak_allocated_bytes"] is None
        else:
            assert 0 < r["cuda_peak_allocated_bytes"] <= r["cuda_peak_reserved_bytes"]


def test_nonfinite_gradient_fails_before_optimizer_ema_and_counter():
    run = trainer()
    before = copy.deepcopy(run.model.state_dict())
    handle = next(run.model.student.parameters()).register_hook(lambda gradient: gradient * float("inf"))
    with pytest.raises(RuntimeError, match="non-finite"):
        run.train_epoch(0, 4)
    handle.remove()
    equal_state(before, run.model.state_dict())
    assert run.global_step == 0 and not run.optimizer.state
    record = json.loads((run.run_dir / "history.jsonl").read_text().splitlines()[-1])
    assert record["event"] == "gradient_check_failed" and not record["optimizer_step_executed"]


def test_simulated_scaler_skip_never_updates_ema_or_counter():
    run = trainer()
    before = copy.deepcopy(run.model.state_dict())
    run.scaler = SimpleNamespace(scale=lambda loss: loss, unscale_=lambda opt: None,
                                 step=lambda opt: None, update=lambda: None)
    with pytest.raises(RuntimeError, match="skipped optimizer"):
        run.train_epoch(0, 4)
    equal_state(before, run.model.state_dict())
    assert run.global_step == 0 and not run.optimizer.state


def test_platform_pins_and_git_provenance(monkeypatch):
    path = folder()
    (path / "requirements-lock-windows-cpu.txt").write_text("reference")
    assert direct_pins_metadata(path, "Linux", "2.11.0+cu128")["direct_pins_file"] is None
    assert direct_pins_metadata(path, "Windows", "2.14.1+cpu")["direct_pins_file"]
    def fake_run(args, **kwargs):
        text = str(path) if "--show-toplevel" in args else (" M src/x.py" if "status" in args else "abc123")
        return SimpleNamespace(returncode=0, stdout=text)
    monkeypatch.setattr("maskcluster_ssl.training.runtime.subprocess.run", fake_run)
    assert git_metadata(path) == {"git_commit": "abc123", "git_dirty": True, "git_status": "available"}
    monkeypatch.setattr("maskcluster_ssl.training.runtime.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    assert git_metadata(path)["git_commit"] is None
