"""Regression checks for checkpoint portability and validation selection."""

import copy
import uuid
from pathlib import Path

import pytest
import torch

from maskcluster_ssl.data.provenance import file_signature, training_signatures
from maskcluster_ssl.models import MaskClusterSSL
from maskcluster_ssl.training.checkpoint import load_checkpoint, validate_resume_config
from maskcluster_ssl.training.trainer import Trainer


def _model():
    return MaskClusterSSL(variant="baseline", image_size=(32, 16), patch_size=8,
                          embed_dim=32, depth=2, num_heads=4, mlp_ratio=2,
                          prototypes=16, reconstruction_layers=(1, 2))


def _trainer(run_dir, config, data_signatures=None):
    network = _model()
    optimizer = torch.optim.AdamW((p for p in network.parameters() if p.requires_grad), lr=1e-3)
    return Trainer(network, torch.nn.Identity(), optimizer, [None], config, run_dir, "cpu",
                   data_signatures=data_signatures)


def _run_dir():
    folder = Path(__file__).resolve().parents[1] / ".tmp" / f"closure_{uuid.uuid4().hex}"
    folder.mkdir(parents=True)
    return folder


def test_trainer_cpu_checkpoint_initializes_enabled_scaler_and_rejects_corruption():
    folder = _run_dir()
    config = {"runtime": {"amp": True, "device": "cpu"}, "training": {"epochs": 1},
              "evaluation": {"every_epochs": 1}}
    trainer = _trainer(folder, config)
    trainer.train_epoch = lambda epoch, total_steps: {}
    trainer.validate = lambda: {"teacher": {"mean_mAP": 0.6},
                                "student": {"mean_mAP": 0.5}}
    trainer.fit()  # The real Trainer writes the checkpoint; no training step runs.
    path = folder / "last.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["scaler"] == {}

    enabled = torch.amp.GradScaler("cpu", enabled=True)
    state = load_checkpoint(path, model=_model(), scaler=enabled, config=config)
    assert state["scaler_resume"] == "initialized_no_amp_state"
    assert enabled.is_enabled() and enabled.state_dict()["scale"] > 0
    resume_dir = folder / "resumed"
    resume_dir.mkdir()
    resumed = _trainer(resume_dir, config)
    resumed.scaler = torch.amp.GradScaler("cpu", enabled=True)
    resumed.resume(path)
    assert '"scaler": "initialized_no_amp_state"' in (resume_dir / "history.jsonl").read_text()

    absent = copy.deepcopy(saved)
    absent["scaler"] = None
    absent_path = folder / "no_scaler.pt"
    torch.save(absent, absent_path)
    assert load_checkpoint(absent_path, model=_model(), scaler=enabled, config=config)[
        "scaler_resume"] == "initialized_no_amp_state"

    valid = copy.deepcopy(saved)
    valid["scaler"] = torch.amp.GradScaler("cpu", enabled=True).state_dict()
    valid_path = folder / "valid_scaler.pt"
    torch.save(valid, valid_path)
    restored = torch.amp.GradScaler("cpu", enabled=True, init_scale=128)
    state = load_checkpoint(valid_path, model=_model(), scaler=restored, config=config)
    assert state["scaler_resume"] == "restored"
    assert restored.state_dict()["scale"] == valid["scaler"]["scale"]

    corrupt = copy.deepcopy(saved)
    corrupt["scaler"] = {"scale": 100}
    corrupt_path = folder / "corrupt_scaler.pt"
    torch.save(corrupt, corrupt_path)
    with pytest.raises(ValueError, match="Incomplete AMP scaler"):
        load_checkpoint(corrupt_path, model=_model(), scaler=enabled, config=config)


def test_simultaneous_best_checkpoints_contain_both_current_scores():
    folder = _run_dir()
    config = {"runtime": {"amp": False, "device": "cpu"}, "training": {"epochs": 1},
              "evaluation": {"every_epochs": 1}}
    trainer = _trainer(folder, config)
    trainer.train_epoch = lambda epoch, total_steps: {}
    trainer.validate = lambda: {"teacher": {"mean_mAP": 0.7},
                                "student": {"mean_mAP": 0.8}}
    trainer.fit()
    for encoder in ("teacher", "student"):
        state = load_checkpoint(folder / f"best_{encoder}.pt", model=_model(), config=config)
        assert state["best_scores"] == {"teacher": 0.7, "student": 0.8}
        assert state["epoch"] == 0


def test_relocation_requires_exact_data_bytes_and_stable_science():
    folder = _run_dir()
    original = folder / "windows_training.h5"
    moved = folder / "linux_training.h5"
    validation = folder / "validation.h5"
    original.write_bytes(b"training fixture")
    moved.write_bytes(original.read_bytes())
    validation.write_bytes(b"validation fixture")
    saved = {"data": {"training": str(original), "validation": str(validation)},
             "model": {"depth": 12}, "training": {"epochs": 20},
             "runtime": {"device": "cpu", "amp": False}}
    current = copy.deepcopy(saved)
    current["data"]["training"] = str(moved)
    current["runtime"].update(device="cuda", amp=True)
    signatures = training_signatures(original, validation)
    assert file_signature(original) == file_signature(moved)
    validate_resume_config(saved, current, saved_data_signatures=signatures,
                           current_data_signatures=training_signatures(moved, validation),
                           allow_cpu_to_cuda_amp=True)
    with pytest.raises(ValueError, match="runtime.amp"):
        validate_resume_config(saved, current, saved_data_signatures=signatures,
                               current_data_signatures=training_signatures(moved, validation))
    moved.write_bytes(b"different training fixture")
    with pytest.raises(ValueError, match="training data signature"):
        validate_resume_config(saved, current, saved_data_signatures=signatures,
                               current_data_signatures=training_signatures(moved, validation),
                               allow_cpu_to_cuda_amp=True)
    moved.write_bytes(original.read_bytes())
    changed = copy.deepcopy(current)
    changed["model"]["depth"] = 13
    with pytest.raises(ValueError, match="model.depth"):
        validate_resume_config(saved, changed, saved_data_signatures=signatures,
                               current_data_signatures=training_signatures(moved, validation),
                               allow_cpu_to_cuda_amp=True)
    with pytest.raises(ValueError, match="no data signatures"):
        validate_resume_config(saved, current, current_data_signatures=signatures)

    # Exercise the saved checkpoint and Trainer.resume path without an optimizer step.
    train_config = copy.deepcopy(saved)
    train_config["training"]["epochs"] = 1
    train_config["evaluation"] = {"every_epochs": 1}
    first = _trainer(folder, train_config, signatures)
    first.train_epoch = lambda epoch, total_steps: {}
    first.validate = lambda: {"teacher": {"mean_mAP": 0.2},
                             "student": {"mean_mAP": 0.3}}
    first.fit()
    moved_config = copy.deepcopy(train_config)
    moved_config["data"]["training"] = str(moved)
    resumed = _trainer(folder, moved_config, training_signatures(moved, validation))
    resumed.resume(folder / "last.pt")
    assert resumed.start_epoch == 1
    moved.write_bytes(b"changed after checkpoint")
    rejected = _trainer(folder, moved_config, training_signatures(moved, validation))
    with pytest.raises(ValueError, match="training data signature"):
        rejected.resume(folder / "last.pt")
