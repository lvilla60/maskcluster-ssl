"""Reproducibility, device selection, and run metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch build")
    return torch.device(requested)


def cosine_value(start: float, end: float, step: int, total_steps: int, warmup_steps: int = 0) -> float:
    if warmup_steps and step < warmup_steps:
        return start * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, progress))
    return end + 0.5 * (start - end) * (1.0 + np.cos(np.pi * progress))


def write_run_metadata(run_dir: Path, config: dict, source_root: Path,
                       data_signatures=None) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    digest = hashlib.sha256()
    paths = list((source_root / "src" / "maskcluster_ssl").rglob("*.py"))
    paths += list((source_root / "scripts").glob("*.py"))
    paths += list((source_root / "configs").glob("*.yaml"))
    paths += [source_root / "pyproject.toml", source_root / "requirements-lock-windows-cpu.txt"]
    for path in sorted(paths):
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(path.read_bytes())
    packages = {}
    for name in ("torch", "torchvision", "numpy", "h5py", "Pillow", "PyYAML", "tqdm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    metadata = {
        "seed": config["seed"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device_request": config["runtime"]["device"],
        "packages": packages,
        "environment_inventory": sorted(
            f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions()
            if dist.metadata.get("Name")
        ),
        "direct_pins_file": "requirements-lock-windows-cpu.txt",
        "source_sha256": digest.hexdigest(),
        "git_commit": None,
        "data_signatures": data_signatures,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
