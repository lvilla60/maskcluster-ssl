"""Atomic, resumable checkpoint persistence."""

from __future__ import annotations

from pathlib import Path
import random
import math

import numpy as np
import torch


ALLOWED_RESUME_CHANGES = {
    "output_dir", "runtime.device", "runtime.num_workers",
    "evaluation.batch_size", "evaluation.query_block_size",
}


def validate_resume_config(
    saved: dict, current: dict, *, saved_data_signatures=None, current_data_signatures=None,
    allow_cpu_to_cuda_amp=False,
) -> None:
    """Allow relocated data only when both content signatures match."""
    def flatten(value, prefix=""):
        if not isinstance(value, dict):
            return {prefix: value}
        result = {}
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
        return result
    old, new = flatten(saved), flatten(current)
    relocated = set()
    if current_data_signatures is not None:
        if saved_data_signatures is None:
            raise ValueError("Checkpoint has no data signatures; training data cannot be verified")
        if set(saved_data_signatures) != {"training", "validation"} or set(current_data_signatures) != {"training", "validation"}:
            raise ValueError("Training and validation data signatures are required")
        for key in ("training", "validation"):
            if saved_data_signatures[key] != current_data_signatures[key]:
                raise ValueError(f"Incompatible {key} data signature")
            relocated.add(f"data.{key}")
    if allow_cpu_to_cuda_amp and old.get("runtime.amp") is False and new.get("runtime.amp") is True:
        relocated.add("runtime.amp")
    changed = [key for key in sorted(old.keys() | new.keys())
               if old.get(key, object()) != new.get(key, object())
               and key not in ALLOWED_RESUME_CHANGES | relocated]
    if changed:
        raise ValueError(f"Incompatible checkpoint configuration: {', '.join(changed)}")


def _optimizer_to_device(optimizer):
    for parameter, state in optimizer.state.items():
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(parameter.device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, list):
                return [move(item) for item in value]
            return value
        optimizer.state[parameter] = {
            key: value.cpu() if key == "step" and isinstance(value, torch.Tensor) else move(value)
            for key, value in state.items()
        }


def save_checkpoint(path, *, model, optimizer, scaler, epoch, global_step, config, best_scores,
                    generators=None, data_signatures=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": config,
        "best_scores": dict(best_scores),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "generator_states": {key: value.get_state() for key, value in (generators or {}).items()},
        "data_signatures": data_signatures,
        "checkpoint_device": next(model.parameters()).device.type,
    }
    torch.save(state, temporary)
    temporary.replace(path)


def _validate_scaler_state(value):
    if value is None or value == {}:
        return False
    if not isinstance(value, dict):
        raise ValueError("Invalid AMP scaler state type")
    required = {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
    if not required <= value.keys():
        raise ValueError("Incomplete AMP scaler state")
    for key in required:
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"Invalid AMP scaler state: {key}")
    if value["scale"] <= 0 or value["growth_factor"] <= 1 or not 0 < value["backoff_factor"] < 1:
        raise ValueError("Invalid AMP scaler scale or factors")
    if value["growth_interval"] < 1 or value["_growth_tracker"] < 0:
        raise ValueError("Invalid AMP scaler counters")
    return True


def load_checkpoint(path, *, model, optimizer=None, scaler=None, map_location="cpu",
                    config=None, generators=None, data_signatures=None):
    # RNG byte tensors must remain on CPU even when weights target CUDA.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if config is not None:
        validate_resume_config(state["config"], config,
                               saved_data_signatures=state.get("data_signatures"),
                               current_data_signatures=data_signatures,
                               allow_cpu_to_cuda_amp=(state.get("checkpoint_device") == "cpu"
                                                      and torch.device(map_location).type == "cuda"))
    has_scaler_state = _validate_scaler_state(state.get("scaler"))
    model.to(map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        _optimizer_to_device(optimizer)
    if scaler is not None:
        if has_scaler_state:
            scaler.load_state_dict(state["scaler"])
            state["scaler_resume"] = "restored" if scaler.is_enabled() else "disabled"
        else:
            state["scaler_resume"] = "initialized_no_amp_state" if scaler.is_enabled() else "disabled"
    torch.set_rng_state(state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_state_all"]])
    if "python_rng_state" in state:
        random.setstate(state["python_rng_state"])
    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])
    for key, generator in (generators or {}).items():
        if key not in state.get("generator_states", {}):
            raise ValueError(f"Missing generator state: {key}")
        generator.set_state(state["generator_states"][key].cpu())
    return state
