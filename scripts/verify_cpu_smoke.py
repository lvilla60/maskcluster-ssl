"""Bounded CPU/CUDA verification: at most one step per variant."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from maskcluster_ssl.data.audit import within_root
from maskcluster_ssl.data.hdf5 import ReIDH5Dataset, TrainingH5Dataset
from maskcluster_ssl.data.transforms import (
    EvalTransform,
    MultiCropTransform,
    corrupt_masked_patches,
    normalize_images,
)
from maskcluster_ssl.evaluation.extract import extract_embeddings
from maskcluster_ssl.evaluation.reid import evaluate_reid, summarize_conditions
from maskcluster_ssl.losses import LossWeights, MaskClusterLoss
from maskcluster_ssl.models import MaskClusterSSL
from maskcluster_ssl.training.checkpoint import load_checkpoint, save_checkpoint
from maskcluster_ssl.training.runtime import seed_everything, select_device, write_run_metadata
from maskcluster_ssl.training.trainer import Trainer


def build_model(variant: str, model_config: dict) -> MaskClusterSSL:
    return MaskClusterSSL(
        variant=variant,
        image_size=model_config["image_size"],
        patch_size=model_config["patch_size"],
        embed_dim=model_config["embed_dim"],
        depth=model_config["depth"],
        num_heads=model_config["num_heads"],
        mlp_ratio=model_config["mlp_ratio"],
        prototypes=model_config["prototypes"],
        reconstruction_layers=model_config["reconstruction_layers"],
        attention_mode=model_config.get("attention_mode", "visible"),
    )


def build_loss(config: dict, patch_size: int) -> MaskClusterLoss:
    return MaskClusterLoss(
        student_temperature=config["loss"]["student_temperature"],
        teacher_temperature=config["loss"]["teacher_temperature"],
        me_max_temperature=config["loss"]["me_max_temperature"],
        patch_size=patch_size,
        weights=LossWeights(**config["loss"]["weights"]),
    )


def real_batch(training_path: Path, config: dict, model_config: dict, batch_size: int):
    transform = MultiCropTransform(
        global_size=model_config["image_size"],
        local_size=model_config["local_size"],
        local_crops=model_config["local_crops"],
        patch_size=model_config["patch_size"],
        mask_ratio=model_config["mask_ratio"],
        seed=config["seed"],
        **model_config.get("photometric", {}),
    )
    dataset = TrainingH5Dataset(training_path, transform=transform)
    try:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=True,
            num_workers=config["runtime"].get("num_workers", 0),
        )
        return next(iter(loader))
    finally:
        dataset.close()


def optimizer_for(model, config):
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )


def assert_finite_gradients(model) -> int:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Missing or non-finite trainable gradients")
    if any(parameter.grad is not None for parameter in model.teacher.parameters()):
        raise RuntimeError("EMA teacher received gradients")
    return len(gradients)


def verify_variant(variant, batch, config, run_dir, device):
    seed_everything(config["seed"])
    model = build_model(variant, config["model"])
    optimizer = optimizer_for(model, config)
    trainer = Trainer(model, build_loss(config, config["model"]["patch_size"]), optimizer, [batch], config, run_dir, device)
    teacher_before = [parameter.detach().clone() for parameter in model.teacher.parameters()]
    summary = trainer.train_epoch(epoch=0, total_steps=1)
    gradient_count = assert_finite_gradients(model)
    if not all(math.isfinite(value) for value in summary.values()):
        raise RuntimeError(f"Non-finite diagnostics for {variant}")
    if not any(not torch.equal(before, after) for before, after in zip(teacher_before, model.teacher.parameters())):
        raise RuntimeError(f"EMA teacher did not update for {variant}")
    expected_zero = {
        "baseline": ("loss/patch_clustering", "loss/reconstruction", "loss/patch_me_max"),
        "no_mim": ("loss/reconstruction",),
        "full": (),
    }[variant]
    if any(abs(summary[name]) > 1e-12 for name in expected_zero):
        raise RuntimeError(f"Inactive loss branch became active for {variant}")

    checkpoint = run_dir / "roundtrip.pt"
    checkpoint_config = copy.deepcopy(config)
    checkpoint_config["model"]["variant"] = variant
    save_checkpoint(
        checkpoint, model=model, optimizer=optimizer, scaler=trainer.scaler,
        epoch=0, global_step=trainer.global_step, config=checkpoint_config,
        best_scores={"teacher": 0.0, "student": 0.0},
    )
    reloaded = build_model(variant, config["model"])
    reloaded_optimizer = optimizer_for(reloaded, config)
    state = load_checkpoint(checkpoint, model=reloaded, optimizer=reloaded_optimizer, map_location=device, config=checkpoint_config)
    for name, value in model.state_dict().items():
        if not torch.equal(value.cpu(), reloaded.state_dict()[name].cpu()):
            raise RuntimeError(f"Checkpoint mismatch at {name}")
    if state["global_step"] != 1:
        raise RuntimeError("Checkpoint did not restore the optimization step")
    for parameter, values in reloaded_optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in values and values[key].device != parameter.device:
                raise RuntimeError(f"Optimizer {key} is on the wrong device")
    return model, {
        "optimization_steps": trainer.global_step,
        "finite_gradient_tensors": gradient_count,
        "losses": summary,
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
    }


def prepare_views(batch, patch_size):
    teacher = tuple(view for view in batch["global_views"])
    targets = tuple(view for view in batch["global_targets"])
    masks = tuple(view for view in batch["patch_masks"])
    locals_ = tuple(view for view in batch["local_views"])
    student = tuple(
        normalize_images(corrupt_masked_patches(target, mask, patch_size))
        for target, mask in zip(targets, masks)
    )
    return teacher, student, locals_, targets, masks


def verify_planned_backbone(training_path, config, device):
    seed_everything(config["seed"])
    model_config = config["planned_backbone"]
    batch = real_batch(training_path, config, model_config, batch_size=1)
    model = build_model("full", model_config).to(device)
    teacher, student, locals_, targets, masks = prepare_views(batch, model_config["patch_size"])
    teacher = tuple(view.to(device) for view in teacher)
    student = tuple(view.to(device) for view in student)
    locals_ = tuple(view.to(device) for view in locals_)
    targets = tuple(view.to(device) for view in targets)
    masks = tuple(view.to(device) for view in masks)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        outputs = model(teacher, student, locals_, masks)
        loss, diagnostics = build_loss(config, model_config["patch_size"]).to(device)(outputs, targets, masks)
    loss.backward()
    gradient_count = assert_finite_gradients(model)
    first = outputs["student_globals"][0]
    result = {
        "optimization_steps": 0,
        "loss": float(loss.detach()),
        "finite_gradient_tensors": gradient_count,
        "cls_shape": list(first["features"].shape),
        "patch_shape": list(first["patch_features"].shape),
        "patch_logits_shape": list(first["patch_logits"].shape),
        "reconstruction_shape": list(outputs["reconstructions"][0].shape),
        "local_crops_in_bounded_check": model_config["local_crops"],
        "diagnostics": {name: float(value) for name, value in diagnostics.items()},
    }
    del outputs, loss, model
    return result


def validation_indices(path: Path, query_count: int, gallery_count: int):
    with h5py.File(path, "r") as handle:
        q_pid = np.asarray(handle["query/person_ids"])
        q_cam = np.asarray(handle["query/camera_ids"])
        g_pid = np.asarray(handle["gallery/person_ids"])
        g_cam = np.asarray(handle["gallery/camera_ids"])
    queries = []
    gallery = []
    for index, (pid, cam) in enumerate(zip(q_pid, q_cam)):
        positives = np.flatnonzero((g_pid == pid) & (g_cam != cam) & (g_pid != -1))
        if positives.size:
            queries.append(index)
            gallery.extend(positives[:2].tolist())
        if len(queries) == query_count:
            break
    if len(queries) != query_count:
        raise RuntimeError("Insufficient valid validation queries")
    chosen = list(dict.fromkeys(gallery))
    for index in range(len(g_pid)):
        if index not in chosen:
            chosen.append(index)
        if len(chosen) == gallery_count:
            break
    return queries, chosen


def verify_validation_subset(model, validation_path, config, device):
    query_indices, gallery_indices = validation_indices(
        validation_path,
        config["smoke"]["validation_queries"],
        config["smoke"]["validation_gallery"],
    )
    transform = EvalTransform(config["model"]["image_size"])
    gallery = ReIDH5Dataset(validation_path, "gallery", transform=transform)
    original = ReIDH5Dataset(validation_path, "query", transform=transform)
    low = ReIDH5Dataset(validation_path, "query", low_resolution=True, transform=transform)

    def loader(dataset, indices):
        return DataLoader(Subset(dataset, indices), batch_size=config["evaluation"]["batch_size"], shuffle=False)

    report = {"query_indices": query_indices, "gallery_indices": gallery_indices, "diagnostic_only": True}
    try:
        for name, encoder in (("teacher", model.teacher), ("student", model.student)):
            before = {key: value.detach().clone() for key, value in encoder.state_dict().items()}
            gallery_embeddings, gallery_meta = extract_embeddings(encoder, loader(gallery, gallery_indices), device)
            conditions = []
            for dataset in (original, low):
                query_embeddings, query_meta = extract_embeddings(encoder, loader(dataset, query_indices), device)
                if not np.allclose(np.linalg.norm(query_embeddings, axis=1), 1.0, atol=1e-5):
                    raise RuntimeError("Query embeddings are not L2 normalized")
                conditions.append(evaluate_reid(
                    query_embeddings, gallery_embeddings,
                    query_meta["person_id"], gallery_meta["person_id"],
                    query_meta["camera_id"], gallery_meta["camera_id"],
                    query_block_size=config["evaluation"]["query_block_size"],
                ))
            if any(not torch.equal(value, before[key]) for key, value in encoder.state_dict().items()):
                raise RuntimeError("Validation modified encoder state")
            report[name] = summarize_conditions(*conditions)
    finally:
        gallery.close()
        original.close()
        low.close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke_cpu.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args()
    config_path = within_root(args.config, ROOT)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.device:
        config["runtime"]["device"] = args.device
    device = select_device(config["runtime"]["device"])
    config["runtime"]["amp"] = device.type == "cuda"
    if config["smoke"]["steps_per_variant"] != 1 or len(config["smoke"]["variants"]) > 3:
        raise ValueError("The smoke test is limited to one step for at most three variants")
    torch.set_num_threads(config["runtime"]["torch_threads"])
    training_path = within_root(config["data"]["training"], ROOT)
    validation_path = within_root(config["data"]["validation"], ROOT)
    config["data"]["training"] = str(training_path)
    config["data"]["validation"] = str(validation_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = within_root(Path(config["output_dir"]) / run_id, ROOT)
    write_run_metadata(run_dir, config, ROOT)

    report = {"run_id": run_id, "purpose": "smoke", "diagnostic_only": True,
              "variants": {}, "total_optimization_steps": 0}
    full_model = None
    for variant in config["smoke"]["variants"]:
        variant_dir = run_dir / variant
        variant_dir.mkdir()
        seed_everything(config["seed"])
        batch = real_batch(training_path, config, config["model"], config["training"]["batch_size"])
        model, result = verify_variant(variant, batch, config, variant_dir, device)
        report["variants"][variant] = result
        result["validation_subset"] = verify_validation_subset(model, validation_path, config, device)
        result["parameters"] = {"total": sum(p.numel() for p in model.parameters()),
                                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        report["total_optimization_steps"] += result["optimization_steps"]
        if variant == "full":
            full_model = model
    if report["total_optimization_steps"] > 3:
        raise RuntimeError("Smoke test exceeded the three-step optimization limit")
    if full_model is None:
        raise RuntimeError("Full variant is required for validation verification")
    report["planned_backbone"] = verify_planned_backbone(training_path, config, device)
    report["validation_subset"] = verify_validation_subset(full_model, validation_path, config, device)
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
