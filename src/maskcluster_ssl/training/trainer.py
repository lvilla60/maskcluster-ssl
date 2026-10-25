"""Configurable MaskCluster training with validation-only model selection."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from maskcluster_ssl.data.transforms import corrupt_masked_patches, normalize_images, sample_generator
from maskcluster_ssl.evaluation.validation import evaluate_validation

from .checkpoint import load_checkpoint, save_checkpoint
from .runtime import cosine_value


class Trainer:
    def __init__(self, model, loss_fn, optimizer, train_loader, config, run_dir, device,
                 data_signatures=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.config = config
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        self.data_signatures = data_signatures
        self.global_step = 0
        self.start_epoch = 0
        self.best_scores = {"teacher": float("-inf"), "student": float("-inf")}
        self.use_amp = bool(config["runtime"].get("amp", True) and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def resume(self, path) -> None:
        if self.data_signatures is None and "data" in self.config:
            raise ValueError("Training resume requires verified data signatures")
        state = load_checkpoint(
            path, model=self.model, optimizer=self.optimizer, scaler=self.scaler,
            map_location=self.device, config=self.config,
            data_signatures=self.data_signatures,
        )
        self.start_epoch = int(state["epoch"]) + 1
        self.global_step = int(state["global_step"])
        self.best_scores = dict(state["best_scores"])
        self._log({"event": "resume", "checkpoint": str(Path(path).resolve()),
                   "device": str(self.device), "scaler": state["scaler_resume"],
                   "source_device": state.get("checkpoint_device"),
                   "amp_enabled": self.use_amp,
                   "data_signatures_verified": self.data_signatures is not None})

    def _schedule(self, total_steps: int) -> tuple[float, float]:
        training = self.config["training"]
        warmup = int(training["warmup_epochs"] * len(self.train_loader))
        lr = cosine_value(training["learning_rate"], training["minimum_learning_rate"], self.global_step, total_steps, warmup)
        momentum = cosine_value(training["ema_momentum"], training["ema_momentum_final"], self.global_step, total_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr, momentum

    def _move_views(self, batch):
        teacher_globals = tuple(view.to(self.device, non_blocking=True) for view in batch["global_views"])
        targets = tuple(view.to(self.device, non_blocking=True) for view in batch["global_targets"])
        masks = tuple(mask.to(self.device, non_blocking=True) for mask in batch["patch_masks"])
        locals_ = tuple(view.to(self.device, non_blocking=True) for view in batch["local_views"])
        student_globals = tuple(normalize_images(corrupt_masked_patches(
            target, mask, self.model.student.backbone.patch_size,
            generator=sample_generator(int(self.config["seed"]), self.global_step, index, "corruption")
        )) for index, (target, mask) in enumerate(zip(targets, masks)))
        return teacher_globals, student_globals, locals_, targets, masks

    def _log(self, record: dict) -> None:
        with (self.run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def train_epoch(self, epoch: int, total_steps: int) -> dict[str, float]:
        self.model.train()
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "epoch"):
            self.train_loader.sampler.epoch = epoch
        totals: dict[str, float] = {}
        batches = 0
        for batch in self.train_loader:
            lr, momentum = self._schedule(total_steps)
            teacher_globals, student_globals, locals_, targets, masks = self._move_views(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(teacher_globals, student_globals, locals_, masks)
                loss, diagnostics = self.loss_fn(outputs, targets, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {self.global_step}: {loss.item()}")
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                self.config["training"]["gradient_clip"],
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.model.update_teacher(momentum)
            record = {name: float(value.cpu()) for name, value in diagnostics.items()}
            record.update(epoch=epoch, step=self.global_step, learning_rate=lr,
                          ema_momentum=momentum, gradient_norm=float(grad_norm))
            self._log(record)
            for key, value in record.items():
                if key.startswith("loss/") or key.startswith("diagnostic/"):
                    totals[key] = totals.get(key, 0.0) + value
            batches += 1
            self.global_step += 1
        return {key: value / max(1, batches) for key, value in totals.items()}

    def validate(self) -> dict:
        validation_path = self.config["data"]["validation"]
        evaluation = self.config["evaluation"]
        results = {}
        for name, encoder in (("teacher", self.model.teacher), ("student", self.model.student)):
            results[name] = evaluate_validation(
                encoder,
                validation_path,
                device=self.device,
                input_size=tuple(self.config["model"]["image_size"]),
                batch_size=evaluation["batch_size"],
                num_workers=self.config["runtime"]["num_workers"],
                query_block_size=evaluation["query_block_size"],
            )
        return results

    def fit(self) -> dict:
        epochs = int(self.config["training"]["epochs"])
        total_steps = max(1, epochs * len(self.train_loader))
        last_validation = None
        for epoch in range(self.start_epoch, epochs):
            train_summary = self.train_epoch(epoch, total_steps)
            if (epoch + 1) % int(self.config["evaluation"]["every_epochs"]) == 0:
                last_validation = self.validate()
                self._log({"epoch": epoch, "validation": last_validation})
                scores = {name: float(last_validation[name]["mean_mAP"])
                          for name in ("teacher", "student")}
                improved = [name for name, score in scores.items()
                            if score > self.best_scores[name]]
                for name in improved:
                    self.best_scores[name] = scores[name]
                for name in improved:
                    save_checkpoint(
                        self.run_dir / f"best_{name}.pt", model=self.model,
                        optimizer=self.optimizer, scaler=self.scaler, epoch=epoch,
                        global_step=self.global_step, config=self.config,
                        best_scores=self.best_scores,
                        data_signatures=self.data_signatures,
                    )
            save_checkpoint(
                self.run_dir / "last.pt", model=self.model, optimizer=self.optimizer,
                scaler=self.scaler, epoch=epoch, global_step=self.global_step,
                config=self.config, best_scores=self.best_scores,
                data_signatures=self.data_signatures,
            )
            self._log({"epoch": epoch, "train_summary": train_summary})
        selected = max(self.best_scores, key=self.best_scores.get)
        summary = {
            "best_scores_mean_mAP": self.best_scores,
            "selected_encoder": selected,
            "selection_rule": "highest mean of original-query and low-resolution-query validation mAP",
            "last_validation": last_validation,
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
