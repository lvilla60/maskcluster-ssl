"""Configurable MaskCluster training with validation-only model selection."""

from __future__ import annotations

import json
import time
import shutil
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
        source = Path(path).resolve().parent
        history = source / "history.jsonl"
        if source != self.run_dir.resolve():
            if not history.exists():
                raise ValueError("Resume requires the source history.jsonl")
            if (self.run_dir / "history.jsonl").exists():
                raise ValueError("Resume destination already contains history")
            records = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
            records = [r for r in records if r.get("epoch", -1) <= int(state["epoch"])]
            (self.run_dir / "history.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
            for name in ("teacher", "student"):
                best = source / f"best_{name}.pt"
                if best.exists():
                    best_state = torch.load(best, map_location="cpu", weights_only=False)
                    if best_state["epoch"] > state["epoch"]:
                        raise ValueError("Best checkpoint is newer than resume boundary; resume last.pt")
                    shutil.copy2(best, self.run_dir / best.name)
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
        record = {"run_id": self.run_dir.name, "variant": self.model.variant,
                  "seed": self.config.get("seed"), **record}
        with (self.run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def train_epoch(self, epoch: int, total_steps: int) -> dict[str, float]:
        self.model.train()
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "epoch"):
            self.train_loader.sampler.epoch = epoch
        totals: dict[str, float] = {}
        batches = 0
        samples = 0
        for batch in self.train_loader:
            lr, momentum = self._schedule(total_steps)
            teacher_globals, student_globals, locals_, targets, masks = self._move_views(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(teacher_globals, student_globals, locals_, masks)
                loss, diagnostics = self.loss_fn(outputs, targets, masks)
            if not torch.isfinite(loss):
                self._log({"event": "nonfinite_loss", "epoch": epoch,
                           "step": self.global_step, "optimizer_step_executed": False})
                raise FloatingPointError(f"Non-finite loss at step {self.global_step}: {loss.item()}")
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            try:
                grad_norm = clip_grad_norm_(
                    [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                    self.config["training"]["gradient_clip"],
                    error_if_nonfinite=True,
                )
            except RuntimeError:
                self._log({"event": "gradient_check_failed", "epoch": epoch,
                           "step": self.global_step, "optimizer_step_executed": False,
                           "policy": "fail_before_step; no scaler backoff or EMA"})
                raise
            executed = []
            hook = self.optimizer.register_step_post_hook(lambda *args: executed.append(True))
            try:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            finally:
                hook.remove()
            if not executed:
                self._log({"event": "optimizer_step_skipped", "epoch": epoch,
                           "step": self.global_step, "optimizer_step_executed": False})
                raise RuntimeError("GradScaler skipped optimizer step; stopping without EMA or counter update")
            self.model.update_teacher(momentum)
            record = {name: float(value.cpu()) for name, value in diagnostics.items()}
            record.update(epoch=epoch, step=self.global_step, learning_rate=lr,
                          ema_momentum=momentum, gradient_norm=float(grad_norm),
                          optimizer_step_executed=True, samples=int(teacher_globals[0].shape[0]))
            self._log(record)
            for key, value in record.items():
                if key.startswith("loss/") or key.startswith("diagnostic/"):
                    totals[key] = totals.get(key, 0.0) + value
            batches += 1
            samples += int(teacher_globals[0].shape[0])
            self.global_step += 1
        summary = {key: value / max(1, batches) for key, value in totals.items()}
        summary.update(steps=batches, samples=samples, learning_rate=lr if batches else None)
        return summary

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

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def fit(self, max_epochs_per_invocation=None) -> dict:
        epochs = int(self.config["training"]["epochs"])
        if max_epochs_per_invocation is not None and (
                isinstance(max_epochs_per_invocation, bool)
                or not isinstance(max_epochs_per_invocation, int)
                or max_epochs_per_invocation < 1):
            raise ValueError("max_epochs_per_invocation must be a positive integer")
        if self.start_epoch >= epochs:
            raise ValueError("Checkpoint already completed the planned epoch budget")
        stop_epoch = min(epochs, self.start_epoch + max_epochs_per_invocation) if max_epochs_per_invocation else epochs
        total_steps = max(1, epochs * len(self.train_loader))
        last_validation = None
        completed_epochs = self.start_epoch
        for epoch in range(self.start_epoch, stop_epoch):
            self._synchronize()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            start_step = self.global_step
            train_summary = self.train_epoch(epoch, total_steps)
            self._synchronize()
            train_seconds = time.perf_counter() - started
            validation_seconds = 0.0
            epoch_validation = None
            if (epoch + 1) % int(self.config["evaluation"]["every_epochs"]) == 0:
                started = time.perf_counter()
                last_validation = self.validate()
                self._synchronize()
                epoch_validation = last_validation
                validation_seconds = time.perf_counter() - started
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
            self._log({"event": "epoch", "epoch": epoch, "train_summary": train_summary,
                       "validation": epoch_validation, "train_seconds": train_seconds,
                       "validation_seconds": validation_seconds,
                       "steps": self.global_step - start_step, "global_step": self.global_step,
                       "samples": train_summary.get("samples", 0),
                       "train_images_per_second": train_summary.get("samples", 0) / train_seconds,
                       "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
                       "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None})
            save_checkpoint(
                self.run_dir / "last.pt", model=self.model, optimizer=self.optimizer,
                scaler=self.scaler, epoch=epoch, global_step=self.global_step,
                config=self.config, best_scores=self.best_scores,
                data_signatures=self.data_signatures,
            )
            completed_epochs = epoch + 1
        selected = max(self.best_scores, key=self.best_scores.get)
        summary = {
            "completed_epochs": completed_epochs,
            "start_epoch": self.start_epoch,
            "epochs_this_invocation": completed_epochs - self.start_epoch,
            "max_epochs_per_invocation": max_epochs_per_invocation,
            "paused": completed_epochs < epochs,
            "status": "paused" if completed_epochs < epochs else "completed",
            "global_step": self.global_step,
            "planned_epochs": epochs,
            "steps_per_epoch": len(self.train_loader),
            "best_scores_mean_mAP": self.best_scores,
            "selected_encoder": selected,
            "selection_rule": "highest mean of original-query and low-resolution-query validation mAP",
            "last_validation": last_validation,
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
