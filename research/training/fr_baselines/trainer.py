"""Trainer implementation for face recognition baselines.

Implements Adam optimizer, multistep LR decay schedule (epochs 10, 20, 30),
gradient accumulation for GTX 1080 Ti 11GB VRAM, resume functionality,
and KEEP_RAW_RESULTS pruning.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from research.training.fr_baselines.models import (
    FRBaselineTrainingModule,
    export_backbone_checkpoint,
)

logger = logging.getLogger(__name__)


class FRBaselineTrainer:
    """Orchestrates MobileFaceNet baseline training."""

    def __init__(
        self,
        model: FRBaselineTrainingModule,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        *,
        batch_size: int = 64,
        micro_batch_size: int = 32,
        grad_accum_steps: int | None = None,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        decay_epochs: tuple[int, ...] = (10, 20, 30),
        gamma: float = 0.1,
        epochs: int = 40,
        device: str | torch.device | None = None,
        output_dir: str | Path = "results/training/fr_baselines",
        keep_raw_results: bool = True,
        resume_checkpoint: str | Path | None = None,
        num_workers: int = 0,
        dry_run: bool = False,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.micro_batch_size = max(1, int(micro_batch_size))
        if grad_accum_steps is not None:
            self.accumulation_steps = max(1, int(grad_accum_steps))
            self.batch_size = self.micro_batch_size * self.accumulation_steps
        else:
            self.batch_size = max(1, int(batch_size))
            self.accumulation_steps = max(1, self.batch_size // self.micro_batch_size)

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.decay_epochs = tuple(decay_epochs)
        self.gamma = float(gamma)
        self.epochs = 1 if dry_run else int(epochs)
        self.keep_raw_results = bool(keep_raw_results)
        self.dry_run = bool(dry_run)
        self.num_workers = int(num_workers)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=list(self.decay_epochs),
            gamma=self.gamma,
        )

        self.current_epoch = 0
        self.best_metric = -float("inf")
        self.history: list[dict[str, Any]] = []

        if resume_checkpoint is not None and Path(resume_checkpoint).is_file():
            self.resume_from_checkpoint(resume_checkpoint)

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train model for a single epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        self.optimizer.zero_grad()
        accumulated_batches = 0

        for step, (images, labels) in enumerate(dataloader):
            if self.dry_run and step >= 2:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            output = self.model(images, labels)
            loss = output["loss"]
            loss_for_accum = loss / self.accumulation_steps
            loss_for_accum.backward()
            accumulated_batches += 1

            if accumulated_batches % self.accumulation_steps == 0 or (
                step + 1 == len(dataloader)
            ):
                self.optimizer.step()
                self.optimizer.zero_grad()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            logits = output["logits"]
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()

        if accumulated_batches % self.accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        return {"train_loss": avg_loss, "train_acc": accuracy}

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model on validation dataloader."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for step, (images, labels) in enumerate(dataloader):
                if self.dry_run and step >= 2:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                output = self.model(images, labels)
                loss = output["loss"]
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                logits = output["logits"]
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels).sum().item()

        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        return {"val_loss": avg_loss, "val_acc": accuracy}

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
        *,
        is_best: bool = False,
    ) -> Path:
        """Save training state checkpoint."""
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "backbone_state_dict": self.model.backbone.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "head_type": self.model.head_type,
            "best_metric": self.best_metric,
        }

        latest_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(ckpt_payload, latest_path)

        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(ckpt_payload, best_path)

        if self.keep_raw_results:
            epoch_path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(ckpt_payload, epoch_path)
            return epoch_path

        return latest_path

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Resume trainer state from checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.current_epoch = ckpt["epoch"] + 1
        self.best_metric = ckpt.get("best_metric", -float("inf"))
        logger.info(
            "Resumed training from epoch %d with best_metric=%.4f",
            self.current_epoch,
            self.best_metric,
        )

    def fit(self) -> dict[str, Any]:
        """Execute full training loop."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.micro_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

        start_time = time.time()
        for epoch in range(self.current_epoch, self.epochs):
            epoch_start = time.time()
            train_metrics = self.train_epoch(train_loader)

            val_metrics: dict[str, float] = {}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                current_score = val_metrics["val_acc"]
            else:
                current_score = -train_metrics["train_loss"]

            self.scheduler.step()
            is_best = current_score > self.best_metric
            if is_best:
                self.best_metric = current_score

            all_metrics = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                "duration_seconds": time.time() - epoch_start,
                **train_metrics,
                **val_metrics,
            }
            self.history.append(all_metrics)

            self.save_checkpoint(epoch, all_metrics, is_best=is_best)

            logger.info(
                "Epoch %02d/%02d: train_loss=%.4f train_acc=%.4f val_acc=%.4f (%.2fs)",
                epoch,
                self.epochs,
                train_metrics["train_loss"],
                train_metrics["train_acc"],
                val_metrics.get("val_acc", 0.0),
                all_metrics["duration_seconds"],
            )

        # Export final clean backbone checkpoint
        backbone_export_path = self.output_dir / "backbone_final.pt"
        export_backbone_checkpoint(
            self.model,
            backbone_export_path,
            metadata={
                "head_type": self.model.head_type,
                "epochs_trained": self.epochs,
                "best_metric": self.best_metric,
            },
        )

        # Write metrics history
        history_path = self.output_dir / "metrics_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "head_type": self.model.head_type,
                    "total_epochs": self.epochs,
                    "best_metric": self.best_metric,
                    "elapsed_seconds": time.time() - start_time,
                    "history": self.history,
                },
                f,
                indent=2,
            )

        if not self.keep_raw_results:
            # Prune large per-epoch checkpoints, preserving latest, best, and final backbone
            for epoch_file in self.output_dir.glob("checkpoint_epoch_*.pt"):
                try:
                    epoch_file.unlink(missing_ok=True)
                except OSError:
                    pass

        return {
            "best_metric": self.best_metric,
            "final_backbone": str(backbone_export_path),
            "metrics_history": str(history_path),
            "history": self.history,
        }
