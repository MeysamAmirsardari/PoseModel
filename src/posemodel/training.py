"""Reproducible training, checkpointing, and embedding extraction."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from posemodel.models import GraphMotionMAE, ModelConfig, ObjectiveConfig
from posemodel.schema import Skeleton
from posemodel.windows import PoseWindowDataset


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 42
    window_size: int = 120
    stride: int = 30
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    mask_ratio: float = 0.6
    ema_momentum: float = 0.996
    gradient_clip: float = 1.0
    kl_warmup_epochs: int = 20
    patience: int = 10
    num_workers: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: ObjectiveConfig = field(default_factory=ObjectiveConfig)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> TrainingConfig:
        known = {
            "seed",
            "window_size",
            "stride",
            "batch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "mask_ratio",
            "ema_momentum",
            "gradient_clip",
            "kl_warmup_epochs",
            "patience",
            "num_workers",
        }
        unexpected = set(value) - known - {"model", "loss"}
        if unexpected:
            raise ValueError(f"Unknown training settings: {sorted(unexpected)}")
        scalar = {key: value[key] for key in known if key in value}
        return cls(
            **scalar,
            model=ModelConfig(**value.get("model", {})),
            loss=ObjectiveConfig(**value.get("loss", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream) or {}
        if not isinstance(value, dict):
            raise ValueError("Training configuration must be a YAML mapping")
        return cls.from_mapping(value)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _epoch(
    model: GraphMotionMAE,
    loader: DataLoader[dict[str, Tensor]],
    objective: ObjectiveConfig,
    mask_ratio: float,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    ema_momentum: float,
    gradient_clip: float,
    kl_multiplier: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["coordinates"],
                batch["confidence"],
                batch["observed"],
                mask_ratio=mask_ratio,
                context=batch["context"],
            )
            metrics = model.objective(
                output,
                batch["coordinates"],
                batch["confidence"],
                batch["observed"],
                objective,
                kl_multiplier=kl_multiplier,
            )
            if optimizer is not None:
                metrics["loss"].backward()  # type: ignore[no-untyped-call]
                clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                model.update_teacher(ema_momentum)
            batch_size = batch["coordinates"].shape[0]
            count += batch_size
            for name, metric in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(metric.detach()) * batch_size
    return {name: total / max(count, 1) for name, total in totals.items()}


def fit(
    model: GraphMotionMAE,
    train_dataset: PoseWindowDataset,
    validation_dataset: PoseWindowDataset,
    config: TrainingConfig,
    output_directory: str | Path,
    *,
    device: str = "auto",
) -> list[dict[str, Any]]:
    """Train with early stopping and write the best portable checkpoint."""
    seed_everything(config.seed)
    target = resolve_device(device)
    model.to(target)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_validation = math_inf = float("inf")
    stale_epochs = 0
    for epoch in range(config.epochs):
        kl_multiplier = min(1.0, (epoch + 1) / max(1, config.kl_warmup_epochs))
        train_metrics = _epoch(
            model,
            train_loader,
            config.loss,
            config.mask_ratio,
            target,
            optimizer=optimizer,
            ema_momentum=config.ema_momentum,
            gradient_clip=config.gradient_clip,
            kl_multiplier=kl_multiplier,
        )
        validation_metrics = _epoch(
            model,
            validation_loader,
            config.loss,
            config.mask_ratio,
            target,
            optimizer=None,
            ema_momentum=config.ema_momentum,
            gradient_clip=config.gradient_clip,
            kl_multiplier=1.0,
        )
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        validation_loss = validation_metrics["loss"]
        if validation_loss < best_validation:
            best_validation = validation_loss
            stale_epochs = 0
            save_checkpoint(output_path / "model.pt", model, config, history)
        else:
            stale_epochs += 1
        scheduler.step()
        (output_path / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale_epochs >= config.patience:
            break
    if best_validation == math_inf:
        raise RuntimeError("Training produced no validation result")
    return history


def save_checkpoint(
    path: str | Path,
    model: GraphMotionMAE,
    training_config: TrainingConfig,
    history: list[dict[str, Any]] | None = None,
) -> None:
    payload = {
        "format_version": 1,
        **model.checkpoint_metadata(),
        "training": asdict(training_config),
        "history": history or [],
        "state_dict": model.state_dict(),
    }
    torch.save(payload, Path(path))


def load_checkpoint(
    path: str | Path, *, device: str = "cpu"
) -> tuple[GraphMotionMAE, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported PoseModel checkpoint version")
    skeleton = Skeleton.from_dict(payload["skeleton"])
    model = GraphMotionMAE(skeleton, ModelConfig(**payload["model"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


@torch.no_grad()
def embed(
    model: GraphMotionMAE,
    dataset: PoseWindowDataset,
    *,
    batch_size: int = 64,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = resolve_device(device)
    model.to(target).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    stops: list[np.ndarray] = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, target)
        latent = model.encode(
            batch["coordinates"],
            batch["confidence"],
            batch["observed"],
            batch["context"],
        )
        embeddings.append(latent.cpu().numpy())
        starts.append(raw_batch["start"].numpy())
        stops.append(raw_batch["stop"].numpy())
    return np.concatenate(embeddings), np.concatenate(starts), np.concatenate(stops)
