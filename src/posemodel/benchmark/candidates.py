"""Representation candidates evaluated by the common benchmark protocol."""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from posemodel.benchmark.data import BenchmarkCorpus, WindowCollection
from posemodel.benchmark.manifest import CandidateSpec
from posemodel.models import GraphMotionMAE, ModelConfig, ObjectiveConfig
from posemodel.training import load_checkpoint, resolve_device, seed_everything


class RepresentationCandidate(ABC):
    supports_corruption = True

    def __init__(self, spec: CandidateSpec) -> None:
        self.spec = spec
        self.fit_seconds = 0.0

    def fit(self, corpus: BenchmarkCorpus) -> None:
        started = time.perf_counter()
        self._fit(corpus)
        self.fit_seconds = time.perf_counter() - started

    @abstractmethod
    def _fit(self, corpus: BenchmarkCorpus) -> None: ...

    @abstractmethod
    def transform(self, windows: WindowCollection, split: str) -> NDArray[np.float32]: ...

    def save_artifact(self, path: Path) -> None:
        del path


def _masked_statistics(windows: WindowCollection) -> NDArray[np.float32]:
    coordinates = np.where(windows.observed[..., None], windows.coordinates, np.nan)
    velocity = np.diff(coordinates, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        features = [
            np.nanmean(coordinates, axis=1),
            np.nanstd(coordinates, axis=1),
            np.nanmean(np.abs(velocity), axis=1),
            np.nanstd(velocity, axis=1),
        ]
    flattened = [feature.reshape(len(windows), -1) for feature in features]
    flattened.extend(
        [
            windows.context.mean(axis=1).reshape(len(windows), -1),
            windows.context.std(axis=1).reshape(len(windows), -1),
        ]
    )
    return np.asarray(np.nan_to_num(np.concatenate(flattened, axis=1)), dtype=np.float32)


class KinematicsCandidate(RepresentationCandidate):
    def _fit(self, corpus: BenchmarkCorpus) -> None:
        self.scaler = StandardScaler().fit(_masked_statistics(corpus.train))

    def transform(self, windows: WindowCollection, split: str) -> NDArray[np.float32]:
        del split
        return np.asarray(self.scaler.transform(_masked_statistics(windows)), dtype=np.float32)


def _flatten_windows(windows: WindowCollection) -> NDArray[np.float32]:
    coordinates = np.where(windows.observed[..., None], windows.coordinates, 0.0)
    values = [
        coordinates.reshape(len(windows), -1),
        windows.confidence.reshape(len(windows), -1),
        windows.observed.reshape(len(windows), -1).astype(np.float32),
        windows.context.reshape(len(windows), -1),
    ]
    return np.asarray(np.concatenate(values, axis=1), dtype=np.float32)


class PCACandidate(RepresentationCandidate):
    def _fit(self, corpus: BenchmarkCorpus) -> None:
        values = _flatten_windows(corpus.train)
        self.scaler = StandardScaler().fit(values)
        normalized = self.scaler.transform(values)
        requested = int(self.spec.settings.get("latent_dim", 32))
        components = max(1, min(requested, normalized.shape[0] - 1, normalized.shape[1]))
        self.model = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=int(self.spec.settings.get("seed", 42)),
        ).fit(normalized)

    def transform(self, windows: WindowCollection, split: str) -> NDArray[np.float32]:
        del split
        normalized = self.scaler.transform(_flatten_windows(windows))
        return np.asarray(self.model.transform(normalized), dtype=np.float32)


def _temporal_features(windows: WindowCollection) -> NDArray[np.float32]:
    coordinates = np.where(windows.observed[..., None], windows.coordinates, 0.0)
    batch, time = coordinates.shape[:2]
    parts = [
        coordinates.reshape(batch, time, -1),
        windows.confidence.reshape(batch, time, -1),
        windows.observed.reshape(batch, time, -1).astype(np.float32),
        windows.context.reshape(batch, time, -1),
    ]
    return np.asarray(np.concatenate(parts, axis=-1), dtype=np.float32)


class _TemporalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, coordinate_dim: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=8, dilation=4),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.latent = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(
            nn.Conv1d(latent + 3, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, coordinate_dim, kernel_size=1),
        )

    def encode(self, values: Tensor) -> Tensor:
        hidden = self.encoder(values.transpose(1, 2))
        return cast(Tensor, self.latent(self.pool(hidden).squeeze(-1)))

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(values)
        repeated = latent.unsqueeze(-1).expand(-1, -1, values.shape[1])
        position = torch.linspace(-1.0, 1.0, values.shape[1], device=values.device)
        position_features = torch.stack(
            [position, torch.sin(torch.pi * position), torch.cos(torch.pi * position)]
        )
        position_features = position_features.unsqueeze(0).expand(values.shape[0], -1, -1)
        reconstruction = self.decoder(torch.cat([repeated, position_features], dim=1)).transpose(
            1, 2
        )
        return latent, reconstruction


class TCNCandidate(RepresentationCandidate):
    def _fit(self, corpus: BenchmarkCorpus) -> None:
        seed = int(self.spec.settings.get("seed", 42))
        seed_everything(seed)
        device = resolve_device(str(self.spec.settings.get("device", "auto")))
        features = _temporal_features(corpus.train)
        target = corpus.train.coordinates.reshape(len(corpus.train), features.shape[1], -1)
        observed = np.repeat(
            corpus.train.observed.reshape(len(corpus.train), features.shape[1], -1),
            corpus.coordinate_dim,
            axis=-1,
        )
        self.model = _TemporalAutoencoder(
            features.shape[-1],
            target.shape[-1],
            int(self.spec.settings.get("hidden_dim", 128)),
            int(self.spec.settings.get("latent_dim", 32)),
        ).to(device)
        dataset = TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(target),
            torch.from_numpy(observed),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(self.spec.settings.get("batch_size", 64)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.spec.settings.get("learning_rate", 3e-4)),
            weight_decay=float(self.spec.settings.get("weight_decay", 0.01)),
        )
        self.model.train()
        for _ in range(int(self.spec.settings.get("epochs", 20))):
            for values, coordinates, valid in loader:
                values, coordinates, valid = (
                    values.to(device),
                    coordinates.to(device),
                    valid.to(device),
                )
                optimizer.zero_grad(set_to_none=True)
                _, reconstruction = self.model(values)
                error = F.smooth_l1_loss(reconstruction, coordinates, reduction="none")
                loss = (error * valid).sum() / valid.sum().clamp_min(1)
                loss.backward()
                optimizer.step()
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def transform(self, windows: WindowCollection, split: str) -> NDArray[np.float32]:
        del split
        values = torch.from_numpy(_temporal_features(windows))
        embeddings: list[NDArray[np.float32]] = []
        batch_size = int(self.spec.settings.get("batch_size", 64))
        for start in range(0, len(values), batch_size):
            latent = self.model.encode(values[start : start + batch_size].to(self.device))
            embeddings.append(latent.cpu().numpy())
        return np.concatenate(embeddings).astype(np.float32)


class PoseModelCandidate(RepresentationCandidate):
    def _fit(self, corpus: BenchmarkCorpus) -> None:
        self.device = str(self.spec.settings.get("device", "auto"))
        target = resolve_device(self.device)
        if "checkpoint" in self.spec.settings:
            checkpoint = Path(self.spec.settings["checkpoint"])
            self.model, _ = load_checkpoint(checkpoint, device="cpu")
            if self.model.skeleton != corpus.skeleton:
                raise ValueError(f"Checkpoint skeleton does not match candidate {self.spec.name}")
            if (
                self.model.config.coordinate_dim != corpus.coordinate_dim
                or self.model.config.context_dim != corpus.context_dim
            ):
                raise ValueError(f"Checkpoint feature dimensions do not match {self.spec.name}")
            self.trained = False
        else:
            model_settings = dict(self.spec.settings.get("model", {}))
            model_settings["coordinate_dim"] = corpus.coordinate_dim
            model_settings["context_dim"] = corpus.context_dim
            model_settings["max_window_size"] = max(
                int(model_settings.get("max_window_size", 0)), corpus.train.coordinates.shape[1]
            )
            self.model = GraphMotionMAE(corpus.skeleton, ModelConfig(**model_settings))
            self._train_model(corpus, target)
            self.trained = True
        self.model.to(target).eval()
        self.target = target
        self.window_size = int(corpus.train.coordinates.shape[1])

    def _train_model(self, corpus: BenchmarkCorpus, device: torch.device) -> None:
        seed = int(self.spec.settings.get("seed", 42))
        seed_everything(seed)
        self.model.to(device)
        objective = ObjectiveConfig(**dict(self.spec.settings.get("loss", {})))
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=float(self.spec.settings.get("learning_rate", 3e-4)),
            weight_decay=float(self.spec.settings.get("weight_decay", 0.05)),
            betas=(0.9, 0.95),
        )
        train_loader = self._loader(corpus.train, shuffle=True, seed=seed)
        validation_loader = (
            self._loader(corpus.validation, shuffle=False, seed=seed)
            if corpus.validation is not None
            else None
        )
        epochs = int(self.spec.settings.get("epochs", 50))
        patience = int(self.spec.settings.get("patience", 10))
        best_loss = float("inf")
        best_state: dict[str, Tensor] | None = None
        stale = 0
        for epoch in range(epochs):
            self.model.train()
            for coordinates, confidence, observed, context in train_loader:
                coordinates = coordinates.to(device)
                confidence = confidence.to(device)
                observed = observed.to(device)
                context = context.to(device)
                optimizer.zero_grad(set_to_none=True)
                output = self.model(
                    coordinates,
                    confidence,
                    observed,
                    context=context,
                    mask_ratio=float(self.spec.settings.get("mask_ratio", 0.6)),
                )
                multiplier = min(
                    1.0,
                    (epoch + 1) / max(1, int(self.spec.settings.get("kl_warmup_epochs", 20))),
                )
                loss = self.model.objective(
                    output,
                    coordinates,
                    confidence,
                    observed,
                    objective,
                    kl_multiplier=multiplier,
                )["loss"]
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.spec.settings.get("gradient_clip", 1.0))
                )
                optimizer.step()
                self.model.update_teacher(float(self.spec.settings.get("ema_momentum", 0.996)))
            validation_loss = (
                self._validation_loss(validation_loader, objective, device)
                if validation_loader is not None
                else float(loss.detach())
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in self.model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is None:
            raise RuntimeError("PoseModel benchmark training produced no checkpoint")
        self.model.load_state_dict(best_state)

    def _loader(self, windows: WindowCollection, *, shuffle: bool, seed: int) -> DataLoader[Any]:
        dataset = TensorDataset(
            torch.from_numpy(windows.coordinates),
            torch.from_numpy(windows.confidence),
            torch.from_numpy(windows.observed),
            torch.from_numpy(windows.context),
        )
        return DataLoader(
            dataset,
            batch_size=int(self.spec.settings.get("batch_size", 32)),
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed),
        )

    @torch.no_grad()
    def _validation_loss(
        self,
        loader: DataLoader[Any],
        objective: ObjectiveConfig,
        device: torch.device,
    ) -> float:
        self.model.eval()
        losses: list[float] = []
        for coordinates, confidence, observed, context in loader:
            coordinates = coordinates.to(device)
            confidence = confidence.to(device)
            observed = observed.to(device)
            context = context.to(device)
            output = self.model(
                coordinates,
                confidence,
                observed,
                context=context,
                mask_ratio=float(self.spec.settings.get("mask_ratio", 0.6)),
            )
            loss = self.model.objective(output, coordinates, confidence, observed, objective)[
                "loss"
            ]
            losses.append(float(loss))
        return float(np.mean(losses))

    @torch.no_grad()
    def transform(self, windows: WindowCollection, split: str) -> NDArray[np.float32]:
        del split
        batch_size = int(self.spec.settings.get("batch_size", 64))
        results: list[NDArray[np.float32]] = []
        for start in range(0, len(windows), batch_size):
            stop = start + batch_size
            latent = self.model.encode(
                torch.from_numpy(windows.coordinates[start:stop]).to(self.target),
                torch.from_numpy(windows.confidence[start:stop]).to(self.target),
                torch.from_numpy(windows.observed[start:stop]).to(self.target),
                torch.from_numpy(windows.context[start:stop]).to(self.target),
            )
            results.append(latent.cpu().numpy())
        return np.concatenate(results).astype(np.float32)

    def save_artifact(self, path: Path) -> None:
        if not self.trained:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                **self.model.checkpoint_metadata(),
                "training": {
                    "window_size": self.window_size,
                    "stride": int(self.spec.settings.get("stride", max(1, self.window_size // 4))),
                    "benchmark_settings": self.spec.settings,
                },
                "history": [],
                "state_dict": self.model.state_dict(),
            },
            path,
        )


class ImportedCandidate(RepresentationCandidate):
    supports_corruption = False

    def _fit(self, corpus: BenchmarkCorpus) -> None:
        del corpus
        with np.load(Path(self.spec.settings["path"]), allow_pickle=False) as archive:
            self.embeddings = {
                split: np.asarray(archive[split], dtype=np.float32)
                for split in ("train", "validation", "test")
                if split in archive.files
            }
            self.recording_ids = {
                split: np.asarray(archive[f"{split}_recording_ids"], dtype=str)
                for split in self.embeddings
                if f"{split}_recording_ids" in archive.files
            }
            self.starts = {
                split: np.asarray(archive[f"{split}_starts"], dtype=np.int64)
                for split in self.embeddings
                if f"{split}_starts" in archive.files
            }

    def transform(self, windows: WindowCollection, split: str) -> NDArray[np.float32]:
        if split not in self.embeddings:
            raise ValueError(f"Imported embeddings do not contain split '{split}'")
        values = self.embeddings[split]
        if split in self.recording_ids and split in self.starts:
            lookup = {
                (str(recording), int(start)): index
                for index, (recording, start) in enumerate(
                    zip(self.recording_ids[split], self.starts[split], strict=True)
                )
            }
            try:
                indices = [
                    lookup[(str(recording), int(start))]
                    for recording, start in zip(windows.recording_ids, windows.starts, strict=True)
                ]
            except KeyError as error:
                raise ValueError(
                    f"Imported embeddings are missing window {error.args[0]}"
                ) from error
            values = values[indices]
        if len(values) != len(windows):
            raise ValueError(
                f"Imported {split} embedding count {len(values)} != window count {len(windows)}"
            )
        return values


def create_candidate(spec: CandidateSpec) -> RepresentationCandidate:
    candidates: dict[str, type[RepresentationCandidate]] = {
        "kinematics": KinematicsCandidate,
        "pca": PCACandidate,
        "tcn": TCNCandidate,
        "posemodel": PoseModelCandidate,
        "imported": ImportedCandidate,
    }
    return candidates[spec.kind](spec)
