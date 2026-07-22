"""Versioned benchmark manifest schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Split = Literal["train", "validation", "test"]
SplitUnit = Literal["animal", "session", "recording"]


@dataclass(frozen=True, slots=True)
class RecordingSpec:
    id: str
    path: Path
    animal_id: str
    session_id: str
    split: Split
    labels: Path | None = None


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    name: str
    kind: Literal["kinematics", "pca", "tcn", "posemodel", "imported"]
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CorruptionSpec:
    name: str
    kind: Literal["missing", "joint_dropout", "jitter"]
    magnitude: float


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    version: int
    name: str
    window_size: int
    stride: int
    seed: int
    split_unit: SplitUnit
    recordings: tuple[RecordingSpec, ...]
    candidates: tuple[CandidateSpec, ...]
    corruptions: tuple[CorruptionSpec, ...] = ()
    max_windows_per_split: int | None = 20_000
    few_shot_fractions: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 1.0)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        manifest_path = Path(path).resolve()
        with manifest_path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream) or {}
        if not isinstance(value, dict):
            raise ValueError("Benchmark manifest must be a YAML mapping")
        base = manifest_path.parent
        recordings = tuple(
            RecordingSpec(
                id=str(item["id"]),
                path=(base / item["path"]).resolve(),
                animal_id=str(item["animal_id"]),
                session_id=str(item["session_id"]),
                split=item["split"],
                labels=(base / item["labels"]).resolve() if item.get("labels") else None,
            )
            for item in value.get("recordings", [])
        )
        candidate_values = []
        for item in value.get("candidates", []):
            settings = dict(item.get("settings", {}))
            for key in ("path", "checkpoint"):
                if key in settings:
                    settings[key] = str((base / settings[key]).resolve())
            candidate_values.append(
                CandidateSpec(
                    name=str(item.get("name", item["kind"])),
                    kind=item["kind"],
                    settings=settings,
                )
            )
        candidates = tuple(candidate_values)
        corruptions = tuple(
            CorruptionSpec(
                name=str(item.get("name", f"{item['kind']}_{item['magnitude']}")),
                kind=item["kind"],
                magnitude=float(item["magnitude"]),
            )
            for item in value.get("corruptions", [])
        )
        config = cls(
            version=int(value.get("version", 1)),
            name=str(value.get("name", manifest_path.stem)),
            window_size=int(value.get("window_size", 120)),
            stride=int(value.get("stride", 30)),
            seed=int(value.get("seed", 42)),
            split_unit=value.get("split_unit", "animal"),
            recordings=recordings,
            candidates=candidates,
            corruptions=corruptions,
            max_windows_per_split=value.get("max_windows_per_split", 20_000),
            few_shot_fractions=tuple(
                float(item)
                for item in value.get("few_shot_fractions", [0.01, 0.05, 0.1, 0.25, 1.0])
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError(f"Unsupported benchmark manifest version: {self.version}")
        if self.window_size <= 1 or self.stride <= 0:
            raise ValueError("window_size must be > 1 and stride must be positive")
        if not self.recordings or not self.candidates:
            raise ValueError("Benchmark requires recordings and candidates")
        if len({recording.id for recording in self.recordings}) != len(self.recordings):
            raise ValueError("Recording IDs must be unique")
        if len({candidate.name for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("Candidate names must be unique")
        allowed_candidates = {"kinematics", "pca", "tcn", "posemodel", "imported"}
        for candidate in self.candidates:
            if candidate.kind not in allowed_candidates:
                raise ValueError(f"Unsupported candidate kind: {candidate.kind}")
            for key in ("path", "checkpoint"):
                if key in candidate.settings and not Path(candidate.settings[key]).exists():
                    raise FileNotFoundError(candidate.settings[key])
            if candidate.kind == "imported" and "path" not in candidate.settings:
                raise ValueError(f"Imported candidate {candidate.name} requires settings.path")
            if (
                candidate.kind in {"tcn", "posemodel"}
                and "checkpoint" not in candidate.settings
                and int(candidate.settings.get("epochs", 20)) <= 0
            ):
                raise ValueError(f"Candidate {candidate.name} must train for at least one epoch")
        splits = {recording.split for recording in self.recordings}
        if not {"train", "test"}.issubset(splits):
            raise ValueError("Benchmark requires explicit train and test recordings")
        for recording in self.recordings:
            if recording.split not in {"train", "validation", "test"}:
                raise ValueError(f"Invalid split for {recording.id}: {recording.split}")
            if not recording.path.exists():
                raise FileNotFoundError(recording.path)
            if recording.labels is not None and not recording.labels.exists():
                raise FileNotFoundError(recording.labels)
        group_splits: dict[str, set[str]] = {}
        for recording in self.recordings:
            if self.split_unit == "animal":
                group = recording.animal_id
            elif self.split_unit == "session":
                group = recording.session_id
            else:
                group = recording.id
            group_splits.setdefault(group, set()).add(recording.split)
        leaked = {group: values for group, values in group_splits.items() if len(values) > 1}
        if leaked:
            raise ValueError(f"{self.split_unit} groups cross benchmark splits: {leaked}")
        if any(not 0 < fraction <= 1 for fraction in self.few_shot_fractions):
            raise ValueError("Few-shot fractions must be in (0, 1]")
        for corruption in self.corruptions:
            if corruption.magnitude < 0:
                raise ValueError("Corruption magnitudes must be non-negative")
            if corruption.kind in {"missing", "joint_dropout"} and corruption.magnitude > 1:
                raise ValueError(f"{corruption.kind} magnitude must be in [0, 1]")
