"""Canonical pose and skeleton data contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Skeleton:
    """Named joint topology shared by all individuals in a sequence."""

    joint_names: tuple[str, ...]
    edges: tuple[tuple[int, int], ...] = ()
    root: int | None = None
    left_right_pairs: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.joint_names:
            raise ValueError("A skeleton must contain at least one joint")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Joint names must be unique")
        n_joints = len(self.joint_names)
        indices = [index for edge in self.edges for index in edge]
        indices.extend(index for pair in self.left_right_pairs for index in pair)
        if any(index < 0 or index >= n_joints for index in indices):
            raise ValueError("Skeleton edge or symmetry index is out of range")
        if self.root is not None and not 0 <= self.root < n_joints:
            raise ValueError("Skeleton root is out of range")

    @classmethod
    def from_names(
        cls,
        joint_names: list[str] | tuple[str, ...],
        edges: list[tuple[str, str]] | tuple[tuple[str, str], ...] = (),
        root: str | None = None,
        left_right_pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...] = (),
    ) -> Skeleton:
        """Construct a skeleton with readable name-based topology."""
        names = tuple(str(name) for name in joint_names)
        lookup = {name: index for index, name in enumerate(names)}
        try:
            edge_indices = tuple((lookup[a], lookup[b]) for a, b in edges)
            pair_indices = tuple((lookup[a], lookup[b]) for a, b in left_right_pairs)
            root_index = lookup[root] if root is not None else None
        except KeyError as error:
            raise ValueError(f"Unknown joint in skeleton topology: {error.args[0]}") from error
        return cls(names, edge_indices, root_index, pair_indices)

    def adjacency(self, include_self: bool = True) -> NDArray[np.bool_]:
        """Return a symmetric boolean adjacency matrix."""
        adjacency = np.zeros((len(self.joint_names), len(self.joint_names)), dtype=bool)
        for source, target in self.edges:
            adjacency[source, target] = True
            adjacency[target, source] = True
        if include_self:
            np.fill_diagonal(adjacency, True)
        return adjacency

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_names": list(self.joint_names),
            "edges": [list(edge) for edge in self.edges],
            "root": self.root,
            "left_right_pairs": [list(pair) for pair in self.left_right_pairs],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Skeleton:
        return cls(
            joint_names=tuple(value["joint_names"]),
            edges=tuple(tuple(edge) for edge in value.get("edges", [])),
            root=value.get("root"),
            left_right_pairs=tuple(tuple(pair) for pair in value.get("left_right_pairs", [])),
        )


@dataclass(slots=True)
class PoseSequence:
    """A pose recording with explicit missingness and provenance.

    Coordinates have shape ``(frames, individuals, joints, dimensions)``. Confidence and
    observed have shape ``(frames, individuals, joints)``.
    """

    coordinates: FloatArray
    confidence: FloatArray
    observed: BoolArray
    fps: float
    skeleton: Skeleton
    individual_names: tuple[str, ...] = ("individual_0",)
    context: FloatArray | None = None
    context_names: tuple[str, ...] = ()
    frame_times: FloatArray | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coordinates = np.asarray(self.coordinates, dtype=np.float32)
        self.confidence = np.asarray(self.confidence, dtype=np.float32)
        self.observed = np.asarray(self.observed, dtype=bool)
        if self.coordinates.ndim != 4:
            raise ValueError("coordinates must have shape (frames, individuals, joints, dims)")
        expected = self.coordinates.shape[:3]
        if self.confidence.shape != expected or self.observed.shape != expected:
            raise ValueError("confidence and observed must match coordinates[:3]")
        if self.coordinates.shape[-1] not in (2, 3):
            raise ValueError("Pose coordinates must be 2D or 3D")
        if len(self.skeleton.joint_names) != self.coordinates.shape[2]:
            raise ValueError("Skeleton joint count does not match coordinates")
        if len(self.individual_names) != self.coordinates.shape[1]:
            raise ValueError("Individual names do not match the individual axis")
        if self.context is None:
            self.context = np.zeros((*self.coordinates.shape[:2], 0), dtype=np.float32)
        else:
            self.context = np.asarray(self.context, dtype=np.float32)
        if self.context.ndim != 3 or self.context.shape[:2] != self.coordinates.shape[:2]:
            raise ValueError("context must have shape (frames, individuals, features)")
        if len(self.context_names) != self.context.shape[-1]:
            raise ValueError("context_names must match the context feature axis")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive and finite")
        if self.frame_times is None:
            self.frame_times = np.arange(self.n_frames, dtype=np.float64) / self.fps
        else:
            self.frame_times = np.asarray(self.frame_times, dtype=np.float64)
            if self.frame_times.shape != (self.n_frames,):
                raise ValueError("frame_times must have one value per frame")
        finite_coordinates = np.isfinite(self.coordinates).all(axis=-1)
        self.observed &= finite_coordinates & np.isfinite(self.confidence)
        self.confidence = np.clip(np.nan_to_num(self.confidence, nan=0.0), 0.0, 1.0)

    @property
    def n_frames(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def n_individuals(self) -> int:
        return int(self.coordinates.shape[1])

    @property
    def n_joints(self) -> int:
        return int(self.coordinates.shape[2])

    @property
    def coordinate_dim(self) -> int:
        return int(self.coordinates.shape[3])

    @property
    def context_dim(self) -> int:
        assert self.context is not None
        return int(self.context.shape[2])

    def with_updates(self, **changes: Any) -> PoseSequence:
        return replace(self, **changes)

    def subset(self, start: int, stop: int) -> PoseSequence:
        if start < 0 or stop > self.n_frames or start >= stop:
            raise ValueError("Invalid frame interval")
        metadata = {**self.metadata, "parent_interval": [start, stop]}
        assert self.frame_times is not None
        assert self.context is not None
        return self.with_updates(
            coordinates=self.coordinates[start:stop].copy(),
            confidence=self.confidence[start:stop].copy(),
            observed=self.observed[start:stop].copy(),
            frame_times=self.frame_times[start:stop].copy(),
            context=self.context[start:stop].copy(),
            metadata=metadata,
        )

    def save(self, path: str | Path) -> None:
        """Save a portable, non-pickled NPZ artifact."""
        payload = {
            "fps": self.fps,
            "skeleton": self.skeleton.to_dict(),
            "individual_names": list(self.individual_names),
            "context_names": list(self.context_names),
            "source": self.source,
            "metadata": self.metadata,
        }
        assert self.frame_times is not None
        assert self.context is not None
        np.savez_compressed(
            Path(path),
            coordinates=self.coordinates,
            confidence=self.confidence,
            observed=self.observed,
            frame_times=self.frame_times,
            context=self.context,
            manifest=np.array(json.dumps(payload)),
        )

    @classmethod
    def load(cls, path: str | Path) -> PoseSequence:
        with np.load(Path(path), allow_pickle=False) as archive:
            payload = json.loads(str(archive["manifest"]))
            return cls(
                coordinates=archive["coordinates"],
                confidence=archive["confidence"],
                observed=archive["observed"],
                frame_times=archive["frame_times"],
                context=archive["context"],
                fps=float(payload["fps"]),
                skeleton=Skeleton.from_dict(payload["skeleton"]),
                individual_names=tuple(payload["individual_names"]),
                context_names=tuple(payload.get("context_names", [])),
                source=payload["source"],
                metadata=payload["metadata"],
            )
