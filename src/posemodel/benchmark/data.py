"""Multi-recording benchmark windows and labels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from posemodel.benchmark.manifest import BenchmarkConfig, RecordingSpec
from posemodel.schema import PoseSequence, Skeleton


@dataclass(slots=True)
class WindowCollection:
    coordinates: NDArray[np.float32]
    confidence: NDArray[np.float32]
    observed: NDArray[np.bool_]
    context: NDArray[np.float32]
    labels: NDArray[np.object_]
    recording_ids: NDArray[np.str_]
    animal_ids: NDArray[np.str_]
    session_ids: NDArray[np.str_]
    starts: NDArray[np.int64]
    stops: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.coordinates.shape[0])

    def take(self, indices: NDArray[np.integer]) -> WindowCollection:
        return WindowCollection(
            **{field: getattr(self, field)[indices] for field in self.__dataclass_fields__}
        )

    @property
    def labeled(self) -> NDArray[np.bool_]:
        return np.asarray([label is not None for label in self.labels], dtype=bool)


@dataclass(frozen=True, slots=True)
class BenchmarkCorpus:
    train: WindowCollection
    validation: WindowCollection | None
    test: WindowCollection
    skeleton: Skeleton
    coordinate_dim: int
    context_dim: int


def _load_labels(recording: RecordingSpec) -> dict[int, object]:
    if recording.labels is None:
        return {}
    frame = pd.read_csv(recording.labels)
    if "label" not in frame.columns:
        raise ValueError(f"Label file must contain a 'label' column: {recording.labels}")
    frame_numbers = frame["frame"] if "frame" in frame.columns else frame.index
    return {
        int(frame_number): label
        for frame_number, label in zip(frame_numbers, frame["label"], strict=True)
        if pd.notna(label)
    }


def _windows_for_recording(
    recording: RecordingSpec,
    window_size: int,
    stride: int,
    maximum: int | None,
    random: np.random.Generator,
) -> WindowCollection:
    sequence = PoseSequence.load(recording.path)
    if sequence.n_frames < window_size:
        raise ValueError(f"Recording {recording.id} is shorter than one benchmark window")
    label_lookup = _load_labels(recording)
    starts = np.arange(0, sequence.n_frames - window_size + 1, stride, dtype=np.int64)
    if maximum is not None and len(starts) > maximum:
        starts = np.sort(random.choice(starts, size=maximum, replace=False))
    stops = starts + window_size
    assert sequence.context is not None
    coordinates = np.stack(
        [sequence.coordinates[start:stop] for start, stop in zip(starts, stops, strict=True)]
    )
    confidence = np.stack(
        [sequence.confidence[start:stop] for start, stop in zip(starts, stops, strict=True)]
    )
    observed = np.stack(
        [sequence.observed[start:stop] for start, stop in zip(starts, stops, strict=True)]
    )
    context = np.stack(
        [sequence.context[start:stop] for start, stop in zip(starts, stops, strict=True)]
    )
    centers = starts + window_size // 2
    labels = np.asarray([label_lookup.get(int(center)) for center in centers], dtype=object)
    count = len(starts)
    return WindowCollection(
        coordinates=coordinates,
        confidence=confidence,
        observed=observed,
        context=context,
        labels=labels,
        recording_ids=np.full(count, recording.id),
        animal_ids=np.full(count, recording.animal_id),
        session_ids=np.full(count, recording.session_id),
        starts=starts,
        stops=stops,
    )


def _concatenate(collections: list[WindowCollection]) -> WindowCollection:
    if not collections:
        raise ValueError("Cannot create an empty benchmark split")
    return WindowCollection(
        **{
            field: np.concatenate([getattr(collection, field) for collection in collections])
            for field in WindowCollection.__dataclass_fields__
        }
    )


def _limit(
    collection: WindowCollection, maximum: int | None, random: np.random.Generator
) -> WindowCollection:
    if maximum is None or len(collection) <= maximum:
        return collection
    indices = np.sort(random.choice(len(collection), size=maximum, replace=False))
    return collection.take(indices)


def load_corpus(config: BenchmarkConfig) -> BenchmarkCorpus:
    """Load prepared recordings and enforce one common pose contract."""
    sequences = [PoseSequence.load(recording.path) for recording in config.recordings]
    reference = sequences[0]
    for recording, sequence in zip(config.recordings, sequences, strict=True):
        if sequence.skeleton != reference.skeleton:
            raise ValueError(f"Skeleton mismatch in recording {recording.id}")
        if sequence.coordinate_dim != reference.coordinate_dim:
            raise ValueError(f"Coordinate dimension mismatch in recording {recording.id}")
        if sequence.context_dim != reference.context_dim:
            raise ValueError(f"Context dimension mismatch in recording {recording.id}")
    by_split: dict[str, list[WindowCollection]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    random = np.random.default_rng(config.seed)
    split_counts = {
        split: sum(recording.split == split for recording in config.recordings)
        for split in ("train", "validation", "test")
    }
    for recording in config.recordings:
        per_recording_maximum = (
            math.ceil(config.max_windows_per_split / split_counts[recording.split])
            if config.max_windows_per_split is not None
            else None
        )
        by_split[recording.split].append(
            _windows_for_recording(
                recording,
                config.window_size,
                config.stride,
                per_recording_maximum,
                random,
            )
        )
    train = _limit(_concatenate(by_split["train"]), config.max_windows_per_split, random)
    test = _limit(_concatenate(by_split["test"]), config.max_windows_per_split, random)
    validation = (
        _limit(_concatenate(by_split["validation"]), config.max_windows_per_split, random)
        if by_split["validation"]
        else None
    )
    if min(len(train), len(test)) < 3:
        raise ValueError("Benchmark train and test splits must each contain at least three windows")
    return BenchmarkCorpus(
        train,
        validation,
        test,
        reference.skeleton,
        reference.coordinate_dim,
        reference.context_dim,
    )
