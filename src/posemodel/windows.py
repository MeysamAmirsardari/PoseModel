"""Leakage-safe temporal splitting and model windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from posemodel.schema import PoseSequence


@dataclass(frozen=True, slots=True)
class FrameInterval:
    start: int
    stop: int
    split: Literal["train", "validation", "test"]

    @property
    def length(self) -> int:
        return self.stop - self.start


def contiguous_split(
    n_frames: int,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    gap: int = 0,
) -> tuple[FrameInterval, FrameInterval, FrameInterval]:
    """Split one recording before windowing, with guard gaps between partitions."""
    if n_frames <= 0 or gap < 0:
        raise ValueError("n_frames must be positive and gap non-negative")
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than one")
    usable = n_frames - 2 * gap
    validation = round(usable * validation_fraction)
    test = round(usable * test_fraction)
    train = usable - validation - test
    if min(train, validation, test) <= 0:
        raise ValueError("Recording is too short for the requested split and gaps")
    train_interval = FrameInterval(0, train, "train")
    validation_interval = FrameInterval(train + gap, train + gap + validation, "validation")
    test_start = validation_interval.stop + gap
    test_interval = FrameInterval(test_start, n_frames, "test")
    return train_interval, validation_interval, test_interval


@dataclass(frozen=True, slots=True)
class WindowIndex:
    start: int
    stop: int
    split: str


def build_window_index(
    intervals: tuple[FrameInterval, ...] | list[FrameInterval],
    *,
    window_size: int,
    stride: int,
) -> list[WindowIndex]:
    if window_size <= 1 or stride <= 0:
        raise ValueError("window_size must be > 1 and stride must be positive")
    windows: list[WindowIndex] = []
    for interval in intervals:
        for start in range(interval.start, interval.stop - window_size + 1, stride):
            windows.append(WindowIndex(start, start + window_size, interval.split))
    return windows


class PoseWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """A zero-copy logical view over windows of one normalized pose sequence."""

    def __init__(self, sequence: PoseSequence, windows: list[WindowIndex]) -> None:
        if not windows:
            raise ValueError("At least one window is required")
        self.sequence = sequence
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        selection = slice(window.start, window.stop)
        assert self.sequence.context is not None
        return {
            "coordinates": torch.from_numpy(self.sequence.coordinates[selection]),
            "confidence": torch.from_numpy(self.sequence.confidence[selection]),
            "observed": torch.from_numpy(self.sequence.observed[selection]),
            "context": torch.from_numpy(self.sequence.context[selection]),
            "start": torch.tensor(window.start, dtype=torch.long),
            "stop": torch.tensor(window.stop, dtype=torch.long),
        }

    def select_split(self, split: str) -> PoseWindowDataset:
        return PoseWindowDataset(self.sequence, [w for w in self.windows if w.split == split])


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
