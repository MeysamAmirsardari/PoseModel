from __future__ import annotations

import numpy as np
import pytest

from posemodel.schema import PoseSequence, Skeleton


@pytest.fixture()
def skeleton() -> Skeleton:
    return Skeleton.from_names(
        ["root", "left", "right"],
        edges=[("root", "left"), ("root", "right")],
        root="root",
        left_right_pairs=[("left", "right")],
    )


@pytest.fixture()
def pose_sequence(skeleton: Skeleton) -> PoseSequence:
    frames = 48
    time = np.arange(frames, dtype=np.float32) / 30.0
    coordinates = np.zeros((frames, 1, 3, 2), dtype=np.float32)
    coordinates[:, 0, 0] = np.stack([time, np.zeros_like(time)], axis=-1)
    coordinates[:, 0, 1] = coordinates[:, 0, 0] + np.array([0.0, 1.0])
    coordinates[:, 0, 2] = coordinates[:, 0, 0] + np.array([0.0, -1.0])
    confidence = np.ones((frames, 1, 3), dtype=np.float32)
    observed = np.ones_like(confidence, dtype=bool)
    coordinates[10:12, 0, 1] = np.nan
    confidence[10:12, 0, 1] = 0.05
    observed[10:12, 0, 1] = False
    return PoseSequence(
        coordinates,
        confidence,
        observed,
        fps=30.0,
        skeleton=skeleton,
        source="synthetic",
    )
