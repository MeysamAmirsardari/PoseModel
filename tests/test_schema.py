from __future__ import annotations

import numpy as np

from posemodel.schema import PoseSequence


def test_pose_sequence_round_trip(tmp_path, pose_sequence: PoseSequence) -> None:
    path = tmp_path / "sequence.npz"
    pose_sequence.save(path)
    loaded = PoseSequence.load(path)
    np.testing.assert_allclose(loaded.coordinates, pose_sequence.coordinates, equal_nan=True)
    np.testing.assert_array_equal(loaded.observed, pose_sequence.observed)
    assert loaded.skeleton == pose_sequence.skeleton
    assert loaded.source == "synthetic"


def test_subset_tracks_parent_interval(pose_sequence: PoseSequence) -> None:
    subset = pose_sequence.subset(5, 15)
    assert subset.n_frames == 10
    assert subset.metadata["parent_interval"] == [5, 15]
