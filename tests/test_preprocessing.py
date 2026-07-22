from __future__ import annotations

import numpy as np

from posemodel.preprocessing import NormalizeConfig, normalize_pose
from posemodel.schema import PoseSequence


def test_normalize_interpolates_short_gap_and_preserves_mask_provenance(
    pose_sequence: PoseSequence,
) -> None:
    normalized, transform = normalize_pose(pose_sequence, NormalizeConfig(interpolate_max_gap=3))
    assert normalized.observed[10:12, 0, 1].all()
    np.testing.assert_allclose(normalized.coordinates[:, 0, 0], 0.0, atol=1e-6)
    restored = transform.invert(normalized.coordinates)
    np.testing.assert_allclose(
        restored[normalized.observed],
        np.where(np.isfinite(pose_sequence.coordinates), pose_sequence.coordinates, restored)[
            normalized.observed
        ],
        atol=1e-5,
    )
