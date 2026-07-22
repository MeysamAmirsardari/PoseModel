from __future__ import annotations

import numpy as np

from posemodel.clustering import discover_states


def test_state_discovery_finds_separated_sticky_states() -> None:
    random = np.random.default_rng(42)
    first = random.normal(-4, 0.1, size=(30, 3))
    second = random.normal(4, 0.1, size=(30, 3))
    result = discover_states(
        np.concatenate([first, second]), min_states=2, max_states=2, stickiness=10
    )
    assert result.n_states == 2
    assert len(np.unique(result.labels[:30])) == 1
    assert len(np.unique(result.labels[30:])) == 1
    np.testing.assert_allclose(result.transition_matrix.sum(axis=1), 1.0)
