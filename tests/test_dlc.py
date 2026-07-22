from __future__ import annotations

import numpy as np
import pandas as pd

from posemodel.io import load_dlc


def test_load_single_animal_dlc_csv(tmp_path) -> None:
    columns = pd.MultiIndex.from_product(
        [["scorer"], ["nose", "tail"], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    values = np.array(
        [
            [1.0, 2.0, 0.9, 3.0, 4.0, 0.8],
            [2.0, 3.0, 0.7, 4.0, 5.0, 0.6],
        ]
    )
    path = tmp_path / "dlc.csv"
    pd.DataFrame(values, columns=columns).to_csv(path)
    sequence = load_dlc(path, fps=25)
    assert sequence.coordinates.shape == (2, 1, 2, 2)
    assert sequence.skeleton.joint_names == ("nose", "tail")
    np.testing.assert_allclose(sequence.confidence[:, 0, 0], [0.9, 0.7])
