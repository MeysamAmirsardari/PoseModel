"""Correct, missingness-aware pose and representation metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import adjusted_mutual_info_score


def mpjpe(
    predicted: NDArray[np.floating],
    target: NDArray[np.floating],
    observed: NDArray[np.bool_],
) -> float:
    predicted_values = np.asarray(predicted)
    target_values = np.asarray(target)
    valid = np.asarray(observed, dtype=bool)
    if predicted_values.shape != target_values.shape or predicted_values.shape[:-1] != valid.shape:
        raise ValueError("Predictions, targets, and observed mask have incompatible shapes")
    if not valid.any():
        return float("nan")
    return float(np.linalg.norm(predicted_values - target_values, axis=-1)[valid].mean())


def clustering_stability(label_runs: list[NDArray[np.integer]]) -> float:
    if len(label_runs) < 2:
        raise ValueError("At least two label runs are required")
    scores = [
        adjusted_mutual_info_score(label_runs[left], label_runs[right])
        for left in range(len(label_runs))
        for right in range(left + 1, len(label_runs))
    ]
    return float(np.mean(scores))
