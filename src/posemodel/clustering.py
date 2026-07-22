"""Stable state discovery with BIC model selection and sticky temporal decoding."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class StateSequence:
    labels: NDArray[np.int64]
    probabilities: NDArray[np.float64]
    transition_matrix: NDArray[np.float64]
    n_states: int
    bic: float


def _transition_matrix(
    labels: NDArray[np.int64], n_states: int, stickiness: float
) -> NDArray[np.float64]:
    counts = np.ones((n_states, n_states), dtype=np.float64)
    counts[np.diag_indices(n_states)] += stickiness
    for source, target in pairwise(labels):
        counts[source, target] += 1.0
    return np.asarray(counts / counts.sum(axis=1, keepdims=True), dtype=np.float64)


def _viterbi(
    log_emission: NDArray[np.float64], transition: NDArray[np.float64]
) -> NDArray[np.int64]:
    n_samples, n_states = log_emission.shape
    score = np.empty((n_samples, n_states), dtype=np.float64)
    backpointer = np.zeros((n_samples, n_states), dtype=np.int64)
    score[0] = log_emission[0] - np.log(n_states)
    log_transition = np.log(np.clip(transition, 1e-12, 1.0))
    for time in range(1, n_samples):
        candidates = score[time - 1, :, None] + log_transition
        backpointer[time] = candidates.argmax(axis=0)
        score[time] = candidates.max(axis=0) + log_emission[time]
    states = np.empty(n_samples, dtype=np.int64)
    states[-1] = score[-1].argmax()
    for time in range(n_samples - 1, 0, -1):
        states[time - 1] = backpointer[time, states[time]]
    return states


def discover_states(
    embeddings: NDArray[np.floating],
    *,
    min_states: int = 2,
    max_states: int = 20,
    stickiness: float = 20.0,
    seed: int = 42,
) -> StateSequence:
    """Select a full-covariance GMM by BIC, then apply sticky temporal decoding."""
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("embeddings must be a 2D array with at least three samples")
    upper = min(max_states, values.shape[0] - 1)
    if min_states > upper:
        raise ValueError("Not enough samples for the requested state range")
    normalized = StandardScaler().fit_transform(values)
    candidates: list[tuple[float, GaussianMixture]] = []
    for n_states in range(min_states, upper + 1):
        model = GaussianMixture(
            n_components=n_states,
            covariance_type="full",
            reg_covar=1e-5,
            n_init=5,
            init_params="random_from_data",
            random_state=seed,
        ).fit(normalized)
        candidates.append((model.bic(normalized), model))
    bic, best = min(candidates, key=lambda item: item[0])
    probabilities = best.predict_proba(normalized)
    initial = probabilities.argmax(axis=1).astype(np.int64)
    transition = _transition_matrix(initial, best.n_components, stickiness)
    labels = _viterbi(np.log(np.clip(probabilities, 1e-12, 1.0)), transition)
    return StateSequence(labels, probabilities, transition, best.n_components, float(bic))
