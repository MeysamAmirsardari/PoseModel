"""Common downstream, clustering, robustness, and efficiency evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

from posemodel.benchmark.data import WindowCollection
from posemodel.benchmark.manifest import CorruptionSpec


@dataclass(slots=True)
class EvaluationState:
    scaler: StandardScaler
    classifier: LogisticRegression | None
    label_encoder: LabelEncoder | None
    clean_test: NDArray[np.float64]


def _finite_embeddings(values: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} embeddings must be a finite 2D array")
    return result


def _continuity(values: NDArray[np.float64], windows: WindowCollection) -> float:
    scores: list[float] = []
    norms = np.linalg.norm(values, axis=1).clip(1e-12)
    for recording in np.unique(windows.recording_ids):
        indices = np.flatnonzero(windows.recording_ids == recording)
        order = indices[np.argsort(windows.starts[indices])]
        if len(order) < 2:
            continue
        left, right = values[order[:-1]], values[order[1:]]
        scores.extend(np.sum(left * right, axis=1) / (norms[order[:-1]] * norms[order[1:]]))
    return float(np.mean(scores)) if scores else float("nan")


def _representation_metrics(
    values: NDArray[np.float64], windows: WindowCollection
) -> dict[str, Any]:
    covariance = np.cov(values, rowvar=False)
    eigenvalues = np.atleast_1d(np.linalg.eigvalsh(np.atleast_2d(covariance))).clip(0)
    effective_rank = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum().clip(1e-12))
    return {
        "dimensions": int(values.shape[1]),
        "samples": int(values.shape[0]),
        "mean_feature_std": float(values.std(axis=0).mean()),
        "effective_rank": effective_rank,
        "temporal_cosine_continuity": _continuity(values, windows),
    }


def _labeled(
    values: NDArray[np.float64], windows: WindowCollection
) -> tuple[NDArray[np.float64], NDArray[np.object_]]:
    mask = windows.labeled
    return values[mask], windows.labels[mask]


def _classification_scores(
    classifier: LogisticRegression,
    values: NDArray[np.float64],
    labels: NDArray[np.int64],
    n_classes: int,
) -> dict[str, float]:
    predictions = classifier.predict(values)
    probabilities = classifier.predict_proba(values)
    binary = label_binarize(labels, classes=np.arange(n_classes))
    if n_classes == 2:
        average_precision = average_precision_score(labels, probabilities[:, 1])
    else:
        average_precision = average_precision_score(binary, probabilities, average="macro")
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "macro_average_precision": float(average_precision),
    }


def _knn_predict(
    train: NDArray[np.float64], labels: NDArray[np.int64], test: NDArray[np.float64], k: int
) -> NDArray[np.int64]:
    """Small deterministic NumPy kNN avoids platform-specific native threading behavior."""
    predictions = np.empty(len(test), dtype=np.int64)
    for index, sample in enumerate(test):
        distances = np.square(train - sample).sum(axis=1)
        neighbors = np.argpartition(distances, k - 1)[:k]
        predictions[index] = np.bincount(labels[neighbors]).argmax()
    return predictions


def _fit_probe(
    train: NDArray[np.float64],
    train_labels: NDArray[np.object_],
    validation: NDArray[np.float64] | None,
    validation_labels: NDArray[np.object_] | None,
    test: NDArray[np.float64],
    test_labels: NDArray[np.object_],
    seed: int,
) -> tuple[dict[str, Any], LogisticRegression, LabelEncoder]:
    encoder = LabelEncoder().fit(train_labels)
    y_train = encoder.transform(train_labels)
    known_test = np.isin(test_labels, encoder.classes_)
    test, test_labels = test[known_test], test_labels[known_test]
    if len(np.unique(y_train)) < 2 or not len(test):
        raise ValueError("Linear probe requires at least two training classes represented in test")
    y_test = encoder.transform(test_labels)
    best_c, best_score = 1.0, -np.inf
    if validation is not None and validation_labels is not None and len(validation_labels):
        known_validation = np.isin(validation_labels, encoder.classes_)
        x_validation = validation[known_validation]
        y_validation = encoder.transform(validation_labels[known_validation])
        for regularization in (0.01, 0.1, 1.0, 10.0):
            candidate = LogisticRegression(
                C=regularization,
                max_iter=2_000,
                class_weight="balanced",
                random_state=seed,
            ).fit(train, y_train)
            score = f1_score(
                y_validation,
                candidate.predict(x_validation),
                average="macro",
                zero_division=0,
            )
            if score > best_score:
                best_c, best_score = regularization, float(score)
    classifier = LogisticRegression(
        C=best_c,
        max_iter=2_000,
        class_weight="balanced",
        random_state=seed,
    ).fit(train, y_train)
    metrics: dict[str, Any] = _classification_scores(
        classifier, test, y_test, len(encoder.classes_)
    )
    metrics["selected_c"] = best_c
    metrics["classes"] = list(map(str, encoder.classes_))
    metrics["knn_accuracy"] = float(
        accuracy_score(y_test, _knn_predict(train, y_train, test, min(5, len(train))))
    )
    return metrics, classifier, encoder


def _few_shot(
    train: NDArray[np.float64],
    labels: NDArray[np.object_],
    test: NDArray[np.float64],
    test_labels: NDArray[np.object_],
    fractions: tuple[float, ...],
    seed: int,
) -> dict[str, dict[str, float]]:
    encoder = LabelEncoder().fit(labels)
    y_train = encoder.transform(labels)
    known = np.isin(test_labels, encoder.classes_)
    test, test_labels = test[known], test_labels[known]
    y_test = encoder.transform(test_labels)
    output: dict[str, dict[str, float]] = {}
    for fraction in fractions:
        scores: list[float] = []
        for repeat in range(3):
            random = np.random.default_rng(seed + repeat)
            selected: list[int] = []
            for label in np.unique(y_train):
                members = np.flatnonzero(y_train == label)
                count = max(1, round(len(members) * fraction))
                selected.extend(random.choice(members, size=count, replace=False).tolist())
            model = LogisticRegression(
                max_iter=2_000,
                class_weight="balanced",
                random_state=seed + repeat,
            ).fit(train[selected], y_train[selected])
            scores.append(
                float(f1_score(y_test, model.predict(test), average="macro", zero_division=0))
            )
        output[f"{fraction:.2f}"] = {
            "macro_f1_mean": float(np.mean(scores)),
            "macro_f1_std": float(np.std(scores)),
        }
    return output


def _clustering(
    train: NDArray[np.float64],
    test: NDArray[np.float64],
    windows: WindowCollection,
    seed: int,
) -> dict[str, Any]:
    upper = min(12, max(2, int(np.sqrt(len(train)))), len(train) - 1)
    candidates: list[tuple[float, GaussianMixture]] = []
    for states in range(2, upper + 1):
        model = GaussianMixture(
            n_components=states,
            reg_covar=1e-5,
            n_init=3,
            init_params="random_from_data",
            random_state=seed,
        ).fit(train)
        candidates.append((float(model.bic(train)), model))
    bic, best = min(candidates, key=lambda item: item[0])
    predictions = best.predict(test)
    result: dict[str, Any] = {
        "states": int(best.n_components),
        "bic": bic,
        "animal_nmi": float(normalized_mutual_info_score(windows.animal_ids, predictions)),
        "session_nmi": float(normalized_mutual_info_score(windows.session_ids, predictions)),
    }
    if windows.labeled.any():
        result["label_ami"] = float(
            adjusted_mutual_info_score(
                windows.labels[windows.labeled], predictions[windows.labeled]
            )
        )
        result["label_ari"] = float(
            adjusted_rand_score(windows.labels[windows.labeled], predictions[windows.labeled])
        )
    runs = []
    for offset in range(5):
        model = GaussianMixture(
            n_components=best.n_components,
            reg_covar=1e-5,
            n_init=1,
            init_params="random_from_data",
            random_state=seed + offset,
        ).fit(train)
        runs.append(model.predict(test))
    stability = [adjusted_mutual_info_score(left, right) for left, right in combinations(runs, 2)]
    result["seed_stability_ami"] = float(np.mean(stability))
    return result


def evaluate_representation(
    train_embeddings: NDArray[np.floating],
    validation_embeddings: NDArray[np.floating] | None,
    test_embeddings: NDArray[np.floating],
    train_windows: WindowCollection,
    validation_windows: WindowCollection | None,
    test_windows: WindowCollection,
    fractions: tuple[float, ...],
    seed: int,
) -> tuple[dict[str, Any], EvaluationState]:
    train_raw = _finite_embeddings(train_embeddings, "train")
    test_raw = _finite_embeddings(test_embeddings, "test")
    validation_raw = (
        _finite_embeddings(validation_embeddings, "validation")
        if validation_embeddings is not None
        else None
    )
    scaler = StandardScaler().fit(train_raw)
    train = scaler.transform(train_raw)
    test = scaler.transform(test_raw)
    validation = scaler.transform(validation_raw) if validation_raw is not None else None
    report: dict[str, Any] = {
        "representation": _representation_metrics(test, test_windows),
        "clustering": _clustering(train, test, test_windows, seed),
    }
    classifier: LogisticRegression | None = None
    label_encoder: LabelEncoder | None = None
    x_train, y_train = _labeled(train, train_windows)
    x_test, y_test = _labeled(test, test_windows)
    if len(y_train) and len(y_test):
        if validation is not None and validation_windows is not None:
            x_validation, y_validation = _labeled(validation, validation_windows)
        else:
            x_validation, y_validation = None, None
        supervised, classifier, label_encoder = _fit_probe(
            x_train, y_train, x_validation, y_validation, x_test, y_test, seed
        )
        supervised["few_shot"] = _few_shot(x_train, y_train, x_test, y_test, fractions, seed)
        report["supervised"] = supervised
    return report, EvaluationState(
        scaler,
        classifier,
        label_encoder,
        test,
    )


def corrupt(windows: WindowCollection, corruption: CorruptionSpec, seed: int) -> WindowCollection:
    random = np.random.default_rng(seed)
    result = windows.take(np.arange(len(windows)))
    if corruption.kind == "missing":
        missing = random.random(result.observed.shape) < corruption.magnitude
    elif corruption.kind == "joint_dropout":
        joint_missing = (
            random.random((len(result), 1, result.observed.shape[2], result.observed.shape[3]))
            < corruption.magnitude
        )
        missing = np.broadcast_to(joint_missing, result.observed.shape)
    elif corruption.kind == "jitter":
        valid_coordinates = result.coordinates[result.observed]
        scale = float(np.std(valid_coordinates)) if len(valid_coordinates) else 1.0
        noise = random.normal(0, corruption.magnitude * scale, result.coordinates.shape)
        result.coordinates = (
            result.coordinates + noise.astype(np.float32) * result.observed[..., None]
        )
        return result
    else:
        raise ValueError(f"Unsupported corruption: {corruption.kind}")
    result.observed &= ~missing
    result.confidence = np.where(result.observed, result.confidence, 0.0)
    result.coordinates = np.where(result.observed[..., None], result.coordinates, 0.0)
    return result


def evaluate_corruption(
    clean_embeddings: NDArray[np.floating],
    corrupted_embeddings: NDArray[np.floating],
    windows: WindowCollection,
    state: EvaluationState,
) -> dict[str, float]:
    clean = state.scaler.transform(_finite_embeddings(clean_embeddings, "clean test"))
    corrupted = state.scaler.transform(_finite_embeddings(corrupted_embeddings, "corrupted test"))
    cosine = np.sum(clean * corrupted, axis=1) / (
        np.linalg.norm(clean, axis=1).clip(1e-12) * np.linalg.norm(corrupted, axis=1).clip(1e-12)
    )
    result = {"latent_cosine": float(np.mean(cosine))}
    if state.classifier is not None and state.label_encoder is not None and windows.labeled.any():
        values = corrupted[windows.labeled]
        labels = windows.labels[windows.labeled]
        known = np.isin(labels, state.label_encoder.classes_)
        encoded = state.label_encoder.transform(labels[known])
        result["macro_f1"] = float(
            f1_score(
                encoded,
                state.classifier.predict(values[known]),
                average="macro",
                zero_division=0,
            )
        )
    return result
