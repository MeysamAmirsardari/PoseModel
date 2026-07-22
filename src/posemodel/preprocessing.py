"""Confidence-aware pose normalization and interpolation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from posemodel.schema import PoseSequence


@dataclass(frozen=True, slots=True)
class NormalizeConfig:
    confidence_threshold: float = 0.2
    interpolate_max_gap: int = 5
    center: bool = True
    scale: bool = True
    minimum_scale: float = 1e-4


@dataclass(frozen=True, slots=True)
class NormalizationTransform:
    centers: NDArray[np.float32]
    scales: NDArray[np.float32]
    config: NormalizeConfig

    def invert(self, coordinates: NDArray[np.floating]) -> NDArray[np.float32]:
        values = np.asarray(coordinates, dtype=np.float32)
        return values * self.scales[..., None, None] + self.centers[:, :, None, :]


def _interpolate_short_gaps(
    coordinates: NDArray[np.float32], observed: NDArray[np.bool_], max_gap: int
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    output = coordinates.copy()
    effective = observed.copy()
    if max_gap <= 0:
        return output, effective
    n_frames, n_individuals, n_joints, n_dimensions = output.shape
    for individual in range(n_individuals):
        for joint in range(n_joints):
            valid = np.flatnonzero(observed[:, individual, joint])
            for left, right in pairwise(valid):
                gap = right - left - 1
                if gap <= 0 or gap > max_gap:
                    continue
                fractions = np.arange(1, gap + 1, dtype=np.float32) / (gap + 1)
                delta = output[right, individual, joint] - output[left, individual, joint]
                for dimension in range(n_dimensions):
                    output[left + 1 : right, individual, joint, dimension] = (
                        output[left, individual, joint, dimension] + fractions * delta[dimension]
                    )
                effective[left + 1 : right, individual, joint] = True
    return output, effective


def _frame_centers(
    coordinates: NDArray[np.float32],
    observed: NDArray[np.bool_],
    root: int | None,
) -> NDArray[np.float32]:
    masked = np.where(observed[..., None], coordinates, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        centers = np.nanmedian(masked, axis=2)
    if root is not None:
        root_valid = observed[:, :, root]
        root_values = coordinates[:, :, root]
        centers = np.where(root_valid[..., None], root_values, centers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        global_center = np.nanmedian(masked, axis=(0, 2), keepdims=False)
    centers = np.where(np.isfinite(centers), centers, global_center[None, ...])
    return np.asarray(np.nan_to_num(centers, nan=0.0), dtype=np.float32)


def _frame_scales(
    centered: NDArray[np.float32],
    observed: NDArray[np.bool_],
    edges: tuple[tuple[int, int], ...],
    minimum: float,
) -> NDArray[np.float32]:
    if edges:
        lengths: list[NDArray[np.float32]] = []
        for source, target in edges:
            valid = observed[:, :, source] & observed[:, :, target]
            length = np.linalg.norm(centered[:, :, source] - centered[:, :, target], axis=-1)
            lengths.append(np.where(valid, length, np.nan).astype(np.float32))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw = np.nanmedian(np.stack(lengths, axis=-1), axis=-1)
    else:
        radii = np.linalg.norm(centered, axis=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw = np.nanmedian(np.where(observed, radii, np.nan), axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fallback = float(np.nanmedian(raw))
    if not np.isfinite(fallback) or fallback < minimum:
        fallback = 1.0
    return np.where(np.isfinite(raw) & (raw >= minimum), raw, fallback).astype(np.float32)


def normalize_pose(
    sequence: PoseSequence, config: NormalizeConfig | None = None
) -> tuple[PoseSequence, NormalizationTransform]:
    """Mask, interpolate, center, and scale a recording without hiding missingness."""
    config = config or NormalizeConfig()
    if not 0 <= config.confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")
    observed = sequence.observed & (sequence.confidence >= config.confidence_threshold)
    coordinates, effective = _interpolate_short_gaps(
        sequence.coordinates, observed, config.interpolate_max_gap
    )
    centers = _frame_centers(coordinates, effective, sequence.skeleton.root)
    centered = coordinates - centers[:, :, None, :] if config.center else coordinates.copy()
    scales = _frame_scales(centered, effective, sequence.skeleton.edges, config.minimum_scale)
    normalized = centered / scales[:, :, None, None] if config.scale else centered
    normalized = np.where(effective[..., None], normalized, 0.0).astype(np.float32)
    center_velocity = np.zeros_like(centers)
    center_velocity[1:] = (centers[1:] - centers[:-1]) * sequence.fps
    center_velocity /= scales[..., None]
    log_scale = np.log(np.maximum(scales, config.minimum_scale))
    scale_velocity = np.zeros_like(scales)
    scale_velocity[1:] = (log_scale[1:] - log_scale[:-1]) * sequence.fps
    context = np.concatenate([center_velocity, scale_velocity[..., None]], axis=-1).astype(
        np.float32
    )
    axes = ("x", "y", "z")[: sequence.coordinate_dim]
    context_names = tuple([f"global_v{axis}" for axis in axes] + ["log_scale_velocity"])
    transform = NormalizationTransform(
        centers=centers if config.center else np.zeros_like(centers),
        scales=scales if config.scale else np.ones_like(scales),
        config=config,
    )
    metadata = {
        **sequence.metadata,
        "normalization": {
            "confidence_threshold": config.confidence_threshold,
            "interpolate_max_gap": config.interpolate_max_gap,
            "center": config.center,
            "scale": config.scale,
        },
    }
    return (
        sequence.with_updates(
            coordinates=normalized,
            observed=effective,
            confidence=np.where(effective, sequence.confidence, 0.0),
            context=context,
            context_names=context_names,
            metadata=metadata,
        ),
        transform,
    )
