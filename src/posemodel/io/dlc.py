"""DeepLabCut CSV and HDF5 readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from posemodel.schema import PoseSequence, Skeleton


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        frame = pd.read_hdf(path)
        if not isinstance(frame.columns, pd.MultiIndex):
            raise ValueError("DeepLabCut HDF5 columns must be a MultiIndex")
        return frame
    if path.suffix.lower() != ".csv":
        raise ValueError("DeepLabCut input must be CSV or HDF5")
    for levels in (4, 3):
        try:
            frame = pd.read_csv(path, header=list(range(levels)), index_col=0)
        except (ValueError, pd.errors.ParserError):
            continue
        values = {str(value).lower() for value in frame.columns.get_level_values(-1)}
        if {"x", "y"}.issubset(values):
            return frame
    raise ValueError("Could not identify DeepLabCut coordinate header levels")


def _drop_scorer(columns: pd.MultiIndex) -> pd.MultiIndex:
    if columns.nlevels >= 3 and len(columns.get_level_values(0).unique()) == 1:
        return columns.droplevel(0)
    return columns


def load_dlc(
    path: str | Path,
    *,
    fps: float,
    skeleton: Skeleton | None = None,
    source: str | None = None,
) -> PoseSequence:
    """Load standard single- or multi-animal DeepLabCut output."""
    input_path = Path(path)
    frame = _read_table(input_path)
    frame.columns = _drop_scorer(frame.columns)
    if frame.columns.nlevels == 2:
        individuals: tuple[str, ...] = ("individual_0",)
        bodypart_level, coordinate_level = 0, 1
    elif frame.columns.nlevels == 3:
        individuals = tuple(dict.fromkeys(map(str, frame.columns.get_level_values(0))))
        bodypart_level, coordinate_level = 1, 2
    else:
        raise ValueError(f"Unsupported DeepLabCut column depth: {frame.columns.nlevels}")

    joint_names = tuple(dict.fromkeys(map(str, frame.columns.get_level_values(bodypart_level))))
    if skeleton is None:
        skeleton = Skeleton(joint_names=joint_names)
    elif skeleton.joint_names != joint_names:
        raise ValueError("Provided skeleton joint order does not match the DeepLabCut file")

    coordinate_labels = {
        str(value).lower() for value in frame.columns.get_level_values(coordinate_level)
    }
    dimensions = tuple(label for label in ("x", "y", "z") if label in coordinate_labels)
    if dimensions not in (("x", "y"), ("x", "y", "z")):
        raise ValueError(f"Unsupported coordinate labels: {sorted(coordinate_labels)}")

    shape = (len(frame), len(individuals), len(joint_names))
    coordinates = np.full((*shape, len(dimensions)), np.nan, dtype=np.float32)
    confidence = np.ones(shape, dtype=np.float32)
    for individual_index, individual in enumerate(individuals):
        for joint_index, joint in enumerate(joint_names):
            prefix: tuple[Any, ...] = (
                (joint,) if frame.columns.nlevels == 2 else (individual, joint)
            )
            for dimension_index, dimension in enumerate(dimensions):
                coordinates[:, individual_index, joint_index, dimension_index] = pd.to_numeric(
                    frame[(*prefix, dimension)], errors="coerce"
                ).to_numpy(dtype=np.float32)
            likelihood_key = (*prefix, "likelihood")
            if likelihood_key in frame.columns:
                confidence[:, individual_index, joint_index] = pd.to_numeric(
                    frame[likelihood_key], errors="coerce"
                ).to_numpy(dtype=np.float32)

    observed = np.isfinite(coordinates).all(axis=-1) & np.isfinite(confidence)
    return PoseSequence(
        coordinates=coordinates,
        confidence=confidence,
        observed=observed,
        fps=fps,
        skeleton=skeleton,
        individual_names=individuals,
        source=source or str(input_path.resolve()),
        metadata={"format": "deeplabcut", "frame_index": list(map(str, frame.index))},
    )
