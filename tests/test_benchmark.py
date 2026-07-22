from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from posemodel.benchmark import BenchmarkConfig, run_benchmark, write_benchmark_index
from posemodel.models import GraphMotionMAE, ModelConfig
from posemodel.preprocessing import normalize_pose
from posemodel.schema import PoseSequence
from posemodel.training import TrainingConfig, save_checkpoint


def _recording(tmp_path, pose_sequence: PoseSequence, name: str, offset: float) -> tuple[str, str]:
    repeated = np.tile(pose_sequence.coordinates, (2, 1, 1, 1))[:64]
    confidence = np.ones(repeated.shape[:-1], dtype=np.float32)
    observed = np.ones_like(confidence, dtype=bool)
    sequence = pose_sequence.with_updates(
        coordinates=repeated + offset,
        confidence=confidence,
        observed=observed,
        context=None,
        context_names=(),
        frame_times=None,
        source=name,
    )
    normalized, _ = normalize_pose(sequence)
    sequence_path = tmp_path / f"{name}.npz"
    normalized.save(sequence_path)
    labels_path = tmp_path / f"{name}.csv"
    pd.DataFrame(
        {"frame": np.arange(64), "label": ["behavior-a"] * 32 + ["behavior-b"] * 32}
    ).to_csv(labels_path, index=False)
    return sequence_path.name, labels_path.name


def test_end_to_end_benchmark(tmp_path, pose_sequence: PoseSequence) -> None:
    recordings = []
    for index, split in enumerate(("train", "validation", "test")):
        sequence, labels = _recording(tmp_path, pose_sequence, f"recording-{index}", index * 0.1)
        recordings.append(
            {
                "id": f"recording-{index}",
                "path": sequence,
                "labels": labels,
                "animal_id": f"animal-{index}",
                "session_id": f"session-{index}",
                "split": split,
            }
        )
    prepared = PoseSequence.load(tmp_path / "recording-0.npz")
    model_config = ModelConfig(
        coordinate_dim=prepared.coordinate_dim,
        context_dim=prepared.context_dim,
        hidden_dim=16,
        latent_dim=4,
        depth=1,
        decoder_depth=1,
        num_heads=4,
        max_window_size=8,
        dropout=0.0,
    )
    save_checkpoint(
        tmp_path / "model.pt",
        GraphMotionMAE(prepared.skeleton, model_config),
        TrainingConfig(window_size=8, stride=4, model=model_config),
    )
    random = np.random.default_rng(5)
    np.savez_compressed(
        tmp_path / "external.npz",
        train=random.normal(size=(15, 4)),
        validation=random.normal(size=(15, 4)),
        test=random.normal(size=(15, 4)),
    )
    manifest = {
        "version": 1,
        "name": "synthetic",
        "window_size": 8,
        "stride": 4,
        "split_unit": "animal",
        "few_shot_fractions": [0.5, 1.0],
        "recordings": recordings,
        "candidates": [
            {"name": "kinematics", "kind": "kinematics"},
            {"name": "pca", "kind": "pca", "settings": {"latent_dim": 4}},
            {
                "name": "tcn",
                "kind": "tcn",
                "settings": {
                    "latent_dim": 4,
                    "hidden_dim": 8,
                    "epochs": 1,
                    "batch_size": 8,
                    "device": "cpu",
                },
            },
            {
                "name": "posemodel",
                "kind": "posemodel",
                "settings": {"checkpoint": "model.pt", "batch_size": 8, "device": "cpu"},
            },
            {
                "name": "posemodel-trained",
                "kind": "posemodel",
                "settings": {
                    "model": {
                        "hidden_dim": 16,
                        "latent_dim": 4,
                        "depth": 1,
                        "decoder_depth": 1,
                        "num_heads": 4,
                        "dropout": 0.0,
                    },
                    "epochs": 1,
                    "patience": 1,
                    "batch_size": 8,
                    "device": "cpu",
                },
            },
            {
                "name": "external",
                "kind": "imported",
                "settings": {"path": "external.npz"},
            },
        ],
        "corruptions": [{"name": "missing", "kind": "missing", "magnitude": 0.3}],
    }
    manifest_path = tmp_path / "benchmark.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    output = tmp_path / "output"
    config = BenchmarkConfig.from_yaml(manifest_path)
    index_path = write_benchmark_index(config, tmp_path / "window-index.csv")
    report = run_benchmark(config, output)
    assert set(report["candidates"]) == {
        "kinematics",
        "pca",
        "tcn",
        "posemodel",
        "posemodel-trained",
        "external",
    }
    assert "macro_f1" in report["candidates"]["pca"]["evaluation"]["supervised"]
    assert "missing" in report["candidates"]["kinematics"]["robustness"]
    assert report["candidates"]["external"]["robustness"] == {}
    assert (output / "model-posemodel-trained.pt").exists()
    assert (output / "benchmark.json").exists()
    assert (output / "benchmark.html").exists()
    assert len(pd.read_csv(index_path)) == 45


def test_manifest_rejects_animal_leakage(tmp_path, pose_sequence: PoseSequence) -> None:
    sequence, _ = _recording(tmp_path, pose_sequence, "recording", 0.0)
    manifest = {
        "version": 1,
        "recordings": [
            {
                "id": "train",
                "path": sequence,
                "animal_id": "same-animal",
                "session_id": "one",
                "split": "train",
            },
            {
                "id": "test",
                "path": sequence,
                "animal_id": "same-animal",
                "session_id": "two",
                "split": "test",
            },
        ],
        "candidates": [{"kind": "pca"}],
    }
    path = tmp_path / "leaky.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="groups cross benchmark splits"):
        BenchmarkConfig.from_yaml(path)
