"""End-to-end benchmark orchestration."""

from __future__ import annotations

import logging
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from posemodel import __version__
from posemodel.benchmark.candidates import create_candidate
from posemodel.benchmark.data import load_corpus
from posemodel.benchmark.evaluation import (
    corrupt,
    evaluate_corruption,
    evaluate_representation,
)
from posemodel.benchmark.manifest import BenchmarkConfig
from posemodel.benchmark.report import write_report

logger = logging.getLogger(__name__)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "candidate"


def write_benchmark_index(config: BenchmarkConfig, path: str | Path) -> Path:
    """Export the exact ordered windows required by external embedding methods."""
    corpus = load_corpus(config)
    tables = []
    for split, windows in (
        ("train", corpus.train),
        ("validation", corpus.validation),
        ("test", corpus.test),
    ):
        if windows is None:
            continue
        tables.append(
            pd.DataFrame(
                {
                    "split": split,
                    "row": np.arange(len(windows)),
                    "recording_id": windows.recording_ids,
                    "animal_id": windows.animal_ids,
                    "session_id": windows.session_ids,
                    "start": windows.starts,
                    "stop": windows.stops,
                    "label": windows.labels,
                }
            )
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(output, index=False)
    return output


def _acceptance_gates(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    for name, result in candidates.items():
        stability = result["evaluation"]["clustering"]["seed_stability_ami"]
        gates[f"{name}: cluster stability >= 0.8"] = {
            "passed": bool(stability >= 0.8),
            "value": stability,
        }
        robustness = result.get("robustness", {})
        clean_f1 = result["evaluation"].get("supervised", {}).get("macro_f1")
        for corruption, metrics in robustness.items():
            if clean_f1 is not None and "macro_f1" in metrics and clean_f1 > 0:
                relative_drop = (clean_f1 - metrics["macro_f1"]) / clean_f1
                gates[f"{name}: {corruption} macro F1 drop <= 20%"] = {
                    "passed": bool(relative_drop <= 0.2),
                    "value": relative_drop,
                }
    posemodels = [value for value in candidates.values() if value["kind"] == "posemodel"]
    baselines = [
        value
        for value in candidates.values()
        if value["kind"] in {"pca", "kinematics", "tcn"} and "supervised" in value["evaluation"]
    ]
    if posemodels and baselines and "supervised" in posemodels[0]["evaluation"]:
        model_score = posemodels[0]["evaluation"]["supervised"]["macro_f1"]
        best_baseline = max(item["evaluation"]["supervised"]["macro_f1"] for item in baselines)
        gates["PoseModel beats the best built-in baseline"] = {
            "passed": bool(model_score > best_baseline),
            "posemodel_macro_f1": model_score,
            "best_baseline_macro_f1": best_baseline,
        }
    return gates


def run_benchmark(
    config: BenchmarkConfig,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Run all candidates against one immutable corpus and write comparable artifacts."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(config)
    candidate_reports: dict[str, dict[str, Any]] = {}
    for spec in config.candidates:
        logger.info("Fitting benchmark candidate %s (%s)", spec.name, spec.kind)
        candidate = create_candidate(spec)
        candidate.fit(corpus)
        started = time.perf_counter()
        train_embeddings = candidate.transform(corpus.train, "train")
        train_seconds = time.perf_counter() - started
        validation_embeddings = None
        validation_seconds = 0.0
        if corpus.validation is not None:
            started = time.perf_counter()
            validation_embeddings = candidate.transform(corpus.validation, "validation")
            validation_seconds = time.perf_counter() - started
        started = time.perf_counter()
        test_embeddings = candidate.transform(corpus.test, "test")
        test_seconds = time.perf_counter() - started
        evaluation, state = evaluate_representation(
            train_embeddings,
            validation_embeddings,
            test_embeddings,
            corpus.train,
            corpus.validation,
            corpus.test,
            config.few_shot_fractions,
            config.seed,
        )
        robustness: dict[str, Any] = {}
        if candidate.supports_corruption:
            for corruption_spec in config.corruptions:
                corrupted_windows = corrupt(corpus.test, corruption_spec, config.seed)
                corrupted_embeddings = candidate.transform(
                    corrupted_windows, f"test_{corruption_spec.name}"
                )
                robustness[corruption_spec.name] = evaluate_corruption(
                    test_embeddings, corrupted_embeddings, corpus.test, state
                )
        candidate_reports[spec.name] = {
            "kind": spec.kind,
            "fit_seconds": candidate.fit_seconds,
            "train_embedding_seconds": train_seconds,
            "validation_embedding_seconds": validation_seconds,
            "test_embedding_seconds": test_seconds,
            "evaluation": evaluation,
            "robustness": robustness,
        }
        logger.info(
            "Finished %s in %.2fs (fit) + %.2fs (test embedding)",
            spec.name,
            candidate.fit_seconds,
            test_seconds,
        )
        candidate.save_artifact(output / f"model-{_safe_name(spec.name)}.pt")
        embedding_payload: dict[str, Any] = {
            "train": train_embeddings,
            "test": test_embeddings,
            "test_recording_ids": corpus.test.recording_ids,
            "test_starts": corpus.test.starts,
            "test_stops": corpus.test.stops,
        }
        if validation_embeddings is not None:
            embedding_payload["validation"] = validation_embeddings
        np.savez_compressed(output / f"embeddings-{_safe_name(spec.name)}.npz", **embedding_payload)
    report: dict[str, Any] = {
        "format_version": 1,
        "name": config.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "posemodel_version": __version__,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "protocol": {
            "split_unit": config.split_unit,
            "window_size": config.window_size,
            "stride": config.stride,
            "seed": config.seed,
            "few_shot_fractions": config.few_shot_fractions,
        },
        "manifest": {
            "recordings": [
                {
                    "id": recording.id,
                    "path": str(recording.path),
                    "labels": str(recording.labels) if recording.labels is not None else None,
                    "animal_id": recording.animal_id,
                    "session_id": recording.session_id,
                    "split": recording.split,
                }
                for recording in config.recordings
            ],
            "candidates": [
                {"name": candidate.name, "kind": candidate.kind, "settings": candidate.settings}
                for candidate in config.candidates
            ],
            "corruptions": [
                {
                    "name": corruption.name,
                    "kind": corruption.kind,
                    "magnitude": corruption.magnitude,
                }
                for corruption in config.corruptions
            ],
        },
        "dataset": {
            "recordings": len(config.recordings),
            "train_windows": len(corpus.train),
            "validation_windows": len(corpus.validation) if corpus.validation is not None else 0,
            "test_windows": len(corpus.test),
            "joints": len(corpus.skeleton.joint_names),
            "coordinate_dim": corpus.coordinate_dim,
            "context_dim": corpus.context_dim,
            "labeled_train_windows": int(corpus.train.labeled.sum()),
            "labeled_test_windows": int(corpus.test.labeled.sum()),
        },
        "candidates": candidate_reports,
        "acceptance_gates": _acceptance_gates(candidate_reports),
    }
    write_report(report, output)
    return report
