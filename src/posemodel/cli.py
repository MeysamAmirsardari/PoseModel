"""PoseModel command-line interface."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from posemodel.benchmark import BenchmarkConfig, run_benchmark, write_benchmark_index
from posemodel.clustering import discover_states
from posemodel.io import load_dlc
from posemodel.models import GraphMotionMAE
from posemodel.preprocessing import NormalizeConfig, normalize_pose
from posemodel.schema import PoseSequence, Skeleton
from posemodel.training import TrainingConfig, fit, load_checkpoint
from posemodel.training import embed as embed_windows
from posemodel.windows import FrameInterval, PoseWindowDataset, build_window_index, contiguous_split

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Self-supervised behavioral representation learning from pose sequences.",
)


@app.command()
def benchmark(
    manifest: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option(help="Benchmark artifact directory.")],
) -> None:
    """Compare representations under one leakage-safe multi-recording protocol."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = BenchmarkConfig.from_yaml(manifest)
    report = run_benchmark(config, output)
    typer.echo(
        f"Benchmarked {len(report['candidates'])} candidates; report: {output / 'benchmark.html'}"
    )


@app.command("benchmark-index")
def benchmark_index(
    manifest: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Argument(help="Output CSV path.")],
) -> None:
    """Export exact split/window identities for external representation methods."""
    path = write_benchmark_index(BenchmarkConfig.from_yaml(manifest), output)
    typer.echo(f"Saved benchmark window index to {path}")


def _load_skeleton(path: Path | None) -> Skeleton | None:
    if path is None:
        return None
    return Skeleton.from_dict(json.loads(path.read_text(encoding="utf-8")))


@app.command()
def inspect(
    input_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    fps: Annotated[float, typer.Option(help="Recording frame rate.")] = 30.0,
) -> None:
    """Validate and summarize a DeepLabCut pose file."""
    sequence = load_dlc(input_path, fps=fps)
    coverage = float(sequence.observed.mean())
    typer.echo(
        json.dumps(
            {
                "frames": sequence.n_frames,
                "individuals": sequence.n_individuals,
                "joints": sequence.n_joints,
                "dimensions": sequence.coordinate_dim,
                "fps": sequence.fps,
                "observed_fraction": coverage,
                "joint_names": sequence.skeleton.joint_names,
            },
            indent=2,
        )
    )


@app.command()
def prepare(
    input_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output_path: Annotated[Path, typer.Argument()],
    fps: Annotated[float, typer.Option(help="Recording frame rate.")],
    skeleton: Annotated[
        Path | None, typer.Option(exists=True, readable=True, help="Skeleton JSON file.")
    ] = None,
    confidence_threshold: Annotated[float, typer.Option()] = 0.2,
    interpolate_max_gap: Annotated[int, typer.Option()] = 5,
) -> None:
    """Convert and normalize DeepLabCut output into the canonical NPZ format."""
    sequence = load_dlc(input_path, fps=fps, skeleton=_load_skeleton(skeleton))
    normalized, _ = normalize_pose(
        sequence,
        NormalizeConfig(
            confidence_threshold=confidence_threshold,
            interpolate_max_gap=interpolate_max_gap,
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.save(output_path)
    typer.echo(f"Saved {normalized.n_frames} normalized frames to {output_path}")


@app.command()
def train(
    sequence_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    config_path: Annotated[
        Path, typer.Option("--config", exists=True, readable=True, help="Training YAML.")
    ],
    output: Annotated[Path, typer.Option(help="Run output directory.")],
    device: Annotated[str, typer.Option(help="auto, cpu, cuda, or mps.")] = "auto",
) -> None:
    """Train on leakage-safe contiguous partitions of one prepared recording."""
    sequence = PoseSequence.load(sequence_path)
    config = TrainingConfig.from_yaml(config_path)
    model_config = replace(
        config.model,
        coordinate_dim=sequence.coordinate_dim,
        context_dim=sequence.context_dim,
    )
    config = replace(config, model=model_config)
    intervals = contiguous_split(sequence.n_frames, gap=config.window_size)
    windows = build_window_index(intervals, window_size=config.window_size, stride=config.stride)
    dataset = PoseWindowDataset(sequence, windows)
    train_dataset = dataset.select_split("train")
    validation_dataset = dataset.select_split("validation")
    model = GraphMotionMAE(sequence.skeleton, config.model)
    history = fit(model, train_dataset, validation_dataset, config, output, device=device)
    split_manifest = [
        {"start": interval.start, "stop": interval.stop, "split": interval.split}
        for interval in intervals
    ]
    (output / "splits.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    typer.echo(f"Finished {len(history)} epochs; best checkpoint: {output / 'model.pt'}")


@app.command()
def embed(
    checkpoint: Annotated[Path, typer.Argument(exists=True, readable=True)],
    sequence_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output_path: Annotated[Path, typer.Argument()],
    device: Annotated[str, typer.Option()] = "auto",
) -> None:
    """Embed every complete window of a prepared recording."""
    sequence = PoseSequence.load(sequence_path)
    model, metadata = load_checkpoint(checkpoint, device="cpu")
    training = metadata["training"]
    interval = FrameInterval(0, sequence.n_frames, "test")
    windows = build_window_index(
        [interval], window_size=training["window_size"], stride=training["stride"]
    )
    dataset = PoseWindowDataset(sequence, windows)
    values, starts, stops = embed_windows(model, dataset, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, embeddings=values, starts=starts, stops=stops)
    typer.echo(f"Saved {len(values)} embeddings to {output_path}")


@app.command()
def cluster(
    embeddings_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output_path: Annotated[Path, typer.Argument()],
    min_states: Annotated[int, typer.Option()] = 2,
    max_states: Annotated[int, typer.Option()] = 20,
    stickiness: Annotated[float, typer.Option()] = 20.0,
    seed: Annotated[int, typer.Option()] = 42,
) -> None:
    """Discover temporally persistent behavioral states."""
    with np.load(embeddings_path, allow_pickle=False) as archive:
        values = archive["embeddings"]
        starts = archive["starts"]
        stops = archive["stops"]
    states = discover_states(
        values,
        min_states=min_states,
        max_states=max_states,
        stickiness=stickiness,
        seed=seed,
    )
    table = pd.DataFrame(
        {
            "start_frame": starts,
            "stop_frame": stops,
            "state": states.labels,
            "state_probability": states.probabilities.max(axis=1),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    summary = {
        "n_states": states.n_states,
        "bic": states.bic,
        "transition_matrix": states.transition_matrix.tolist(),
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(f"Saved {states.n_states} behavioral states to {output_path}")


if __name__ == "__main__":
    app()
