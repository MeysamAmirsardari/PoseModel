from __future__ import annotations

from posemodel.models import GraphMotionMAE, ModelConfig
from posemodel.preprocessing import normalize_pose
from posemodel.schema import PoseSequence
from posemodel.training import TrainingConfig, embed, fit, load_checkpoint
from posemodel.windows import FrameInterval, PoseWindowDataset, build_window_index


def test_train_checkpoint_and_embed(tmp_path, pose_sequence: PoseSequence) -> None:
    sequence, _ = normalize_pose(pose_sequence)
    intervals = [
        FrameInterval(0, 24, "train"),
        FrameInterval(24, 48, "validation"),
    ]
    dataset = PoseWindowDataset(
        sequence,
        build_window_index(intervals, window_size=8, stride=8),
    )
    model_config = ModelConfig(
        coordinate_dim=2,
        context_dim=sequence.context_dim,
        hidden_dim=16,
        latent_dim=4,
        depth=1,
        decoder_depth=1,
        num_heads=4,
        dropout=0.0,
        max_window_size=8,
    )
    training_config = TrainingConfig(
        window_size=8,
        stride=8,
        batch_size=2,
        epochs=1,
        patience=1,
        model=model_config,
    )
    model = GraphMotionMAE(sequence.skeleton, model_config)
    history = fit(
        model,
        dataset.select_split("train"),
        dataset.select_split("validation"),
        training_config,
        tmp_path,
        device="cpu",
    )
    loaded, metadata = load_checkpoint(tmp_path / "model.pt")
    embeddings, starts, stops = embed(loaded, dataset, batch_size=2, device="cpu")
    assert len(history) == 1
    assert metadata["format_version"] == 1
    assert embeddings.shape == (6, 4)
    assert starts[0] == 0
    assert stops[-1] == 48
