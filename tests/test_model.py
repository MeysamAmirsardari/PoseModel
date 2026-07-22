from __future__ import annotations

import torch

from posemodel.models import GraphMotionMAE, ModelConfig, ObjectiveConfig
from posemodel.schema import Skeleton


def test_model_forward_objective_and_encoding(skeleton: Skeleton) -> None:
    torch.manual_seed(1)
    config = ModelConfig(
        hidden_dim=24,
        latent_dim=8,
        depth=1,
        decoder_depth=1,
        num_heads=4,
        dropout=0.0,
        max_window_size=16,
    )
    model = GraphMotionMAE(skeleton, config)
    coordinates = torch.randn(2, 8, 1, 3, 2)
    confidence = torch.ones(2, 8, 1, 3)
    observed = torch.ones(2, 8, 1, 3, dtype=torch.bool)
    output = model(coordinates, confidence, observed, mask_ratio=0.5)
    losses = model.objective(output, coordinates, confidence, observed, ObjectiveConfig())
    assert output["reconstruction"].shape == coordinates.shape
    assert output["latent"].shape == (2, 8)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    model.update_teacher(0.99)
    model.eval()
    assert model.encode(coordinates, confidence, observed).shape == (2, 8)
