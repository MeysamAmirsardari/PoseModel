"""Graph-temporal masked teacher-student autoencoder."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from posemodel.schema import Skeleton


@dataclass(frozen=True, slots=True)
class ModelConfig:
    coordinate_dim: int = 2
    context_dim: int = 0
    hidden_dim: int = 192
    latent_dim: int = 32
    depth: int = 6
    decoder_depth: int = 2
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    max_window_size: int = 512
    max_individuals: int = 8
    variational: bool = True


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    reconstruction_weight: float = 1.0
    velocity_weight: float = 0.25
    bone_weight: float = 0.1
    latent_prediction_weight: float = 1.0
    kl_weight: float = 0.001


def _sinusoidal_positions(length: int, width: int) -> Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10_000.0) / width)
    )
    values = torch.zeros(length, width)
    values[:, 0::2] = torch.sin(positions * frequencies)
    values[:, 1::2] = torch.cos(positions * frequencies[: values[:, 1::2].shape[1]])
    return values


class FactorizedGraphTemporalBlock(nn.Module):
    """Spatial graph attention followed by temporal and cross-individual attention."""

    def __init__(self, config: ModelConfig, disallowed_spatial: Tensor) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.register_buffer("disallowed_spatial", disallowed_spatial, persistent=False)
        self.spatial_norm = nn.LayerNorm(hidden)
        self.temporal_norm = nn.LayerNorm(hidden)
        self.interaction_norm = nn.LayerNorm(hidden)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.spatial = nn.MultiheadAttention(
            hidden, config.num_heads, config.dropout, batch_first=True
        )
        self.temporal = nn.MultiheadAttention(
            hidden, config.num_heads, config.dropout, batch_first=True
        )
        self.interaction = nn.MultiheadAttention(
            hidden, config.num_heads, config.dropout, batch_first=True
        )
        expanded = int(hidden * config.mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, expanded),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(expanded, hidden),
            nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        batch, time, individuals, joints, hidden = tokens.shape
        spatial = self.spatial_norm(tokens).reshape(batch * time * individuals, joints, hidden)
        spatial, _ = self.spatial(
            spatial, spatial, spatial, attn_mask=self.disallowed_spatial, need_weights=False
        )
        tokens = tokens + self.dropout(spatial.reshape(batch, time, individuals, joints, hidden))

        temporal = self.temporal_norm(tokens).permute(0, 2, 3, 1, 4)
        temporal = temporal.reshape(batch * individuals * joints, time, hidden)
        temporal, _ = self.temporal(temporal, temporal, temporal, need_weights=False)
        temporal = temporal.reshape(batch, individuals, joints, time, hidden).permute(0, 3, 1, 2, 4)
        tokens = tokens + self.dropout(temporal)

        if individuals > 1:
            interaction = self.interaction_norm(tokens).reshape(
                batch * time, individuals * joints, hidden
            )
            interaction, _ = self.interaction(
                interaction, interaction, interaction, need_weights=False
            )
            interaction = interaction.reshape(batch, time, individuals, joints, hidden)
            tokens = tokens + self.dropout(interaction)
        feed_forward = cast(Tensor, self.ffn(self.ffn_norm(tokens)))
        return tokens + feed_forward


class PoseEncoder(nn.Module):
    def __init__(self, config: ModelConfig, n_joints: int, disallowed_spatial: Tensor) -> None:
        super().__init__()
        feature_dim = 2 * config.coordinate_dim + config.context_dim + 2
        self.config = config
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.mask_token = nn.Parameter(torch.zeros(config.hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.joint_embedding = nn.Embedding(n_joints, config.hidden_dim)
        self.individual_embedding = nn.Embedding(config.max_individuals, config.hidden_dim)
        self.time_embedding: Tensor
        self.register_buffer(
            "time_embedding",
            _sinusoidal_positions(config.max_window_size, config.hidden_dim),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            FactorizedGraphTemporalBlock(config, disallowed_spatial) for _ in range(config.depth)
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        coordinates: Tensor,
        confidence: Tensor,
        observed: Tensor,
        token_mask: Tensor,
        context: Tensor | None = None,
    ) -> Tensor:
        batch, time, individuals, joints, _ = coordinates.shape
        if time > self.config.max_window_size:
            raise ValueError("Window exceeds max_window_size")
        if individuals > self.config.max_individuals:
            raise ValueError("Individual count exceeds max_individuals")
        velocity = torch.zeros_like(coordinates)
        velocity[:, 1:] = coordinates[:, 1:] - coordinates[:, :-1]
        feature_parts = [
            coordinates,
            velocity,
            confidence.unsqueeze(-1),
            observed.float().unsqueeze(-1),
        ]
        if self.config.context_dim:
            if context is None or context.shape[-1] != self.config.context_dim:
                raise ValueError("Context features do not match model context_dim")
            feature_parts.append(context.unsqueeze(3).expand(-1, -1, -1, joints, -1))
        features = torch.cat(feature_parts, dim=-1)
        tokens = self.input_projection(features)
        tokens = torch.where(token_mask.unsqueeze(-1), self.mask_token, tokens)
        joint_ids = torch.arange(joints, device=tokens.device)
        individual_ids = torch.arange(individuals, device=tokens.device)
        tokens = tokens + self.joint_embedding(joint_ids)[None, None, None, :, :]
        tokens = tokens + self.individual_embedding(individual_ids)[None, None, :, None, :]
        tokens = tokens + self.time_embedding[:time][None, :, None, None, :]
        for module in self.blocks:
            tokens = cast(FactorizedGraphTemporalBlock, module)(tokens)
        return cast(Tensor, self.output_norm(tokens))


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: Tensor, observed: Tensor) -> Tensor:
        batch, time, individuals, joints, hidden = tokens.shape
        flattened = tokens.reshape(batch, time * individuals * joints, hidden)
        padding = ~observed.reshape(batch, time * individuals * joints)
        all_missing = padding.all(dim=1)
        if all_missing.any():
            padding = padding.clone()
            padding[all_missing, 0] = False
        query = self.query.expand(batch, -1, -1)
        pooled, _ = self.attention(
            query, flattened, flattened, key_padding_mask=padding, need_weights=False
        )
        return cast(Tensor, self.norm(pooled[:, 0]))


class PoseDecoder(nn.Module):
    def __init__(self, config: ModelConfig, n_joints: int, disallowed_spatial: Tensor) -> None:
        super().__init__()
        self.config = config
        self.latent_projection = nn.Linear(config.latent_dim, config.hidden_dim)
        self.joint_embedding = nn.Embedding(n_joints, config.hidden_dim)
        self.individual_embedding = nn.Embedding(config.max_individuals, config.hidden_dim)
        self.time_embedding: Tensor
        self.register_buffer(
            "time_embedding",
            _sinusoidal_positions(config.max_window_size, config.hidden_dim),
            persistent=False,
        )
        decoder_config = ModelConfig(**{**asdict(config), "depth": config.decoder_depth})
        self.blocks = nn.ModuleList(
            FactorizedGraphTemporalBlock(decoder_config, disallowed_spatial)
            for _ in range(config.decoder_depth)
        )
        self.coordinate_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.coordinate_dim),
        )

    def forward(self, latent: Tensor, time: int, individuals: int, joints: int) -> Tensor:
        batch = latent.shape[0]
        tokens = self.latent_projection(latent)[:, None, None, None, :]
        tokens = tokens.expand(batch, time, individuals, joints, -1)
        joint_ids = torch.arange(joints, device=latent.device)
        individual_ids = torch.arange(individuals, device=latent.device)
        tokens = tokens + self.joint_embedding(joint_ids)[None, None, None, :, :]
        tokens = tokens + self.individual_embedding(individual_ids)[None, None, :, None, :]
        tokens = tokens + self.time_embedding[:time][None, :, None, None, :]
        for module in self.blocks:
            tokens = cast(FactorizedGraphTemporalBlock, module)(tokens)
        return cast(Tensor, self.coordinate_head(tokens))


class GraphMotionMAE(nn.Module):
    """Masked graph-temporal model with an EMA contextual teacher."""

    def __init__(self, skeleton: Skeleton, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        if config.hidden_dim % config.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        adjacency = skeleton.adjacency()
        if not skeleton.edges:
            adjacency[:] = True
        disallowed = torch.from_numpy(~adjacency)
        self.skeleton = skeleton
        self.config = config
        self.student = PoseEncoder(config, len(skeleton.joint_names), disallowed)
        self.teacher = copy.deepcopy(self.student)
        self.teacher.requires_grad_(False)
        self.pool = AttentionPool(config.hidden_dim, config.num_heads)
        self.mu_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.token_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.decoder = PoseDecoder(config, len(skeleton.joint_names), disallowed)

    def train(self, mode: bool = True) -> GraphMotionMAE:
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        if not 0 <= momentum <= 1:
            raise ValueError("Teacher momentum must be in [0, 1]")
        for teacher, student in zip(
            self.teacher.parameters(), self.student.parameters(), strict=True
        ):
            teacher.data.lerp_(student.data, 1.0 - momentum)

    @staticmethod
    def sample_mask(observed: Tensor, ratio: float) -> Tensor:
        if not 0 < ratio < 1:
            raise ValueError("Mask ratio must be between zero and one")
        # Mix independent tokens, whole-joint tubes, and whole-frame masks. If each component
        # uses this probability their expected union is the requested mask ratio.
        component_probability = 1.0 - (1.0 - ratio) ** (1.0 / 3.0)
        token_mask = torch.rand_like(observed, dtype=torch.float32) < component_probability
        joint_tubes = (
            torch.rand(
                observed.shape[0],
                1,
                observed.shape[2],
                observed.shape[3],
                device=observed.device,
            )
            < component_probability
        )
        frame_mask = (
            torch.rand(
                observed.shape[0],
                observed.shape[1],
                observed.shape[2],
                1,
                device=observed.device,
            )
            < component_probability
        )
        mask = (token_mask | joint_tubes | frame_mask) & observed
        # Guarantee one target for every sample that contains any observed token.
        flattened_observed = observed.flatten(1)
        flattened_mask = mask.flatten(1)
        for sample in range(observed.shape[0]):
            if flattened_observed[sample].any() and not flattened_mask[sample].any():
                first = torch.nonzero(flattened_observed[sample], as_tuple=False)[0, 0]
                flattened_mask[sample, first] = True
        return cast(Tensor, flattened_mask.reshape_as(observed))

    def encode(
        self,
        coordinates: Tensor,
        confidence: Tensor,
        observed: Tensor,
        context: Tensor | None = None,
    ) -> Tensor:
        missing = ~observed
        tokens = self.student(coordinates, confidence, observed, missing, context)
        pooled = self.pool(tokens, observed)
        return cast(Tensor, self.mu_head(pooled))

    def forward(
        self,
        coordinates: Tensor,
        confidence: Tensor,
        observed: Tensor,
        *,
        mask_ratio: float = 0.6,
        token_mask: Tensor | None = None,
        context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if coordinates.ndim != 5:
            raise ValueError("coordinates must have shape (batch, time, individuals, joints, dims)")
        if token_mask is None:
            token_mask = self.sample_mask(observed, mask_ratio)
        student_mask = token_mask | ~observed
        student_tokens = self.student(coordinates, confidence, observed, student_mask, context)
        with torch.no_grad():
            teacher_tokens = self.teacher(coordinates, confidence, observed, ~observed, context)
        pooled = self.pool(student_tokens, observed)
        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled).clamp(-10.0, 10.0)
        if self.config.variational and self.training:
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            latent = mu
        _, time, individuals, joints, _ = coordinates.shape
        reconstruction = self.decoder(latent, time, individuals, joints)
        return {
            "reconstruction": reconstruction,
            "student_tokens": self.token_predictor(student_tokens),
            "teacher_tokens": teacher_tokens.detach(),
            "mu": mu,
            "logvar": logvar,
            "latent": latent,
            "token_mask": token_mask,
        }

    def objective(
        self,
        output: dict[str, Tensor],
        coordinates: Tensor,
        confidence: Tensor,
        observed: Tensor,
        config: ObjectiveConfig | None = None,
        *,
        kl_multiplier: float = 1.0,
    ) -> dict[str, Tensor]:
        config = config or ObjectiveConfig()
        mask = output["token_mask"] & observed
        weights = confidence * mask.float()
        denominator = weights.sum().clamp_min(1.0) * coordinates.shape[-1]
        coordinate_error = F.smooth_l1_loss(output["reconstruction"], coordinates, reduction="none")
        reconstruction = (coordinate_error * weights.unsqueeze(-1)).sum() / denominator

        target_velocity = coordinates[:, 1:] - coordinates[:, :-1]
        predicted_velocity = output["reconstruction"][:, 1:] - output["reconstruction"][:, :-1]
        velocity_mask = observed[:, 1:] & observed[:, :-1]
        velocity_mask &= mask[:, 1:] | mask[:, :-1]
        velocity_weights = velocity_mask.float() * torch.minimum(
            confidence[:, 1:], confidence[:, :-1]
        )
        velocity_denominator = velocity_weights.sum().clamp_min(1.0) * coordinates.shape[-1]
        velocity = (
            F.smooth_l1_loss(predicted_velocity, target_velocity, reduction="none")
            * velocity_weights.unsqueeze(-1)
        ).sum() / velocity_denominator

        cosine = 1.0 - F.cosine_similarity(
            output["student_tokens"], output["teacher_tokens"], dim=-1
        )
        latent_prediction = (cosine * mask.float()).sum() / mask.sum().clamp_min(1)

        bone = coordinates.new_zeros(())
        if self.skeleton.edges:
            bone_errors: list[Tensor] = []
            for source, target in self.skeleton.edges:
                true_length = torch.linalg.vector_norm(
                    coordinates[..., source, :] - coordinates[..., target, :], dim=-1
                )
                predicted_length = torch.linalg.vector_norm(
                    output["reconstruction"][..., source, :]
                    - output["reconstruction"][..., target, :],
                    dim=-1,
                )
                valid = observed[..., source] & observed[..., target]
                relevant = mask[..., source] | mask[..., target]
                edge_mask = valid & relevant
                edge_loss = F.smooth_l1_loss(predicted_length, true_length, reduction="none")
                bone_errors.append(
                    (edge_loss * edge_mask.float()).sum() / edge_mask.sum().clamp_min(1)
                )
            bone = torch.stack(bone_errors).mean()

        kl = (
            -0.5
            * (1.0 + output["logvar"] - output["mu"].square() - output["logvar"].exp())
            .sum(dim=-1)
            .mean()
        )
        total = (
            config.reconstruction_weight * reconstruction
            + config.velocity_weight * velocity
            + config.bone_weight * bone
            + config.latent_prediction_weight * latent_prediction
            + config.kl_weight * kl_multiplier * kl
        )
        return {
            "loss": total,
            "reconstruction": reconstruction.detach(),
            "velocity": velocity.detach(),
            "bone": bone.detach(),
            "latent_prediction": latent_prediction.detach(),
            "kl": kl.detach(),
        }

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {"model": asdict(self.config), "skeleton": self.skeleton.to_dict()}
