from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .heads import BaselineRegressionHead
from .temporal import BaselineTemporalEncoder
from .wrappers import build_visual_encoder


class ConcatenationFusion(nn.Module):
    """Fuse visual and temporal descriptors with direct concatenation."""

    def __init__(self, visual_dim: int, temporal_dim: int) -> None:
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.temporal_dim = int(temporal_dim)
        self.output_dim = self.visual_dim + self.temporal_dim

    def forward(self, visual_features: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        if visual_features.ndim != 2:
            raise ValueError(f"Expected visual_features with shape (B, Dv), got {tuple(visual_features.shape)}.")
        if temporal_features.ndim != 2:
            raise ValueError(f"Expected temporal_features with shape (B, Dt), got {tuple(temporal_features.shape)}.")
        if visual_features.shape[0] != temporal_features.shape[0]:
            raise ValueError(
                "Visual and temporal batches must match: "
                f"{visual_features.shape[0]} vs {temporal_features.shape[0]}."
            )
        return torch.cat([visual_features, temporal_features], dim=1)


class BaselineFusionModel(nn.Module):
    """Double-blind MERCon baseline model."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        visual_feature_dim = int(model_cfg.get("visual_feature_dim", 1024))
        temporal_hidden_dim = int(model_cfg.get("temporal_hidden_dim", 128))

        self.visual_encoder = build_visual_encoder(config)
        self.temporal_encoder = BaselineTemporalEncoder.from_config(config)
        self.fusion = ConcatenationFusion(
            visual_dim=visual_feature_dim,
            temporal_dim=temporal_hidden_dim,
        )
        self.head = BaselineRegressionHead.from_config(
            config,
            input_dim=self.fusion.output_dim,
        )

    def forward(
        self,
        image: torch.Tensor,
        weather_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        visual_features = self.visual_encoder(image)
        temporal_features = self.temporal_encoder(weather_sequence)
        fused_features = self.fusion(visual_features, temporal_features)
        prediction = self.head(fused_features)
        return prediction, None
