from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from .fusion import LadderFusion, LearnableFusionMatrix
from .heads import MLPHead, SpaceTimeDecoder
from .temporal import PyramidTCN
from .wrappers import build_visual_encoder


class IrradianceBenchmarkModel(nn.Module):
    """Unified benchmarking model for visual backbone ablations."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        self.fusion_mode = str(model_cfg.get("fusion_mode", "ladder")).lower()
        self.use_aux_decoder = bool(model_cfg.get("use_aux_decoder", False))
        self.target_channels = list(model_cfg.get("target_channels", [64, 128, 256, 512]))
        self.temporal_embedding_dim = int(model_cfg.get("temporal_embedding_dim", 128))
        self.visual_encoder = build_visual_encoder(config)
        self.temporal_encoder = PyramidTCN(
            input_channels=int(model_cfg.get("temporal_channels", 7)),
            embedding_dim=self.temporal_embedding_dim,
        )

        if self.fusion_mode == "matrix":
            self.fusion = LearnableFusionMatrix(
                num_levels=4,
                visual_dims=self.target_channels,
                temporal_dim=self.temporal_embedding_dim,
                fused_dim=self.temporal_embedding_dim,
            )
            head_input_dim = (4 * self.temporal_embedding_dim) + self.temporal_embedding_dim
        elif self.fusion_mode == "ladder":
            self.fusion = nn.ModuleList(
                [
                    LadderFusion(visual_channels=channels, temporal_channels=self.temporal_embedding_dim)
                    for channels in self.target_channels
                ]
            )
            head_input_dim = sum(self.target_channels) + self.temporal_embedding_dim
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.head = MLPHead(
            input_dim=head_input_dim,
            hidden_dim=int(model_cfg.get("head_hidden_dim", 256)),
            output_dim=len(model_cfg["horizons"]),
            dropout=float(model_cfg.get("dropout", 0.3)),
        )
        self.aux_decoder = SpaceTimeDecoder(self.target_channels[-1]) if self.use_aux_decoder else None

    def fuse_features(
        self,
        visual_features: List[torch.Tensor],
        temporal_features: List[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_mode == "matrix":
            fused_vectors, _ = self.fusion(visual_features, temporal_features)
            fused = torch.cat([torch.cat(fused_vectors, dim=1), temporal_features[-1]], dim=1)
            deepest_visual = visual_features[-1]
            return fused, deepest_visual

        fused_pyramid = [
            fusion_block(visual, temporal)
            for fusion_block, visual, temporal in zip(self.fusion, visual_features, temporal_features)
        ]
        pooled = [self.avg_pool(feature).flatten(1) for feature in fused_pyramid]
        fused = torch.cat([*pooled, temporal_features[-1]], dim=1)
        return fused, fused_pyramid[-1]

    def forward(
        self,
        image: torch.Tensor,
        weather_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        visual_features = self.visual_encoder(image)
        temporal_features = self.temporal_encoder(weather_sequence)
        fused_representation, deepest_visual = self.fuse_features(visual_features, temporal_features)
        predictions = self.head(fused_representation)
        aux_prediction = self.aux_decoder(deepest_visual) if self.use_aux_decoder else None
        return predictions, aux_prediction


def build_model(config: Dict[str, Any]) -> IrradianceBenchmarkModel:
    return IrradianceBenchmarkModel(config)
