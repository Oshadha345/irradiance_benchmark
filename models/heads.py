from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class BaselineRegressionHead(nn.Module):
    """Map the concatenated visual-temporal vector to a single 10-minute forecast."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.dropout = float(dropout)

        self.layers = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any], *, input_dim: int) -> "BaselineRegressionHead":
        model_cfg = config["model"]
        return cls(
            input_dim=input_dim,
            hidden_dim=int(model_cfg.get("head_hidden_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.3)),
            output_dim=int(model_cfg.get("output_dim", 1)),
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        if fused_features.ndim != 2:
            raise ValueError(
                f"Expected fused_features with shape (B, D), got {tuple(fused_features.shape)}."
            )
        return self.layers(fused_features)
