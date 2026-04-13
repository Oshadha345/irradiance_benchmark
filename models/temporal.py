from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class BaselineTemporalEncoder(nn.Module):
    """Encode the 40x7 weather history with a single-layer LSTM."""

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaselineTemporalEncoder":
        model_cfg = config["model"]
        return cls(
            input_dim=int(model_cfg.get("temporal_channels", 7)),
            hidden_dim=int(model_cfg.get("temporal_hidden_dim", 128)),
            num_layers=int(model_cfg.get("temporal_num_layers", 1)),
        )

    def forward(self, weather_sequence: torch.Tensor) -> torch.Tensor:
        if weather_sequence.ndim != 3:
            raise ValueError(
                f"Expected weather_sequence with shape (B, T, C), got {tuple(weather_sequence.shape)}."
            )
        _, (hidden_state, _) = self.lstm(weather_sequence)
        return hidden_state[-1]
