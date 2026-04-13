from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Compact forecast head used after visual-temporal fusion."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SpaceTimeDecoder(nn.Module):
    """Auxiliary decoder for next-frame supervision."""

    def __init__(self, input_dim: int, output_channels: int = 3) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(input_dim, 256, kernel_size=1)

        def up_block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.up1 = up_block(256, 128)
        self.up2 = up_block(128, 64)
        self.up3 = up_block(64, 32)
        self.final = nn.Conv2d(32, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.final(x)

