from __future__ import annotations

from typing import Any, Dict

from .fusion import BaselineFusionModel


class IrradianceBaselineModel(BaselineFusionModel):
    """Named entrypoint for the MERCon baseline architecture."""


class IrradianceBenchmarkModel(IrradianceBaselineModel):
    """Backward-compatible alias retained for train/eval entrypoints."""


def build_model(config: Dict[str, Any]) -> IrradianceBenchmarkModel:
    return IrradianceBenchmarkModel(config)
