from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import timm
from timm.models import load_checkpoint as timm_load_checkpoint


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resolve_module(root: nn.Module, dotted_name: str) -> nn.Module:
    module = root
    for token in dotted_name.split("."):
        module = getattr(module, token)
    return module


class FeatureProjector(nn.Module):
    def __init__(self, in_channels: Sequence[int], out_channels: Sequence[int]) -> None:
        super().__init__()
        if len(in_channels) != 4 or len(out_channels) != 4:
            raise ValueError("Expected exactly four backbone stages.")
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                )
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return [layer(feature) for layer, feature in zip(self.layers, features)]


class BaseEncoderWrapper(nn.Module):
    def __init__(self, target_channels: Sequence[int]) -> None:
        super().__init__()
        self.target_channels = list(target_channels)

    @property
    def final_stage_module(self) -> nn.Module:
        return self.projector.layers[-1]


class TimmEncoderWrapper(BaseEncoderWrapper):
    """Unified `timm` wrapper for Swin and ConvNeXt backbones."""

    def __init__(
        self,
        model_name: str,
        *,
        pretrained: bool,
        weight_path: str | None,
        image_size: int,
        target_channels: Sequence[int],
        out_indices: Sequence[int] = (0, 1, 2, 3),
        model_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(target_channels=target_channels)
        model_kwargs = dict(model_kwargs or {})
        local_weight_path = Path(weight_path).expanduser() if weight_path else None
        use_remote_pretrained = pretrained and (local_weight_path is None or not local_weight_path.is_file())
        self.backbone = timm.create_model(
            model_name,
            pretrained=use_remote_pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
            img_size=image_size,
            **model_kwargs,
        )
        if local_weight_path is not None and local_weight_path.is_file():
            timm_load_checkpoint(self.backbone, str(local_weight_path), strict=False)
        self.feature_channels = list(self.backbone.feature_info.channels())
        self.projector = FeatureProjector(self.feature_channels, self.target_channels)
        module_name = self.backbone.feature_info.module_name(out_indices[-1])
        self.backbone_final_stage = _resolve_module(self.backbone, module_name)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.backbone(x)
        return self.projector(features)


class MambaEncoderWrapper(BaseEncoderWrapper):
    """Wrapper for locally vendored Mamba-family visual backbones."""

    VMAMBA_PRESETS: Dict[str, Dict[str, Any]] = {
        "tiny": {
            "depths": [2, 2, 8, 2],
            "dims": 96,
            "ssm_d_state": 1,
            "ssm_ratio": 1.0,
            "ssm_dt_rank": "auto",
            "ssm_conv": 3,
            "ssm_conv_bias": False,
            "forward_type": "v05_noz",
            "mlp_ratio": 4.0,
            "drop_path_rate": 0.2,
            "norm_layer": "ln2d",
            "downsample_version": "v3",
            "patchembed_version": "v2",
        },
        "small": {
            "depths": [2, 2, 15, 2],
            "dims": 96,
            "ssm_d_state": 1,
            "ssm_ratio": 2.0,
            "ssm_dt_rank": "auto",
            "ssm_conv": 3,
            "ssm_conv_bias": False,
            "forward_type": "v05_noz",
            "mlp_ratio": 4.0,
            "drop_path_rate": 0.3,
            "norm_layer": "ln2d",
            "downsample_version": "v3",
            "patchembed_version": "v2",
        },
        "base": {
            "depths": [2, 2, 15, 2],
            "dims": 128,
            "ssm_d_state": 1,
            "ssm_ratio": 2.0,
            "ssm_dt_rank": "auto",
            "ssm_conv": 3,
            "ssm_conv_bias": False,
            "forward_type": "v05_noz",
            "mlp_ratio": 4.0,
            "drop_path_rate": 0.6,
            "norm_layer": "ln2d",
            "downsample_version": "v3",
            "patchembed_version": "v2",
        },
    }

    MAMBAVISION_CONSTRUCTORS = {
        "tiny": "mamba_vision_T",
        "tiny2": "mamba_vision_T2",
        "small": "mamba_vision_S",
        "base": "mamba_vision_B",
        "large": "mamba_vision_L",
        "large2": "mamba_vision_L2",
    }

    SPATIAL_PRESETS: Dict[str, Dict[str, Any]] = {
        "tiny": {"depths": [2, 4, 8, 4], "dims": 64, "d_state": 1, "dt_init": "random", "mlp_ratio": 4.0, "drop_path_rate": 0.2},
        "small": {"depths": [2, 4, 21, 5], "dims": 64, "d_state": 1, "dt_init": "random", "mlp_ratio": 4.0, "drop_path_rate": 0.3},
        "base": {"depths": [2, 4, 21, 5], "dims": 96, "d_state": 1, "dt_init": "random", "mlp_ratio": 4.0, "drop_path_rate": 0.5},
    }

    def __init__(
        self,
        family: str,
        scale: str,
        *,
        pretrained: bool,
        weight_path: str | None,
        image_size: int,
        target_channels: Sequence[int],
        model_kwargs: Dict[str, Any] | None = None,
        backbone_root: str | Path | None = None,
    ) -> None:
        super().__init__(target_channels=target_channels)
        self.family = family.lower()
        self.scale = scale.lower()
        self.image_size = image_size
        self.weight_path = weight_path
        self.model_kwargs = dict(model_kwargs or {})
        self.backbone_root = Path(backbone_root or Path(__file__).resolve().parent / "backbones")
        self._ensure_backbone_root_on_path()
        self.backbone = self._build_backbone(pretrained=pretrained)
        sample_channels = self._infer_stage_channels()
        self.projector = FeatureProjector(sample_channels, self.target_channels)

    def _ensure_backbone_root_on_path(self) -> None:
        candidate_paths = [self.backbone_root.resolve()]
        if self.family == "spatial_mamba":
            candidate_paths.extend(
                [
                    (self.backbone_root / "spatial_mamba" / "kernels" / "dwconv2d").resolve(),
                    (self.backbone_root / "spatial_mamba" / "kernels" / "selective_scan").resolve(),
                ]
            )

        for candidate in candidate_paths:
            candidate_str = str(candidate)
            if candidate.exists() and candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

    def _build_backbone(self, *, pretrained: bool) -> nn.Module:
        if self.family == "vmamba":
            module = importlib.import_module("vmamba.vmamba")
            preset = dict(self.VMAMBA_PRESETS[self.scale])
            preset.update(self.model_kwargs)
            return module.Backbone_VSSM(
                out_indices=(0, 1, 2, 3),
                pretrained=self.weight_path if pretrained else None,
                imgsize=self.image_size,
                patch_size=4,
                in_chans=3,
                num_classes=1000,
                **preset,
            )

        if self.family == "mambavision":
            module = importlib.import_module("mambavision.mamba_vision")
            constructor_name = self.MAMBAVISION_CONSTRUCTORS[self.scale]
            constructor = getattr(module, constructor_name)
            constructor_kwargs = dict(self.model_kwargs)
            constructor_kwargs["resolution"] = self.image_size
            if self.weight_path is not None:
                constructor_kwargs["model_path"] = self.weight_path
            model = constructor(pretrained=pretrained, **constructor_kwargs)
            self._mambavision_features: Dict[str, torch.Tensor] = {}
            for index, level in enumerate(model.levels):
                level.register_forward_hook(self._make_hook(f"stage_{index + 1}"))
            return model

        if self.family == "spatial_mamba":
            module = importlib.import_module("spatial_mamba.spatialmamba")
            preset = dict(self.SPATIAL_PRESETS[self.scale])
            preset.update(self.model_kwargs)
            return module.Backbone_SpatialMamba(
                out_indices=(0, 1, 2, 3),
                pretrained=self.weight_path if pretrained else None,
                img_size=self.image_size,
                patch_size=4,
                in_chans=3,
                num_classes=1000,
                norm_layer="ln",
                **preset,
            )

        raise ValueError(f"Unsupported Mamba backbone family: {self.family}")

    def _make_hook(self, name: str):
        def hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self._mambavision_features[name] = output

        return hook

    def _infer_stage_channels(self) -> List[int]:
        with torch.no_grad():
            was_training = self.backbone.training
            self.backbone.eval()
            sample = torch.zeros(1, 3, self.image_size, self.image_size)
            features = self._forward_backbone(sample)
            if was_training:
                self.backbone.train()
        return [int(feature.shape[1]) for feature in features]

    def _forward_backbone(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.family == "mambavision":
            self._mambavision_features.clear()
            if hasattr(self.backbone, "forward_features"):
                _ = self.backbone.forward_features(x)
            else:
                _ = self.backbone(x)
            return [self._mambavision_features[f"stage_{idx}"] for idx in range(1, 5)]

        outputs = self.backbone(x)
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 4:
            raise RuntimeError(f"{self.family} backbone did not return a 4-stage pyramid.")
        return list(outputs)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self._forward_backbone(x)
        return self.projector(features)


def build_visual_encoder(config: Dict[str, Any]) -> BaseEncoderWrapper:
    visual_cfg = config["visual"]
    family = str(visual_cfg["family"]).lower()
    target_channels = list(config["model"].get("target_channels", [64, 128, 256, 512]))
    image_size = int(config["data"].get("image_size", 512))
    model_kwargs = dict(visual_cfg.get("model_kwargs", {}))

    if family == "timm":
        return TimmEncoderWrapper(
            model_name=str(visual_cfg["name"]),
            pretrained=bool(visual_cfg.get("pretrained", True)),
            weight_path=visual_cfg.get("weight_path"),
            image_size=image_size,
            target_channels=target_channels,
            model_kwargs=model_kwargs,
        )

    if family in {"vmamba", "mambavision", "spatial_mamba"}:
        return MambaEncoderWrapper(
            family=family,
            scale=str(visual_cfg.get("scale", "tiny")),
            pretrained=bool(visual_cfg.get("pretrained", True)),
            weight_path=visual_cfg.get("weight_path"),
            image_size=image_size,
            target_channels=target_channels,
            model_kwargs=model_kwargs,
        )

    raise ValueError(f"Unsupported visual.family: {family}")
