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
    tokens = dotted_name.split(".")
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if hasattr(module, token):
            module = getattr(module, token)
            index += 1
            continue
        if index + 1 < len(tokens) and tokens[index + 1].isdigit():
            flattened = f"{token}_{tokens[index + 1]}"
            if hasattr(module, flattened):
                module = getattr(module, flattened)
                index += 2
                continue
        if token.isdigit():
            module = module[int(token)]  # type: ignore[index]
            index += 1
            continue
        raise AttributeError(f"Unable to resolve module path '{dotted_name}' from '{type(root).__name__}'.")
    return module


class FeatureProjector(nn.Module):
    """Project a 4-stage pyramid into a single 1024-D visual descriptor."""

    def __init__(self, in_channels: Sequence[int], output_dim: int = 1024) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("Expected exactly four backbone stages.")
        if output_dim % 4 != 0:
            raise ValueError("output_dim must be divisible by four for stage-wise pooling.")

        self.output_dim = int(output_dim)
        self.stage_dim = self.output_dim // 4
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, self.stage_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.stage_dim),
                    nn.GELU(),
                )
                for in_ch in in_channels
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _to_nchw(feature: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(f"Expected a 4D feature map, got shape {tuple(feature.shape)}")
        if feature.shape[1] == expected_channels:
            return feature
        if feature.shape[-1] == expected_channels:
            return feature.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f"Unable to determine channel axis for feature map with shape {tuple(feature.shape)} "
            f"and expected channel count {expected_channels}."
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        pooled = []
        for layer, feature in zip(self.layers, features):
            conv = layer[0]
            normalized = self._to_nchw(feature, expected_channels=conv.in_channels)
            projected = layer(normalized)
            pooled.append(self.pool(projected).flatten(1))
        return torch.cat(pooled, dim=1)


class BaseEncoderWrapper(nn.Module):
    def __init__(self, output_dim: int = 1024) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    @property
    def final_stage_module(self) -> nn.Module:
        return self.projector.layers[-1]


class TimmEncoderWrapper(BaseEncoderWrapper):
    """Unified `timm` wrapper for Swin and ConvNeXt backbones."""

    @staticmethod
    def _needs_explicit_img_size(model_name: str) -> bool:
        lowered = model_name.lower()
        return lowered.startswith("swin")

    def __init__(
        self,
        model_name: str,
        *,
        pretrained: bool,
        weight_path: str | None,
        image_size: int,
        output_dim: int = 1024,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        model_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(output_dim=output_dim)
        model_kwargs = dict(model_kwargs or {})
        local_weight_path = Path(weight_path).expanduser() if weight_path else None
        use_remote_pretrained = pretrained and (local_weight_path is None or not local_weight_path.is_file())
        create_kwargs = {
            "pretrained": use_remote_pretrained,
            "features_only": True,
            "out_indices": tuple(out_indices),
            **model_kwargs,
        }
        # Swin needs the training resolution baked into the patch embedding logic.
        # ConvNeXt and similar CNN backbones do not accept this argument.
        if self._needs_explicit_img_size(model_name):
            create_kwargs["img_size"] = image_size
        self.backbone = timm.create_model(
            model_name,
            **create_kwargs,
        )
        if local_weight_path is not None and local_weight_path.is_file():
            timm_load_checkpoint(self.backbone, str(local_weight_path), strict=False)
        self.feature_channels = list(self.backbone.feature_info.channels())
        self.projector = FeatureProjector(self.feature_channels, output_dim=self.output_dim)
        module_name = self.backbone.feature_info.module_name(out_indices[-1])
        self.backbone_final_stage = _resolve_module(self.backbone, module_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    MAMBAVISION_BASE_DIMS = {
        "tiny": 80,
        "tiny2": 80,
        "small": 96,
        "base": 128,
        "large": 196,
        "large2": 196,
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
        output_dim: int = 1024,
        model_kwargs: Dict[str, Any] | None = None,
        backbone_root: str | Path | None = None,
    ) -> None:
        super().__init__(output_dim=output_dim)
        self.family = family.lower()
        self.scale = scale.lower()
        self.image_size = image_size
        self.weight_path = weight_path
        self.model_kwargs = dict(model_kwargs or {})
        self.backbone_root = Path(backbone_root or Path(__file__).resolve().parent / "backbones")
        self._ensure_backbone_root_on_path()
        self.backbone = self._build_backbone(pretrained=pretrained)
        sample_channels = self._infer_stage_channels()
        self.projector = FeatureProjector(sample_channels, output_dim=self.output_dim)

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
        if hasattr(self.backbone, "dims"):
            dims = getattr(self.backbone, "dims")
            if isinstance(dims, Iterable):
                return [int(channel) for channel in list(dims)]

        if self.family == "mambavision":
            base_dim = int(self.MAMBAVISION_BASE_DIMS[self.scale])
            return [int(base_dim * (2**idx)) for idx in range(4)]

        raise RuntimeError(f"Unable to infer stage channels for backbone family '{self.family}'.")

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._forward_backbone(x)
        return self.projector(features)


def build_visual_encoder(config: Dict[str, Any]) -> BaseEncoderWrapper:
    visual_cfg = config["visual"]
    family = str(visual_cfg["family"]).lower()
    image_size = int(config["data"].get("image_size", 224))
    output_dim = int(config["model"].get("visual_feature_dim", 1024))
    model_kwargs = dict(visual_cfg.get("model_kwargs", {}))

    if family == "timm":
        return TimmEncoderWrapper(
            model_name=str(visual_cfg["name"]),
            pretrained=bool(visual_cfg.get("pretrained", True)),
            weight_path=visual_cfg.get("weight_path"),
            image_size=image_size,
            output_dim=output_dim,
            model_kwargs=model_kwargs,
        )

    if family in {"vmamba", "mambavision", "spatial_mamba"}:
        return MambaEncoderWrapper(
            family=family,
            scale=str(visual_cfg.get("scale", "tiny")),
            pretrained=bool(visual_cfg.get("pretrained", True)),
            weight_path=visual_cfg.get("weight_path"),
            image_size=image_size,
            output_dim=output_dim,
            model_kwargs=model_kwargs,
        )

    raise ValueError(f"Unsupported visual.family: {family}")
