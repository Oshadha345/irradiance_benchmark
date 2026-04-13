from __future__ import annotations

import json
import random
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_if_relative(base_dir: Path, value: Any) -> Any:
    if value is None or not isinstance(value, str):
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        return value
    return str((PROJECT_ROOT / candidate).resolve())


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["visual"]["weight_path"] = _resolve_if_relative(config_path.parent, config["visual"].get("weight_path"))
    # The double-blind MERCon baseline uses a fixed 40-step temporal window across datasets.
    config.setdefault("data", {})
    config.setdefault("model", {})
    config["data"]["sequence_length"] = int(config["model"].get("baseline_sequence_length", 40))
    return config


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_slug(config: Dict[str, Any]) -> str:
    visual_cfg = config["visual"]
    family = str(visual_cfg["family"]).lower()
    if family == "timm":
        model_name = str(visual_cfg["name"]).replace("/", "_")
        scale = str(visual_cfg.get("scale", model_name.split("_")[-1]))
    else:
        model_name = family
        scale = str(visual_cfg.get("scale", "default"))
    return f"{model_name}_{scale}"


def slugify_comment(comment: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", comment.strip()).strip("_").lower()
    return slug


def create_run_dir(
    config: Dict[str, Any],
    output_root: str | Path | None = None,
    comment: str | None = None,
) -> Path:
    dataset = str(config["data"]["dataset_type"]).lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(output_root or PROJECT_ROOT / "results")
    parts = [model_slug(config)]
    comment_slug = slugify_comment(comment or "")
    if comment_slug:
        parts.append(comment_slug)
    parts.append(timestamp)
    run_dir = root / dataset / "_".join(parts)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        cleaned_key = key.replace("_orig_mod.", "").replace("module.", "")
        cleaned[cleaned_key] = value
    return cleaned


def extract_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "state_dict" in checkpoint:
        return clean_state_dict(checkpoint["state_dict"])
    if "model_state_dict" in checkpoint:
        return clean_state_dict(checkpoint["model_state_dict"])
    return clean_state_dict(checkpoint)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = extract_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    return {
        "checkpoint": checkpoint,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def config_to_jsonable(config: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy(config)
