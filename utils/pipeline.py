from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from datasets import unwrap_dataset
from utils.metrics import summarize_multi_horizon_metrics


class BoundaryLoss(nn.Module):
    """Sobel edge loss for auxiliary next-frame supervision."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3),
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = prediction.size(1)
        pred_x = F.conv2d(prediction, self.sobel_x.repeat(channels, 1, 1, 1), padding=1, groups=channels)
        pred_y = F.conv2d(prediction, self.sobel_y.repeat(channels, 1, 1, 1), padding=1, groups=channels)
        target_x = F.conv2d(target, self.sobel_x.repeat(channels, 1, 1, 1), padding=1, groups=channels)
        target_y = F.conv2d(target, self.sobel_y.repeat(channels, 1, 1, 1), padding=1, groups=channels)
        return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


def dataset_statistics(dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
    dataset = unwrap_dataset(dataloader.dataset)
    return {
        "k_mean": float(dataset.mean["k_index"]),
        "k_std": float(dataset.std["k_index"]),
    }


def batch_to_device(batch: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(item.to(device, non_blocking=True) if hasattr(item, "to") else item for item in batch)


def denormalize_k_index(values: np.ndarray, k_mean: float, k_std: float) -> np.ndarray:
    return (values * k_std) + k_mean


def select_primary_horizon(values: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if values.ndim == 1:
        return values[..., None]
    return values[:, :1]


def run_inference(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    use_aux_decoder: bool,
    criterion: nn.Module | None = None,
    desc: str = "Evaluating",
) -> Dict[str, Any]:
    stats = dataset_statistics(dataloader)
    model.eval()
    losses = []
    pred_k, target_k, baseline_k = [], [], []
    pred_ghi, target_ghi, baseline_ghi = [], [], []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            batch = batch_to_device(batch, device)
            images, weather_seq, targets, ghi_cs = batch[:4]
            targets = select_primary_horizon(targets)
            ghi_cs = select_primary_horizon(ghi_cs)
            predictions, _ = model(images, weather_seq)
            if criterion is not None:
                losses.append(float(criterion(predictions, targets).item()))

            pred_np = select_primary_horizon(predictions.detach().cpu().numpy())
            target_np = select_primary_horizon(targets.detach().cpu().numpy())
            ghi_np = select_primary_horizon(ghi_cs.detach().cpu().numpy())
            persistence_np = weather_seq[:, -1, 0].detach().cpu().numpy()[:, None]

            pred_real = denormalize_k_index(pred_np, stats["k_mean"], stats["k_std"])
            target_real = denormalize_k_index(target_np, stats["k_mean"], stats["k_std"])
            persistence_real = denormalize_k_index(persistence_np, stats["k_mean"], stats["k_std"])

            pred_k.append(pred_real)
            target_k.append(target_real)
            baseline_k.append(persistence_real)
            pred_ghi.append(pred_real * ghi_np)
            target_ghi.append(target_real * ghi_np)
            baseline_ghi.append(persistence_real * ghi_np)

    if not pred_k:
        raise RuntimeError("No samples were produced by the dataloader. Check dataset paths and split settings.")

    outputs = {
        "pred_k": np.concatenate(pred_k, axis=0),
        "target_k": np.concatenate(target_k, axis=0),
        "baseline_k": np.concatenate(baseline_k, axis=0),
        "pred_ghi": np.concatenate(pred_ghi, axis=0),
        "target_ghi": np.concatenate(target_ghi, axis=0),
        "baseline_ghi": np.concatenate(baseline_ghi, axis=0),
        "loss": float(np.mean(losses)) if losses else None,
    }
    return outputs


def build_metric_report(
    outputs: Dict[str, Any],
    horizons: list[int],
) -> Dict[str, Any]:
    primary_horizon = [int(horizons[0])] if horizons else [10]
    return {
        "ghi": summarize_multi_horizon_metrics(
            predictions=outputs["pred_ghi"],
            targets=outputs["target_ghi"],
            baseline=outputs["baseline_ghi"],
            horizons=primary_horizon,
        ),
        "k_index": summarize_multi_horizon_metrics(
            predictions=outputs["pred_k"],
            targets=outputs["target_k"],
            baseline=outputs["baseline_k"],
            horizons=primary_horizon,
        ),
        "loss": outputs["loss"],
    }
