from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


EPS = 1e-8


def _to_numpy(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array.astype(np.float64, copy=False)
    try:
        import torch

        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy().astype(np.float64, copy=False)
    except Exception:
        pass
    return np.asarray(array, dtype=np.float64)


def _ensure_single_horizon(array: Any) -> np.ndarray:
    values = _to_numpy(array)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError(f"Expected a single-horizon array with shape (B, 1), got {values.shape}.")
    return values


def rmse(predictions: Any, targets: Any) -> float:
    pred = _ensure_single_horizon(predictions)
    target = _ensure_single_horizon(targets)
    return float(np.sqrt(np.nanmean((pred - target) ** 2)))


def nrmse(predictions: Any, targets: Any, *, normalization: str = "mean") -> float:
    pred = _ensure_single_horizon(predictions)
    target = _ensure_single_horizon(targets)
    error = rmse(pred, target)
    if normalization == "mean":
        denominator = float(np.nanmean(np.abs(target)))
    elif normalization == "range":
        denominator = float(np.nanmax(target) - np.nanmin(target))
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")
    return float((error / (denominator + EPS)) * 100.0)


def mae(predictions: Any, targets: Any) -> float:
    pred = _ensure_single_horizon(predictions)
    target = _ensure_single_horizon(targets)
    return float(np.nanmean(np.abs(pred - target)))


def mbe(predictions: Any, targets: Any) -> float:
    pred = _ensure_single_horizon(predictions)
    target = _ensure_single_horizon(targets)
    return float(np.nanmean(pred - target))


def r2_score(predictions: Any, targets: Any) -> float:
    pred = _ensure_single_horizon(predictions)
    target = _ensure_single_horizon(targets)
    target_mean = float(np.nanmean(target))
    ss_res = float(np.nansum((target - pred) ** 2))
    ss_tot = float(np.nansum((target - target_mean) ** 2))
    return float(1.0 - (ss_res / (ss_tot + EPS)))


def forecast_skill(predictions: Any, targets: Any, baseline: Any) -> float:
    model_rmse = rmse(predictions, targets)
    persistence_rmse = rmse(baseline, targets)
    return float((1.0 - (model_rmse / (persistence_rmse + EPS))) * 100.0)


def summarize_single_horizon_metrics(
    predictions: Any,
    targets: Any,
    *,
    baseline: Any | None = None,
    normalization: str = "mean",
) -> Dict[str, float]:
    summary = {
        "RMSE": rmse(predictions, targets),
        "nRMSE": nrmse(predictions, targets, normalization=normalization),
        "MAE": mae(predictions, targets),
        "MBE": mbe(predictions, targets),
        "R2": r2_score(predictions, targets),
        "FS": float("nan") if baseline is None else forecast_skill(predictions, targets, baseline),
    }
    return summary


def summarize_multi_horizon_metrics(
    predictions: Any,
    targets: Any,
    *,
    baseline: Any | None = None,
    horizons: Sequence[int] | None = None,
    normalization: str = "mean",
) -> Dict[str, Dict[str, float]]:
    """Backward-compatible wrapper around the new single-horizon metric bundle."""

    label = str(list(horizons)[0]) if horizons else "10"
    overall = summarize_single_horizon_metrics(
        predictions=predictions,
        targets=targets,
        baseline=baseline,
        normalization=normalization,
    )
    return {label: overall, "overall": overall}
