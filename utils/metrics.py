from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import numpy as np


EPS = 1e-8


def _to_numpy(array: Any) -> np.ndarray:
    if array is None:
        return None
    if isinstance(array, np.ndarray):
        return array.astype(np.float64, copy=False)
    try:
        import torch

        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy().astype(np.float64, copy=False)
    except Exception:
        pass
    return np.asarray(array, dtype=np.float64)


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {array.shape}.")
    return array


def _masked_pair(predictions: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(predictions) & np.isfinite(targets)
    return predictions[mask], targets[mask]


def rmse(predictions: Any, targets: Any, axis: int | None = 0) -> np.ndarray:
    pred, target = _to_numpy(predictions), _to_numpy(targets)
    return np.sqrt(np.nanmean((pred - target) ** 2, axis=axis))


def nrmse(
    predictions: Any,
    targets: Any,
    axis: int | None = 0,
    *,
    normalization: str = "mean",
) -> np.ndarray:
    pred, target = _to_numpy(predictions), _to_numpy(targets)
    error = rmse(pred, target, axis=axis)
    if normalization == "mean":
        denominator = np.nanmean(np.abs(target), axis=axis)
    elif normalization == "range":
        denominator = np.nanmax(target, axis=axis) - np.nanmin(target, axis=axis)
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")
    return (error / (denominator + EPS)) * 100.0


def mae(predictions: Any, targets: Any, axis: int | None = 0) -> np.ndarray:
    pred, target = _to_numpy(predictions), _to_numpy(targets)
    return np.nanmean(np.abs(pred - target), axis=axis)


def mbe(predictions: Any, targets: Any, axis: int | None = 0) -> np.ndarray:
    pred, target = _to_numpy(predictions), _to_numpy(targets)
    return np.nanmean(pred - target, axis=axis)


def r2_score(predictions: Any, targets: Any, axis: int | None = 0) -> np.ndarray:
    pred, target = _to_numpy(predictions), _to_numpy(targets)
    target_mean = np.nanmean(target, axis=axis, keepdims=True)
    ss_res = np.nansum((target - pred) ** 2, axis=axis)
    ss_tot = np.nansum((target - target_mean) ** 2, axis=axis)
    return 1.0 - (ss_res / (ss_tot + EPS))


def forecast_skill(
    predictions: Any,
    targets: Any,
    baseline: Any,
    axis: int | None = 0,
) -> np.ndarray:
    model_rmse = rmse(predictions, targets, axis=axis)
    persistence_rmse = rmse(baseline, targets, axis=axis)
    return (1.0 - (model_rmse / (persistence_rmse + EPS))) * 100.0


def metric_bundle(
    predictions: Any,
    targets: Any,
    baseline: Any | None = None,
    *,
    axis: int | None = 0,
    normalization: str = "mean",
) -> Dict[str, Any]:
    bundle = {
        "RMSE": np.asarray(rmse(predictions, targets, axis=axis)),
        "nRMSE": np.asarray(nrmse(predictions, targets, axis=axis, normalization=normalization)),
        "MAE": np.asarray(mae(predictions, targets, axis=axis)),
        "MBE": np.asarray(mbe(predictions, targets, axis=axis)),
        "R2": np.asarray(r2_score(predictions, targets, axis=axis)),
    }
    bundle["FS"] = (
        np.asarray(forecast_skill(predictions, targets, baseline, axis=axis))
        if baseline is not None
        else np.full_like(bundle["RMSE"], np.nan, dtype=np.float64)
    )
    return bundle


def summarize_multi_horizon_metrics(
    predictions: Any,
    targets: Any,
    *,
    baseline: Any | None = None,
    horizons: Sequence[int] | None = None,
    normalization: str = "mean",
) -> Dict[str, Dict[str, float]]:
    pred = _ensure_2d(_to_numpy(predictions))
    target = _ensure_2d(_to_numpy(targets))
    if pred.shape != target.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match target shape {target.shape}.")

    base = _ensure_2d(_to_numpy(baseline)) if baseline is not None else None
    if base is not None and base.shape != target.shape:
        raise ValueError(f"Baseline shape {base.shape} does not match target shape {target.shape}.")

    horizons = list(horizons) if horizons is not None else list(range(1, pred.shape[1] + 1))
    per_horizon = metric_bundle(pred, target, baseline=base, axis=0, normalization=normalization)
    flattened = metric_bundle(
        pred.reshape(-1, 1),
        target.reshape(-1, 1),
        baseline=base.reshape(-1, 1) if base is not None else None,
        axis=0,
        normalization=normalization,
    )

    summary: Dict[str, Dict[str, float]] = {}
    for index, horizon in enumerate(horizons):
        summary[str(horizon)] = {
            metric_name: float(metric_values[index])
            for metric_name, metric_values in per_horizon.items()
        }

    summary["overall"] = {metric_name: float(metric_values[0]) for metric_name, metric_values in flattened.items()}
    return summary
