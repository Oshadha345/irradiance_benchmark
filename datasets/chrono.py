from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class SplitWindows:
    trainval_start: pd.Timestamp | None
    trainval_end: pd.Timestamp | None
    test_start: pd.Timestamp | None
    test_end: pd.Timestamp | None


def parse_optional_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp


def resolve_split_windows(config: dict[str, Any]) -> SplitWindows | None:
    data_cfg = config.get("data", {})
    split_cfg = data_cfg.get("split_windows") or {}
    if not isinstance(split_cfg, dict):
        return None

    windows = SplitWindows(
        trainval_start=parse_optional_timestamp(split_cfg.get("trainval_start")),
        trainval_end=parse_optional_timestamp(split_cfg.get("trainval_end")),
        test_start=parse_optional_timestamp(split_cfg.get("test_start")),
        test_end=parse_optional_timestamp(split_cfg.get("test_end")),
    )
    if all(value is None for value in (windows.trainval_start, windows.trainval_end, windows.test_start, windows.test_end)):
        return None
    return windows


def index_mask_between(index: pd.Index, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.Series:
    values = pd.DatetimeIndex(index)
    mask = pd.Series(True, index=values)
    if start is not None:
        mask &= values >= start
    if end is not None:
        mask &= values <= end
    return mask


def filter_frame_to_windows(df: pd.DataFrame, windows: SplitWindows) -> pd.DataFrame:
    starts = [value for value in (windows.trainval_start, windows.test_start) if value is not None]
    ends = [value for value in (windows.trainval_end, windows.test_end) if value is not None]
    start = min(starts) if starts else None
    end = max(ends) if ends else None
    mask = index_mask_between(df.index, start, end)
    return df.loc[mask.values]


def split_samples_by_windows(
    samples: Iterable[tuple[str, str | None, object]],
    windows: SplitWindows,
    *,
    val_split: float,
) -> tuple[list[int], list[int], list[int], str]:
    sample_list = list(samples)
    def in_range(value: object, start: pd.Timestamp | None, end: pd.Timestamp | None) -> bool:
        timestamp = pd.Timestamp(value)
        if start is not None and timestamp < start:
            return False
        if end is not None and timestamp > end:
            return False
        return True

    trainval_idx = [
        index
        for index, (_, _, dt) in enumerate(sample_list)
        if in_range(dt, windows.trainval_start, windows.trainval_end)
    ]
    test_idx = [
        index
        for index, (_, _, dt) in enumerate(sample_list)
        if in_range(dt, windows.test_start, windows.test_end)
    ]
    if len(trainval_idx) < 2 or not test_idx:
        raise RuntimeError("Custom timestamp split produced an empty partition.")

    val_len = max(1, int(len(trainval_idx) * val_split))
    val_len = min(val_len, len(trainval_idx) - 1)
    train_idx = trainval_idx[:-val_len]
    val_idx = trainval_idx[-val_len:]
    summary = (
        f"trainval={windows.trainval_start or 'min'}..{windows.trainval_end or 'max'} "
        f"test={windows.test_start or 'min'}..{windows.test_end or 'max'}"
    )
    return train_idx, val_idx, test_idx, summary


def renormalize_on_window(dataset: Any, start: pd.Timestamp | None, end: pd.Timestamp | None) -> None:
    cols_to_norm = ["k_index", "temperature", "pressure", "SZA", "Azimuth"]
    restored = dataset.df[dataset.feature_cols].copy()
    restored[cols_to_norm] = restored[cols_to_norm] * (dataset.std[cols_to_norm] + 1e-6) + dataset.mean[cols_to_norm]

    mask = index_mask_between(restored.index, start, end)
    stats_frame = restored.loc[mask.values]
    if stats_frame.empty:
        raise RuntimeError("Unable to recompute normalization statistics for the requested train/validation window.")

    dataset.mean = stats_frame[dataset.feature_cols].mean()
    dataset.std = stats_frame[dataset.feature_cols].std()
    dataset.df[cols_to_norm] = (restored[cols_to_norm] - dataset.mean[cols_to_norm]) / (dataset.std[cols_to_norm] + 1e-6)
