from __future__ import annotations

from typing import Any, Dict, Tuple

from torch.utils.data import DataLoader, Dataset, Subset


def get_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset_type = str(config["data"]["dataset_type"]).lower()
    if dataset_type == "folsom":
        from .folsom_loader import get_data_loaders as _get_data_loaders
    elif dataset_type == "nrel":
        from .nrel_loader import get_data_loaders as _get_data_loaders
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")
    return _get_data_loaders(config)


def unwrap_dataset(dataset: Dataset) -> Dataset:
    current = dataset
    while isinstance(current, Subset):
        current = current.dataset
    return current

