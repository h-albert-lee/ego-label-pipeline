"""Dataset adapters for the object-ownership labeling pipeline."""

from egoownership.datasets.adapters import (
    LabelingDatasetAdapter,
    get_dataset_adapter,
    normalize_dataset_id,
    register_dataset_adapter,
)

__all__ = [
    "LabelingDatasetAdapter",
    "get_dataset_adapter",
    "normalize_dataset_id",
    "register_dataset_adapter",
]
