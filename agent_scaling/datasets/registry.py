import json
from typing import Dict, List, Type, TypeVar

from .base import Dataset, DatasetInstance

_dataset_registry: Dict[str, Type[Dataset]] = {}

_dataset_instance_registry: Dict[str, Type[DatasetInstance]] = {}

T = TypeVar("T", bound=DatasetInstance)

TDataset = TypeVar("TDataset", bound=Dataset)


def register_dataset(name: str | List[str]):
    """Decorator to register a dataset class."""

    def decorator(cls: Type[TDataset]) -> Type[TDataset]:
        if isinstance(name, str):
            _dataset_registry[name] = cls
        else:
            for n in name:
                _dataset_registry[n] = cls
        return cls

    return decorator


def register_dataset_instance(name: str | List[str]):
    """Decorator to register a dataset instance class."""

    def decorator(cls: Type[T]) -> Type[T]:
        if isinstance(name, str):
            _dataset_instance_registry[name] = cls
        else:
            for n in name:
                _dataset_instance_registry[n] = cls
        return cls

    return decorator


def list_registered_datasets() -> list[str]:
    """List all registered dataset names."""
    return list(_dataset_registry.keys())


def list_registered_dataset_instances() -> list[str]:
    """List all registered dataset instance names."""
    return list(_dataset_instance_registry.keys())


def get_dataset_instance(dataset_id: str, **kwargs) -> DatasetInstance:
    if dataset_id not in _dataset_instance_registry:
        raise ValueError(
            f"Dataset instance '{dataset_id}' not found in registry. Dataset instances in registry are: {json.dumps(list(_dataset_instance_registry.keys()), indent=2)}"
        )
    return _dataset_instance_registry[dataset_id](**kwargs)


def get_dataset(dataset_id: str, **kwargs) -> Dataset:
    if dataset_id not in _dataset_registry:
        raise ValueError(
            f"Dataset '{dataset_id}' not found in registry. Datasets in registry are: {json.dumps(list(_dataset_registry.keys()), indent=2)}"
        )
    return _dataset_registry[dataset_id](**kwargs)


def get_dataset_cls(dataset_id: str) -> Type[Dataset]:
    if dataset_id not in _dataset_registry:
        raise ValueError(
            f"Dataset '{dataset_id}' not found in registry. Datasets in registry are: {json.dumps(list(_dataset_registry.keys()), indent=2)}"
        )
    return _dataset_registry[dataset_id]


def get_dataset_instance_cls(dataset_id: str) -> Type[DatasetInstance]:
    if dataset_id not in _dataset_instance_registry:
        raise ValueError(
            f"Dataset instance '{dataset_id}' not found in registry. Dataset instances in registry are: {json.dumps(list(_dataset_instance_registry.keys()), indent=2)}"
        )
    return _dataset_instance_registry[dataset_id]
