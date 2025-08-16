from pydantic import Field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from .base import ConfigBase


class DataConfig(ConfigBase):
    dataset_type: str = Field(..., description="Type of dataset")
    loader_type: str = Field(..., description="Type of data loader")
    data_path: str = Field(..., description="Path to the dataset")
    batch_size: int = Field(4, description="Batch size")
    num_workers: int = Field(0, description="Number of workers for data loading")
    dataset_params: Optional[Dict[str, Any]] = Field({}, description="Additional dataset parameters")
    loaders_params: Optional[Dict[str, Any]] = Field({}, description="Additional data loader parameters")


class TransformConfig(ConfigBase):
    transforms: Optional[Union[str, List[str]]] = Field(None, description="Transforms to apply")
    params: Optional[Dict[str, Any]] = Field({}, description="Parameters for transforms")


class ModelConfig(ConfigBase):
    architecture: str = Field(..., description="Model architecture type")
    params: Optional[Dict[str, Any]] = Field({}, description="Model parameters")


class MetricsConfig(ConfigBase):
    metrics: List[str] = Field(..., description="List of metrics to evaluate")
    params: Optional[Dict[str, Any]] = Field({}, description="Additional parameters for metrics")
