from pydantic import Field
from typing import Dict, Any, Optional, List, Union

from .base import ConfigBase


class DataConfig(ConfigBase):
    """
    Configuration for data loading and preprocessing.
    Used to specify the dataset and loader configurations.

    :param dataset_type: Type of dataset. Must be in 'DATASET_REGISTRY'
    :type dataset_type: str
    :param loader_type: Type of data loader. Must be in 'LOADERS_REGISTRY'
    :type loader_type: str
    :param data_path: The path to the data to be used
    :type data_path: str
    :param batch_size: Batch size
    :type batch_size: int
    :param num_workers: Number of workers for data loading
    :type num_workers: int
    :param dataset_params: Additional dataset parameters
    :type dataset_params: Dict[str, Any]
    :param loaders_params: Additional loaders parameters
    :type loaders_params: Dict[str, Any]
    """
    dataset_type: str = Field(..., description="Type of dataset")
    loader_type: str = Field(..., description="Type of data loader")
    data_path: str = Field(..., description="Path to the dataset")
    batch_size: int = Field(4, description="Batch size")
    num_workers: int = Field(0, description="Number of workers for data loading")
    dataset_params: Optional[Dict[str, Any]] = Field({}, description="Additional dataset parameters")
    loaders_params: Optional[Dict[str, Any]] = Field({}, description="Additional data loader parameters")


class TransformConfig(ConfigBase):
    """
    Configuration for data transformations.

    :param transforms: Transformations to apply. Can be a single transform or a list of
    transforms, in which case the transformations will follow in the way they are
    defined in the list. For example ["per_image_asinh_normalize", "per_image_log_normalize"] -
    First asinh is applied then log normalization. Must be in the 'TRANSFORM_REGISTRY'
    :type transforms: Optional[Union[str, List[str]]]
    :param params: Additional parameters for transforms. For a single transform pass the relevant
    parameter and its value. Example:
        {
            param1: value1,
            param2: value2
        }
    For multiple transforms you need to provide a dictionary for each transform with its parameters. Example:
        {
            "per_image_asinh_normalize": {
                param1: value1,
                param2: value2
            },
            "per_image_log_normalize": {
                param1: value1,
                param2: value2
            }
        }
    :type params: Optional[Dict[str, Any]]
    """
    transforms: Optional[Union[str, List[str]]] = Field(None, description="Transforms to apply")
    params: Optional[Dict[str, Any]] = Field({}, description="Parameters for transforms")


class ModelConfig(ConfigBase):
    """
    Model configuration.

    :param architecture: Model architecture type. Must be in 'MODEL_REGISTRY'
    :type architecture: str
    :param params: Additional parameters to be passed to the model
    :type params: Optional[Dict[str, Any]]
    """
    architecture: str = Field(..., description="Model architecture type")
    params: Optional[Dict[str, Any]] = Field({}, description="Model parameters")


class MetricsConfig(ConfigBase):
    """
    Metrics configuration.

    :param metrics: List of metrics to evaluate
    :type metrics: List[str]
    :param params: Additional parameters for metrics
    :type params: Optional[Dict[str, Any]]
    """
    metrics: List[str] = Field(..., description="List of metrics to evaluate")
    params: Optional[Dict[str, Any]] = Field({}, description="Additional parameters for metrics")
