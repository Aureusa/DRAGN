from pathlib import Path

from typing import List

from testing.metrics import Metric
from data.transforms import _BaseTransform
from data.loaders import _BaseLoader
from config.common import DataConfig, TransformConfig, ModelConfig, MetricsConfig
from core.registry import METRICS_REGISTRY, MODEL_REGISTRY, DATASET_REGISTRY, LOADERS_REGISTRY, TRANSFORM_REGISTRY
from networks.models._base_model import BaseModel
from utils.persistence import load_pkl_file


def build_model(config: ModelConfig) -> BaseModel:
    """
    Build the model based on the configuration. Optionally if dir_ is passed
    load the model with the corresponding checkpoint.

    :param config: The model configuration.
    :type config: ModelConfig
    :param dir_: The directory to load the checkpoint from. If None instantiates
    a fresh model.
    :type dir_: str | None
    """
    architecture = config.architecture
    params = config.params

    # Build the model
    model = MODEL_REGISTRY.build(
        architecture,
        params
    )

    if not model:
        raise ValueError(f"Model `{architecture}` not found in registry")

    return model

def build_loaders(config: DataConfig, transform: _BaseTransform | None = None) -> _BaseLoader:
    """
    Build the dataset based on the configuration as well as optionally
    passing a transform.

    :param config: The data configuration.
    :type config: DataConfig
    :param transform: The transform to apply to the data.
    :type transform: _BaseTransform | None
    """
    dataset_type = config.dataset_type
    loader_type = config.loader_type
    data_path = config.data_path
    batch_size = config.batch_size
    num_workers = config.num_workers
    dataset_params = config.dataset_params
    loaders_params = config.loaders_params

    # Get the dataset and loader
    dataset = DATASET_REGISTRY.get(dataset_type)
    loader = LOADERS_REGISTRY.get(loader_type)

    # Load the data
    input_data, target_data = load_pkl_file(data_path)
    
    # Create the dataset instance
    dataset_obj = dataset(
        input_data,
        target_data,
        transform=transform,
        **dataset_params
    )

    # Create the data loader instance
    loader_obj = loader(
        dataset_obj,
        batch_size=batch_size,
        num_workers=num_workers,
        **loaders_params
    )
    return loader_obj

def build_transform(config: TransformConfig) -> _BaseTransform:
    """
    Build the transform based on the configuration.

    :param config: The transform configuration.
    :type config: TransformConfig
    """
    # Handle both single and composed transforms
    transforms = config.transforms
    params = config.params or {}

    if isinstance(transforms, str):
        # Single transform case
        transform_class = TRANSFORM_REGISTRY.get(transforms)

        if not transform_class:
            raise ValueError(f"Transform {transforms} not found in registry")
        return transform_class.from_config(params)
    elif isinstance(transforms, list):
        transform_class = TRANSFORM_REGISTRY.get("compose")

        if not transform_class:
            raise ValueError("Transform compose not found in registry")
        return transform_class.from_config(params)
    else:
        raise ValueError(f"Invalid transform configuration, should be a string or a list of strings: {transforms}")

def build_metrics(config: MetricsConfig) -> List[Metric]:
        """Build metrics"""
        metrics = []
        for metric_name in config.metrics:
            metric_type = METRICS_REGISTRY.get(
                metric_name,
            )
            metrics.append(metric_type(**config.params))
        return metrics
