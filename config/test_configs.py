from pydantic import Field
from typing import Dict, Any, Optional
from pathlib import Path

from .base import ConfigBase
from .common import ModelConfig, DataConfig, TransformConfig, MetricsConfig


class TestingConfig(ConfigBase):
    """
    Configuration for the testing a model.

    :param model_filename: Filename of the model to be tested (without .pth extension)
    :type model_filename: str
    :param data_folder: The folder where the model and its data is stored
    :type data_folder: str
    :param testing_strategy: The strategy to use for testing. Must be in 'TESTING_STRATEGY_REGISTRY'
    :type testing_strategy: str
    :param testing_strategy_params: Additional parameters for the testing strategy
    :type testing_strategy_params: Optional[Dict[str, Any]]
    :param verbose: Whether to print verbose output
    :type verbose: bool
    """
    model_filename: str = Field(..., description="Filename of the model to be tested (without .pth extension)")
    data_folder: str = Field(..., description="The folder where the model and its data is stored")
    testing_strategy: str = Field(..., description="The strategy to use for testing")
    testing_strategy_params: Optional[Dict[str, Any]] = Field({}, description="Additional parameters for the testing strategy")
    verbose: bool = Field(False, description="Whether to print verbose output")


class TestExperimentConfig(ConfigBase):
    """
    A hierarchical configuration class for organizing testing settings.
    This is the main configuration class for testing experiments.

    Structure:
        - testing_config (TestingConfig): General testing parameters (e.g., model filename, data folder).
        - model_settings_config (ModelConfig): Model architecture and related settings.
        - test_data_config (DataConfig): Testing dataset configuration.
        - transform_config (Optional[TransformConfig]): Data transformation configuration (optional).
        - metrics_config (Optional[MetricsConfig]): Metrics configuration (optional).
    Properties:
        - experiment_dir (Path): Path to the experiment directory, derived from training_config.
    Methods:
        - __str__: Returns a formatted string summarizing the experiment configuration.
    """
    testing_config: TestingConfig
    model_settings_config: ModelConfig
    test_data_config: DataConfig
    transform_config: Optional[TransformConfig] = Field(None, description="Transform configuration")
    metrics_config: Optional[MetricsConfig] = Field(None, description="Metrics configuration")

    @property
    def experiment_dir(self) -> Path:
        return Path(self.testing_config.data_folder)
    
    def __str__(self):
        info = f"Experiment Folder: {self.experiment_dir}\n"
        info += f"Model: {self.model_settings_config.architecture}\n"
        info += f"Model File: {self.testing_config.model_filename}\n"
        info += f"Dataset: {self.test_data_config.dataset_type}\n"
        info += f"Transforms: {self.transform_config.transforms if self.transform_config else 'None'}\n"
        info += f"Metrics: {', '.join(self.metrics_config.metrics)}\n"
        return info
