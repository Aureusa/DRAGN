from pydantic import Field
from typing import Dict, Any, Optional
from pathlib import Path

from .base import ConfigBase
from .common import DataConfig, TransformConfig, MetricsConfig


class SingleModelConfig(ConfigBase):
    """
    Configuration for a single model in a multi-model testing setup.

    :param architecture: Model architecture type. Must be in 'MODEL_REGISTRY'
    :type architecture: str
    :param full_filepath: Full file path to the model
    :type full_filepath: str
    :param params: Model parameters
    :type params: Optional[Dict[str, Any]]
    :param transform_config: Transform configuration
    :type transform_config: Optional[TransformConfig]
    """
    architecture: str = Field(..., description="Model architecture type")
    full_filepath: str = Field(..., description="Full file path to the model")
    params: Optional[Dict[str, Any]] = Field({}, description="Model parameters")
    transform_config: Optional[TransformConfig] = Field(None, description="Transform configuration")

    def get_loading_args(self):
        return {
            "filename": self.full_filepath.split("/")[-1],
            "dir_": "/".join(self.full_filepath.split("/")[:-1])
        }


class MultiModelTestingConfig(ConfigBase):
    """
    Configuration for multi-model testing setup.

    :param data_folder: The path to the folder to save the results to
    :type data_folder: str
    :param testing_strategy: The strategy to use for testing. Must be in 'TESTING_STRATEGY_REGISTRY'
    :type testing_strategy: str
    :param testing_strategy_params: Additional parameters for the testing strategy
    :type testing_strategy_params: Optional[Dict[str, Any]]
    :param verbose: Whether to print verbose output
    :type verbose: bool
    """
    data_folder: str = Field(..., description="The folder where to save the test results")
    testing_strategy: str = Field(..., description="The strategy to use for testing")
    testing_strategy_params: Optional[Dict[str, Any]] = Field({}, description="Additional parameters for the testing strategy")
    verbose: bool = Field(False, description="Whether to print verbose output")


class MultiModelTestExperimentConfig(ConfigBase):
    """
    A hierarchical configuration class for organizing testing settings for multi model testing.
    This is the main configuration class for testing multi model experiments.

    Structure:
        - multi_model_testing_config (MultiModelTestingConfig): General testing parameters (e.g., data folder, testing strategy).
        - models_settings_config (Dict[str, SingleModelConfig]): Model architecture and related settings for each model.
        - data_config (DataConfig): Testing dataset configuration.
        - metrics_config (Optional[MetricsConfig]): Metrics configuration (optional).
    Properties:
        - experiment_dir (Path): Path to the experiment directory, derived from training_config.
    Methods:
        - __str__: Returns a formatted string summarizing the experiment configuration.
    """
    multi_model_testing_config: MultiModelTestingConfig
    models_settings_config: Dict[str, SingleModelConfig]
    data_config: DataConfig
    metrics_config: Optional[MetricsConfig] = Field(None, description="Metrics configuration")
    
    @property
    def experiment_dir(self) -> Path:
        return Path(self.multi_model_testing_config.data_folder)

    def __str__(self):
        info = f"Experiment Folder: {self.experiment_dir}\n"
        info += f"Models: {list(self.models_settings_config.keys())}\n"
        info += f"Testing Strategy: {self.multi_model_testing_config.testing_strategy}\n"
        return info
