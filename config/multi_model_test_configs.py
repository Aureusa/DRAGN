from pydantic import Field
from typing import Dict, Any, Optional
from pathlib import Path

from .base import ConfigBase
from .common import DataConfig, TransformConfig, MetricsConfig


class SingleModelConfig(ConfigBase):
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
    data_folder: str = Field(..., description="The folder where to save the test results")
    testing_strategy: str = Field(..., description="The strategy to use for testing")
    testing_strategy_params: Optional[Dict[str, Any]] = Field({}, description="Additional parameters for the testing strategy")
    verbose: bool = Field(False, description="Whether to print verbose output")


class MultiModelTestExperimentConfig(ConfigBase):
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
