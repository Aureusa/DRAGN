from pydantic import Field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from .base import ConfigBase
from .common import ModelConfig, DataConfig, TransformConfig


class MetricsConfig(ConfigBase):
    metrics: List[str] = Field(..., description="List of metrics to evaluate")
    params: Optional[Dict[str, Any]] = Field({}, description="Additional parameters for metrics")


class TestingConfig(ConfigBase):
    model_filename: str = Field(..., description="Filename of the model to be tested (without .pth extension)")
    data_folder: str = Field(..., description="The folder where the model and its data is stored")
    testing_strategy: str = Field(..., description="The strategy to use for testing")
    verbose: bool = Field(False, description="Whether to print verbose output")


class TestExperimentConfig(ConfigBase):
    testing_config: TestingConfig
    model_settings_config: ModelConfig
    test_data_config: DataConfig
    transform_config: Optional[TransformConfig] = Field(None, description="Transform configuration")
    metrics_config: MetricsConfig
    
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
