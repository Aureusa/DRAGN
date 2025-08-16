from pydantic import Field
from typing import Dict, Any, Optional
from pathlib import Path

from .base import ConfigBase
from .common import DataConfig, TransformConfig, ModelConfig


class OptimizerConfig(ConfigBase):
    name: str = Field("adam", description="Optimizer type")
    lr: float = Field(0.001, description="Learning rate")
    weight_decay: float = Field(1e-4, description="Weight decay")
    params: Optional[Dict[str, Any]] = Field({}, description="Additional optimizer parameters")


class TrainingConfig(ConfigBase):
    model_filename: str = Field(..., description="Filename to use for saving the model (without .pth extension)")
    data_folder: str = Field(..., description="The folder to store data relevant to training")
    training_strategy: str = Field(..., description="The strategy to use for training")
    num_epochs: int = Field(50, description="Number of training epochs", gt=0)
    log_every: int = Field(100, description="Log every N steps")
    save_every: int = Field(1000, description="Save model every N steps")
    training_logger_verbose: bool = Field(True, description="Enable verbose logging for the training logger")
    checkpoint_manager_verbose: bool = Field(True, description="Enable verbose logging for the checkpoint manager")


class LossFnConfig(ConfigBase):
    loss: str = Field("l1_loss", description="Loss function type")
    params: Optional[Dict[str, Any]] = Field({}, description="Parameters for the loss function")


class ExperimentConfig(ConfigBase):
    training_config: TrainingConfig
    model_settings_config: ModelConfig
    train_data_config: DataConfig
    val_data_config: Optional[DataConfig] = Field(None, description="Validation data configuration")
    transform_config: Optional[TransformConfig] = Field(None, description="Transform configuration")
    loss_fn_config: LossFnConfig
    optimizer_config: OptimizerConfig
    
    @property
    def experiment_dir(self) -> Path:
        return Path(self.training_config.data_folder)

    def __str__(self):
        info = f"Experiment Folder: {self.experiment_dir}\n"
        info += f"Model: {self.model_settings_config.architecture}\n"
        info += f"Model File: {self.training_config.model_filename}\n"
        info += f"Dataset: {self.train_data_config.dataset_type}\n"
        info += f"Transforms: {self.transform_config.transforms if self.transform_config else 'None'}\n"
        info += f"Loss Function: {self.loss_fn_config.loss}\n"
        info += f"Optimizer: {self.optimizer_config.name} (lr={self.optimizer_config.lr})\n"
        info += f"Training: {self.training_config.num_epochs} epochs\n"
        info += f"Training Strategy: {self.training_config.training_strategy}\n"
        return info
    