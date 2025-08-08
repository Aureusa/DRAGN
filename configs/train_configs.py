from pydantic import Field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from .base import ConfigBase


class OptimizerConfig(ConfigBase):
    name: str = Field("adam", description="Optimizer type")
    lr: float = Field(0.001, description="Learning rate")
    weight_decay: float = Field(1e-4, description="Weight decay")
    params: Optional[Dict[str, Any]] = Field({}, description="Additional optimizer parameters")


class TrainingConfig(ConfigBase):
    num_epochs: int = Field(50, description="Number of training epochs", gt=0)
    log_every: int = Field(100, description="Log every N steps")
    save_every: int = Field(1000, description="Save model every N steps")
    early_stopping: Optional[int] = Field(None, description="Early stopping patience")
    gradient_clipping: Optional[float] = Field(None, description="Gradient clipping value")
    mixed_precision: bool = Field(True, description="Use mixed precision training")


class ModelConfig(ConfigBase):
    name: str = Field(..., description="The name of the model being used")
    architecture: str = Field(..., description="Model architecture type")
    training_strategy: str = Field("standard", description="Training strategy")
    params: Dict[str, Any] = Field(..., description="Model parameters")


class DatasetConfig(ConfigBase):
    dataset_type: str = Field(..., description="Type of dataset")
    loader_type: str = Field(..., description="Type of data loader")
    train_data_path: str = Field(..., description="Path to the training dataset")
    val_data_path: str = Field(..., description="Path to the validation dataset")
    test_data_path: Optional[str] = Field(None, description="Path to the test dataset")
    batch_size: int = Field(4, description="Batch size for training and validation")
    num_workers: int = Field(0, description="Number of workers for data loading")
    params: Optional[Dict[str, Any]] = Field({}, description="Additional dataset parameters")
    loaders_params: Optional[Dict[str, Any]] = Field({}, description="Parameters for data loaders")


class TransformConfig(ConfigBase):
    transforms: Optional[Union[str, List[str]]] = Field(None, description="Transforms to apply")
    params: Optional[Dict[str, Any]] = Field({}, description="Parameters for transforms")


class LossFnConfig(ConfigBase):
    name: str = Field("l1_loss", description="Loss function type")
    params: Optional[Dict[str, Any]] = Field({}, description="Parameters for the loss function")


class ExperimentConfig(ConfigBase):
    experiment_name: str = Field(..., description="Experiment name")
    model_settings_config: ModelConfig# = Field(..., description="Model configuration")
    dataset_config: DatasetConfig# = Field(..., description="Dataset configuration") 
    transform_config: Optional[TransformConfig] = Field(None, description="Transform configuration")
    loss_fn_config: LossFnConfig# = Field(..., description="Loss function configuration")
    optimizer_config: OptimizerConfig# = Field(..., description="Optimizer configuration")
    training_config: TrainingConfig# = Field(..., description="Training configuration")
    data_folder: str = Field("./experiments", description="Data folder for experiments")
    
    @property
    def experiment_dir(self) -> Path:
        return Path(self.data_folder) / self.experiment_name
    
    def __str__(self):
        info = f"Experiment: {self.experiment_name}\n"
        info += f"Model: {self.model_settings_config.name} ({self.model_settings_config.architecture})\n"
        info += f"Dataset: {self.dataset_config.dataset_type}\n"
        info += f"Transforms: {self.transform_config.transforms if self.transform_config else 'None'}\n"
        info += f"Loss Function: {self.loss_fn_config.name}\n"
        info += f"Optimizer: {self.optimizer_config.name} (lr={self.optimizer_config.lr})\n"
        info += f"Training: {self.training_config.num_epochs} epochs\n"
        info += f"Data Folder: {self.data_folder}\n"
        return info
    