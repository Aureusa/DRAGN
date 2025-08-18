from pydantic import Field
from typing import Dict, Any, Optional
from pathlib import Path

from .base import ConfigBase
from .common import DataConfig, TransformConfig, ModelConfig


class OptimizerConfig(ConfigBase):
    """
    Optimizer configuration.

    :param name: Optimizer type (right now only adam is supported, this is subject to change)
    :type name: str
    :param lr: Learning rate
    :type lr: float
    :param weight_decay: Weight decay
    :type weight_decay: float
    :param params: Additional optimizer parameters
    :type params: Optional[Dict[str, Any]]
    """
    name: str = Field("adam", description="Optimizer type")
    lr: float = Field(0.001, description="Learning rate")
    weight_decay: float = Field(1e-4, description="Weight decay")
    params: Optional[Dict[str, Any]] = Field({}, description="Additional optimizer parameters")


class TrainingConfig(ConfigBase):
    """
    Training configuration.

    :param model_filename: Filename to use for saving the model (without .pth extension)
    :type model_filename: str
    :param data_folder: The folder to store data relevant to training
    :type data_folder: str
    :param training_strategy: The strategy to use for training. Must be in
    'TRAINING_STRATEGY_REGISTRY'
    :type training_strategy: str
    :param num_epochs: Number of training epochs
    :type num_epochs: int
    :param log_epoch: Frequency of logging (in training steps)
    :type log_epoch: int
    :param save_epoch: Frequency of saving (in training steps)
    :type save_epoch: int
    :param training_logger_verbose: Whether or not to enable verbose logging for the training logger
    :type training_logger_verbose: bool
    :param checkpoint_manager_verbose: Whether or not to enable verbose logging for the checkpoint manager
    :type checkpoint_manager_verbose: bool
    """
    model_filename: str = Field(..., description="Filename to use for saving the model (without .pth extension)")
    data_folder: str = Field(..., description="The folder to store data relevant to training")
    training_strategy: str = Field(..., description="The strategy to use for training")
    num_epochs: int = Field(50, description="Number of training epochs", gt=0)
    log_every: int = Field(100, description="Log every N steps", gt=0)
    save_every: int = Field(1000, description="Save model every N steps", gt=0)
    training_logger_verbose: bool = Field(True, description="Enable verbose logging for the training logger")
    checkpoint_manager_verbose: bool = Field(True, description="Enable verbose logging for the checkpoint manager")


class LossFnConfig(ConfigBase):
    """
    Loss (Objective) function configuration.

    :param loss: the type of loss function to use. Must be in 'LOSS_REGISTRY'
    :type loss: str
    :param params: Additional parameters for the loss function
    :type params: Optional[Dict[str, Any]]
    """
    loss: str = Field("l1_loss", description="Loss function type")
    params: Optional[Dict[str, Any]] = Field({}, description="Parameters for the loss function")


class ExperimentConfig(ConfigBase):
    """
    A hierarchical configuration class for organizing experiment settings during training.
    This is the main configuration class for training experiments.
    
    Structure:
        - training_config (TrainingConfig): General training parameters (e.g., epochs, strategy, data folder, model filename).
        - model_settings_config (ModelConfig): Model architecture and related settings.
        - train_data_config (DataConfig): Training dataset configuration.
        - val_data_config (Optional[DataConfig]): Validation dataset configuration (optional).
        - transform_config (Optional[TransformConfig]): Data transformation configuration (optional).
        - loss_fn_config (LossFnConfig): Loss function configuration.
        - optimizer_config (OptimizerConfig): Optimizer settings (e.g., type, learning rate).
    Properties:
        - experiment_dir (Path): Path to the experiment directory, derived from training_config.
    Methods:
        - __str__: Returns a formatted string summarizing the experiment configuration.
    """
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
    