import torch
from typing import Any, Dict, Tuple

from config.train_configs import OptimizerConfig
from core.registry import TRAINING_STRATEGY_REGISTRY
from ..loss_functions import Loss
from networks.models import BaseModel
from .base_strategy import TrainingStrategy


@TRAINING_STRATEGY_REGISTRY.register("standard")
class StandardTrainingStrategy(TrainingStrategy):
    """
    Standard training for UNet, ResNet, etc.
    """
    def train_step(
            self,
            model: BaseModel,
            batch: Dict[str, Any],
            loss_fn: Loss,
            optimizer: torch.optim.Optimizer
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Execute one training step

        :param model: The model being trained
        :type model: BaseModel
        :param batch: The batch of data to train on
        :type batch: Dict[str, Any]
        :param loss_fn: The loss function to use
        :type loss_fn: Loss
        :param optimizer: The optimizer to use
        :type optimizer: torch.optim.Optimizer
        """
        inputs = batch["input"]
        targets = batch["target"]
        psf = inputs - targets
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = loss_fn(inputs, outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss, {
            "train_loss": loss.item(),
            "outputs": outputs.detach()
        }
    
    def val_step(self, model: BaseModel, batch: Dict[str, Any], loss_fn: Loss) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Execute one validation step

        :param model: The model being validated
        :type model: BaseModel
        :param batch: The batch of data to validate on
        :type batch: Dict[str, Any]
        :param loss_fn: The loss function to use
        :type loss_fn: Loss
        """
        inputs = batch["input"]
        targets = batch["target"]
        psf = inputs - targets
        
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_fn(inputs, outputs, targets)

        return loss, {
            "val_loss": loss.item(),
            "outputs": outputs
        }

    def setup_optimizers(self, model: BaseModel, config: OptimizerConfig) -> Any:
        """
        Setup optimizers for training.

        :param model: The model to optimize
        :type model: BaseModel
        :param config: The optimizer configuration
        :type config: OptimizerConfig
        """
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            **config.params
        )
        return optimizer
