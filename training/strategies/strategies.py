from typing import Dict, Any, Tuple
import torch

from core.registry import TRAINING_STRATEGY_REGISTRY
from .base_strategy import TrainingStrategy


@TRAINING_STRATEGY_REGISTRY.register("standard")
class StandardTrainingStrategy(TrainingStrategy):
    """Standard training for UNet, ResNet, etc."""
    
    def train_step(self, model, batch: Dict[str, Any], loss_fn, optimizer) -> Tuple[torch.Tensor, Dict[str, Any]]:
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
    
    def val_step(self, model, batch: Dict[str, Any], loss_fn) -> Tuple[torch.Tensor, Dict[str, Any]]:
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
    
    def setup_optimizers(self, model, config) -> Any:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            **config.params
        )
        return optimizer
