from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch

class TrainingStrategy(ABC):
    """Base class for different training strategies"""
    def __init__(self, *args, **kwargs):
        del kwargs, args
    
    @abstractmethod
    def train_step(self, model, batch: Dict[str, Any], loss_fn, optimizer) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Execute one training step"""
        pass
    
    @abstractmethod
    def val_step(self, model, batch: Dict[str, Any], loss_fn) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Execute one validation step"""
        pass
    
    @abstractmethod
    def setup_optimizers(self, model, config, optimizers_state) -> Any:
        """Setup optimizers for this strategy"""
        pass
