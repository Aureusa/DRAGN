# dragn/training/strategies.py
from abc import ABC, abstractmethod
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
        loss = loss_fn(inputs, outputs, targets, psf)
        
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
            loss = loss_fn(inputs, outputs, targets, psf)

        return loss, {
            "val_loss": loss.item(),
            "outputs": outputs
        }
    
    def setup_optimizers(self, model, config, optimizer_state: Dict[str, Any] | None) -> Any:
        optimizer_config = config.optimizer_config
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            **optimizer_config.params
        )
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        return optimizer


@TRAINING_STRATEGY_REGISTRY.register("gan")
class GANTrainingStrategy(TrainingStrategy):
    """GAN training with generator and discriminator"""
    
    def train_step(self, model, batch: Dict[str, Any], loss_fn, optimizers) -> Tuple[torch.Tensor, Dict[str, Any]]:
        inputs = batch["input"]
        targets = batch["target"]
        
        optimizer_G, optimizer_D = optimizers
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        # Real samples
        real_outputs = model.discriminator(targets)
        real_labels = torch.ones_like(real_outputs)
        d_loss_real = loss_fn.discriminator_loss(real_outputs, real_labels)
        
        # Fake samples
        with torch.no_grad():
            fake_targets = model.generator(inputs)
        fake_outputs = model.discriminator(fake_targets.detach())
        fake_labels = torch.zeros_like(fake_outputs)
        d_loss_fake = loss_fn.discriminator_loss(fake_outputs, fake_labels)
        
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        
        fake_targets = model.generator(inputs)
        fake_outputs = model.discriminator(fake_targets)
        fake_labels = torch.ones_like(fake_outputs)  # Generator wants to fool discriminator
        
        g_loss_adv = loss_fn.generator_adversarial_loss(fake_outputs, fake_labels)
        g_loss_content = loss_fn.generator_content_loss(fake_targets, targets)
        g_loss = g_loss_adv + loss_fn.lambda_content * g_loss_content
        
        g_loss.backward()
        optimizer_G.step()
        
        return g_loss, {
            "train_loss_G": g_loss.item(),
            "train_loss_D": d_loss.item(),
            "train_loss_G_adv": g_loss_adv.item(),
            "train_loss_G_content": g_loss_content.item(),
            "outputs": fake_targets.detach()
        }
    
    def val_step(self, model, batch: Dict[str, Any], loss_fn) -> Tuple[torch.Tensor, Dict[str, Any]]:
        inputs = batch["input"]
        targets = batch["target"]
        
        with torch.no_grad():
            # Generator validation
            fake_targets = model.generator(inputs)
            g_loss_content = loss_fn.generator_content_loss(fake_targets, targets)
            
            # Discriminator validation
            real_outputs = model.discriminator(targets)
            fake_outputs = model.discriminator(fake_targets)
            
            real_labels = torch.ones_like(real_outputs)
            fake_labels = torch.zeros_like(fake_outputs)
            
            d_loss_real = loss_fn.discriminator_loss(real_outputs, real_labels)
            d_loss_fake = loss_fn.discriminator_loss(fake_outputs, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2
        
        return g_loss_content, {
            "val_loss_G": g_loss_content.item(),
            "val_loss_D": d_loss.item(),
            "outputs": fake_targets
        }

    def setup_optimizers(self, model, config, optimizers_state: Tuple[Dict[str, Any], Dict[str, Any]] | Tuple[None, None]) -> Tuple[Any, Any]:
        optimizer_G = torch.optim.Adam(
            model.generator.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            **config.optimizer.params
        )
        optimizer_D = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            **config.optimizer.params
        )

        if optimizers_state[0] is not None:
            optimizer_G.load_state_dict(optimizers_state[0])
        if optimizers_state[1] is not None:
            optimizer_D.load_state_dict(optimizers_state[1])

        return (optimizer_G, optimizer_D)
