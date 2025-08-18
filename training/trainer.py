from typing import Dict

import numpy as np
from tqdm import tqdm

from config import ExperimentConfig
from core.registry import LOSS_REGISTRY, TRAINING_STRATEGY_REGISTRY
from utils import print_box
from utils.building import build_loaders, build_model, build_transform
from utils.device import get_device, move_batch_to_device
from utils.log_utils import log_execution

from .loss_functions import Loss
from .checkpoint_manager import CheckpointManager
from .training_logger import TrainingLogger


class UniversalTrainer:
    """Universal trainer that uses configuration-driven training strategies"""
    @log_execution("Configuring Trainer...", "Trainer configured successfully!")
    def __init__(self, config: ExperimentConfig) -> None:
        """
        Initialize the UniversalTrainer with the given configuration.

        :param config: Experiment configuration object.
        :type config: ExperimentConfig
        """
        self.config = config

        # DEPRECATED: Get device; should be handled by the config file in the future
        self.device = get_device()

        # Build components
        self.model = build_model(config.model_settings_config).to(self.device)
        self.transform = build_transform(config.transform_config) if config.transform_config else None
        self.train_loader = build_loaders(config.train_data_config, self.transform)
        self.val_loader = build_loaders(config.val_data_config, self.transform)
        self.loss_fn = self._build_loss()
        
        # Get training strategy based on model type
        self.strategy = TRAINING_STRATEGY_REGISTRY.build(
            config.training_config.training_strategy,
            {}
        )
                
        # Setup optimizers using strategy
        self.optimizers = self.strategy.setup_optimizers(self.model, config.optimizer_config)

        # Save configuration details
        self.num_epochs = config.training_config.num_epochs
        self.model_filename = config.training_config.model_filename
        self.experiment_dir = config.experiment_dir
        self.log_every = config.training_config.log_every
        self.save_every = config.training_config.save_every
        self.max_steps = config.training_config.num_epochs * len(self.train_loader)
        self.current_step = 0

        if not self.experiment_dir.exists():
            self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.experiment_dir,
            model_filename=self.model_filename,
            device=self.device,
            verbose=config.training_config.checkpoint_manager_verbose
        )

        # Training Logger
        self.logger = TrainingLogger(
            save_dir=self.experiment_dir,
            verbose=config.training_config.training_logger_verbose
        )

        # Load checkpoint if exists
        if self.checkpoint_manager.checkpoint_exists():
            self.model, self.optimizers, self.current_step = self.checkpoint_manager.load_checkpoint(self.model, self.optimizers)

        # Print configuration details
        print_box(str(config))

    def train_epoch(self) -> Dict[str, float]:
        """
        Train one epoch using the model's strategy
        """
        self.model.train()
        epoch_metrics = {}
        
        for batch in tqdm(self.train_loader, desc="Training"):
            batch = move_batch_to_device(batch, self.device)
            
            # Use strategy-specific training step
            loss, metrics = self.strategy.train_step(
                model=self.model,
                batch=batch,
                loss_fn=self.loss_fn,
                optimizer=self.optimizers
            )

            self.current_step += 1

            # Accumulate metrics
            for key, value in metrics.items():
                if key == "outputs":
                    continue
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)

            # Logging
            if self.current_step % self.log_every == 0:
                metrics2log = {name: value for name, value in metrics.items() if name != "outputs"}
                self.logger.log_step(self.current_step, metrics2log)

            # Saving
            if self.current_step % self.save_every == 0:
                self.checkpoint_manager.save_checkpoint(self.current_step, self.model, self.optimizers)

        # Average metrics
        return {key: np.mean(values) for key, values in epoch_metrics.items()}
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate one epoch using the model's strategy
        """
        self.model.eval()
        epoch_metrics = {}
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            batch = move_batch_to_device(batch, self.device)

            # Use strategy-specific validation step
            loss, metrics = self.strategy.val_step(
                model=self.model,
                batch=batch,
                loss_fn=self.loss_fn
            )
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
        
        return {key: np.mean(values) for key, values in epoch_metrics.items()}
    
    def train(self):
        """
        Main training loop. Trains the model for a specified number of epochs.
        """
        best_val_loss = self.logger.get_best_val_loss()
        
        for epoch in range(self.num_epochs):
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch() if self.val_loader else {}
            
            # Logging
            all_metrics = {**train_metrics, **val_metrics} # Aggregate all metrics
            self.logger.log_epoch(all_metrics)

            # Saving if best model
            if self.val_loader:
                # Checkpointing (strategy-aware)
                primary_val_loss = self._get_primary_val_loss(val_metrics)
                if primary_val_loss < best_val_loss:
                    self.checkpoint_manager.save_checkpoint(
                        self.current_step,
                        self.model,
                        self.optimizers,
                        is_best=True
                    )

        self.checkpoint_manager.save_checkpoint(
            self.current_step,
            self.model,
            self.optimizers,
            is_last=True
        )
        
    def _build_loss(self) -> Loss:
        """
        Build the loss function based on the configuration

        :return: The loss function
        :rtype: Loss
        """
        loss_fn_config = self.config.loss_fn_config
        loss_fn_name = loss_fn_config.loss
        loss_fn_params = loss_fn_config.params or {}

        loss_fn_class = LOSS_REGISTRY.get(loss_fn_name)
        if not loss_fn_class:
            raise ValueError(f"Loss function {loss_fn_name} not found in registry")

        return loss_fn_class.from_config(loss_fn_params)

    def _get_primary_val_loss(self, val_metrics: Dict[str, float]) -> float:
        """
        Get the primary validation loss based on model type

        :param val_metrics: Dictionary containing validation metrics
        :type val_metrics: Dict[str, float]
        :return: The primary validation loss
        :rtype: float
        """
        if "val_loss" in val_metrics:
            return val_metrics["val_loss"]
        if "val_loss_G" in val_metrics:
            return val_metrics["val_loss_G"]
        if "val_loss_NCE" in val_metrics:
            return val_metrics["val_loss_NCE"]
        return float("inf")
