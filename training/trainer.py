from anyio import Path
from typing import Dict, Any
import torch
from tqdm import tqdm
import numpy as np

from configs import ExperimentConfig
from core.registry import MODEL_REGISTRY, TRAINING_STRATEGY_REGISTRY, DATASET_REGISTRY, LOADERS_REGISTRY, TRANSFORM_REGISTRY, LOSS_REGISTRY
from loggers_utils import TrainingLogger
from loggers_utils import log_execution
from utils import get_device, load_pkl_file, print_box


class UniversalTrainer:
    @log_execution("Configuring Trainer...", "Trainer configured successfully!")
    def __init__(self, config: ExperimentConfig):
        # Save configuration details
        self.config = config
        self.num_epochs = config.training_config.num_epochs
        self.model_filename = config.model_settings_config.name
        self.experiment_dir = config.experiment_dir

        # DEPRECATED: Get device; should be handled by the config file in the future
        self.device = get_device()

        # Setup logging
        self.logger = self._setup_logger()
        
        # Build components
        self.model = self._build_model()
        self.train_loader, self.val_loader, self.test_loader = self._build_loaders()
        self.loss_fn = self._build_loss()
        
        # Get training strategy based on model type
        self.strategy = TRAINING_STRATEGY_REGISTRY.build(
            config.model_settings_config.training_strategy,
            {}
        )
                
        # Setup optimizers using strategy
        self.optimizers = self.strategy.setup_optimizers(self.model, config, self.logger.get_optimizers_state())

        # Print configuration details
        print_box(str(config))

    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch using the model's strategy"""
        self.model.train()
        epoch_metrics = {}
        
        for batch in tqdm(self.train_loader, desc="Training"):
            batch = self._move_batch_to_device(batch)
            
            # Use strategy-specific training step
            loss, metrics = self.strategy.train_step(
                model=self.model,
                batch=batch,
                loss_fn=self.loss_fn,
                optimizer=self.optimizers
            )
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key == "outputs":
                    continue
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
        
        # Average metrics
        return {key: np.mean(values) for key, values in epoch_metrics.items()}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate one epoch using the model's strategy"""
        self.model.eval()
        epoch_metrics = {}
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            batch = self._move_batch_to_device(batch)
            
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
        """Main training loop - works for any model type"""
        best_val_loss = self.logger.get_best_val_loss()
        
        for epoch in range(self.num_epochs):
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Logging
            all_metrics = {**train_metrics, **val_metrics} # Aggregate all metrics
            self.logger.log(epoch, all_metrics)
            
            # Checkpointing (strategy-aware)
            primary_val_loss = self._get_primary_val_loss(val_metrics)
            if primary_val_loss < best_val_loss:
                best_val_loss = primary_val_loss
                self.model.save_checkpoint(
                    f"{self.model_filename}_best_model.pth",
                    self.experiment_dir
                )

            self.model.save_checkpoint(
                f"{self.model_filename}_epoch_{epoch}.pth",
                self.experiment_dir
            )

    def _build_model(self):
        """Build the model based on the configuration"""
        model_config = self.config.model_settings_config

        if not model_config:
            raise ValueError("Model configuration is required")
        
        model_architecture = MODEL_REGISTRY.get(model_config.architecture)

        if not model_architecture:
            raise ValueError(f"Model {model_config.architecture} not found in registry")

        model = model_architecture.from_config(model_config.params)

        if not self.logger.new:
            model.load_checkpoint(self.logger.get_best_model_path(), self.experiment_dir)

        model.to(self.device)
        return model
    
    def _build_loaders(self):
        """Build the dataset based on the configuration"""
        dataset_config = self.config.dataset_config

        # Ensure dataset configuration is provided
        if not dataset_config:
            raise ValueError("Dataset configuration is required")
        
        # Get the dataset class and loaders class from the registry
        dataset_class = DATASET_REGISTRY.get(dataset_config.dataset_type)
        if not dataset_class:
            raise ValueError(f"Dataset {dataset_config.dataset_type} not found in registry")
        loaders_class = LOADERS_REGISTRY.get(dataset_config.loader_type)
        if not loaders_class:
            raise ValueError(f"Data loader {dataset_config.loader_type} not found in registry")
        
        # Build the transform if specified
        transform = self._build_transform()
        
        # Get the path to the datasets
        train_data_path = dataset_config.train_data_path
        val_data_path = dataset_config.val_data_path
        test_data_path = dataset_config.test_data_path

        def load_data(data_path: str, type_data_path: str):
            if data_path is None:
                raise ValueError(f"{type_data_path} data path is required in the dataset configuration")
            
            if not Path(data_path).exists():
                raise FileNotFoundError(f"{type_data_path} data file not found: {data_path}")

            try:
                X_data, y_data = load_pkl_file(data_path)
            except ValueError as e:
                raise ValueError(f"Error loading train data from {train_data_path}: expected a tuple (X, Y) in the .pkl file. Original error: {e}")
            
            return X_data, y_data

        # Load the datasets
        X_train, y_train = load_data(train_data_path, "train")
        X_val, y_val = load_data(val_data_path, "validation")
        X_test, y_test = load_data(test_data_path, "test") if test_data_path else (None, None)

        # Create the dataset instances
        train_dataset = dataset_class(
            source=X_train,
            target=y_train,
            transform=transform
        )
        val_dataset = dataset_class(
            source=X_val,
            target=y_val,
            transform=transform
        )
        test_dataset = dataset_class(
            source=X_test,
            target=y_test,
            transform=transform
        ) if X_test is not None else None

        # Create the data loaders
        train_loader = loaders_class(
            dataset=train_dataset,
            batch_size=dataset_config.batch_size,
            num_workers=dataset_config.num_workers,
            shuffle=True,
            **dataset_config.loaders_params or {}
        )
        val_loader = loaders_class(
            dataset=val_dataset,
            batch_size=dataset_config.batch_size,
            num_workers=dataset_config.num_workers,
            shuffle=False,
            **dataset_config.loaders_params or {}
        )
        test_loader = loaders_class(
            dataset=test_dataset,
            batch_size=dataset_config.batch_size,
            num_workers=dataset_config.num_workers,
            shuffle=False,
            **dataset_config.loaders_params or {}
        ) if test_dataset else None
        return train_loader, val_loader, test_loader

    def _build_transform(self):
        """Build the transform based on the configuration"""
        transform_config = self.config.transform_config

        if not transform_config:
            return None
        
        # Handle both single and composed transforms
        transforms = transform_config.transforms
        params = transform_config.params or {}

        if isinstance(transforms, str):
            # Single transform case
            transform_class = TRANSFORM_REGISTRY.get(transforms)

            if not transform_class:
                raise ValueError(f"Transform {transforms} not found in registry")
            return transform_class.from_config(params)
        elif isinstance(transforms, list):
            transform_class = TRANSFORM_REGISTRY.get("compose")

            if not transform_class:
                raise ValueError("Transform compose not found in registry")
            return transform_class.from_config(params)
        else:
            raise ValueError(f"Invalid transform configuration: {transforms}")
        
    def _setup_logger(self):
        adversarial_models = ["gan"]  # DEPRECATED: Quick fix for adversarial models
        return TrainingLogger(
            save_dir=self.experiment_dir,
            adversarial_logger=True if self.config.model_settings_config.architecture in adversarial_models else False
        )
        
    def _build_loss(self):
        """Build the loss function based on the configuration"""
        loss_fn_config = self.config.loss_fn_config
        loss_fn_name = loss_fn_config.name
        loss_fn_params = loss_fn_config.params or {}

        loss_fn_class = LOSS_REGISTRY.get(loss_fn_name)
        if not loss_fn_class:
            raise ValueError(f"Loss function {loss_fn_name} not found in registry")

        return loss_fn_class.from_config(loss_fn_params)

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to the configured device"""
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

    def _get_primary_val_loss(self, val_metrics: Dict[str, float]) -> float:
        """Get the primary validation loss based on model type"""
        if "val_loss" in val_metrics:
            return val_metrics["val_loss"]
        elif "val_loss_G" in val_metrics:
            return val_metrics["val_loss_G"]
        elif "val_loss_NCE" in val_metrics:
            return val_metrics["val_loss_NCE"]
        else:
            # Return the first available metric
            return list(val_metrics.values())[0]
        