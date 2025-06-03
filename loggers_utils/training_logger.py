"""
NOTE FOR USERS:

This module provides the `TrainingLogger` class for managing and saving the
training history of machine learning models, including support for adversarial (GAN) training.
It tracks training and validation losses, best validation loss, and optimizer
states, and saves them to disk for later analysis or resuming training.

**Key Points:**
- Use `TrainingLogger` to log losses and optimizer states during training.
- Supports both standard and adversarial (GAN) training workflows.
- Training history is saved as a JSON file; optimizer states are saved as `.pth` files.
- Automatically resumes from existing history if present in the specified directory.

**Important:**
- Specify a directory for saving logs and optimizer states when initializing.
- For GANs, set `adverserial_logger=True` to log discriminator losses and states.
- For more details, see the class docstring or contact the maintainers.

Example usage:

from loggers_utils.training_logger import TrainingLogger
import torch

# Initialize logger (for standard training)
logger = TrainingLogger(save_dir="./logs")

# During training loop:
for epoch in range(num_epochs):
    train_loss = ...  # compute training loss
    val_loss = ...    # compute validation loss
    best_val_loss = min(val_loss, logger.get_best_val_loss())
    optimizer = ...   # your optimizer

    logger.log_epoch(
        train_loss=train_loss,
        val_loss=val_loss,
        best_val_loss=best_val_loss,
        optimizer=optimizer
    )

# For adversarial (GAN) training:
logger_gan = TrainingLogger(save_dir="./logs_gan", adverserial_logger=True)

for epoch in range(num_epochs):
    train_loss = ...
    val_loss = ...
    train_loss_D = ...  # discriminator loss
    val_loss_D = ...
    best_val_loss = min(val_loss, logger_gan.get_best_val_loss())
    optimizer_G = ...   # generator optimizer
    optimizer_D = ...   # discriminator optimizer

    logger_gan.log_epoch(
        train_loss=train_loss,
        val_loss=val_loss,
        best_val_loss=best_val_loss,
        optimizer=optimizer_G,
        optimizer2=optimizer_D,
        train_loss_D=train_loss_D,
        val_loss_D=val_loss_D
    )
"""
import json
import torch
from pathlib import Path

from loggers_utils.execution_logger import log_execution
from utils import print_box
from utils_utils.validation import validate_type


class TrainingLogger:
    """
    TrainingLogger is a class that manages the logging of training history for machine learning models.
    It keeps track of training and validation losses, best validation loss, and optimizer states.
    The logger can also handle adversarial losses, which are commonly used in Generative Adversarial Networks (GANs).
    The training history is saved to a JSON file, and the optimizer states are saved to .pth files.
    The logger is initialized with a directory where the training history will be saved.
    It also provides methods to retrieve the optimizer states for resuming training.
    """
    @log_execution("Initializing TrainingLogger...", "TrainingLogger initialized successfully!")
    def __init__(self, save_dir, adverserial_logger=False):
        """
        Initializes the TrainingLogger. This class is responsible for logging the training history,
        including training and validation losses, best validation loss, and optimizer states.
        Once a logger is initialized, it will load the existing training history from the specified directory.
        If the history file does not exist, it will create a new one.

        :param save_dir: Directory where the training history will be saved.
        :type save_dir: str
        :param adverserial_logger: Whether to log adversarial losses (for GANs).
        :type adverserial_logger: bool
        """
        self._adverserial_logger = adverserial_logger
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        info = f"Path to save directory: {self.save_dir}"
        print_box(info)

        self._load_history()

    @log_execution(f"Logging epoch...", "Logging completed successfully!")
    def log_epoch(
            self,
            train_loss: float,
            val_loss: float,
            best_val_loss: float,
            optimizer: torch.optim.Optimizer,
            optimizer2: torch.optim.Optimizer|None = None,
            train_loss_D: float|None = None,
            val_loss_D: float|None = None,
            save_history: bool = True
        ):
        """
        Logs the training and validation losses for the current epoch, updates the history,
        and saves the optimizer states. If adversarial losses are provided, they will also be logged.

        :param train_loss: Training loss for the current epoch.
        :type train_loss: float
        :param val_loss: Validation loss for the current epoch.
        :type val_loss: float
        :param best_val_loss: Best validation loss observed so far.
        :type best_val_loss: float
        :param optimizer: Optimizer used for training.
        :type optimizer: torch.optim.Optimizer
        :param optimizer2: Second optimizer (optional, used in GANs usually the discriminator's
        optimizer).
        :type optimizer2: torch.optim.Optimizer|None
        :param train_loss_D: Training loss for the discriminator (optional, used in GANs).
        :type train_loss_D: float|None
        :param val_loss_D: Validation loss for the discriminator (optional, used in GANs).
        :type val_loss_D: float|None
        :param save_history: Whether to save the training history to a file.
        :type save_history: bool
        """
        validate_type(train_loss, float)
        validate_type(val_loss, float)
        validate_type(best_val_loss, float)
        validate_type(optimizer, torch.optim.Optimizer)
        validate_type(optimizer2, torch.optim.Optimizer, allow_none=True)
        validate_type(train_loss_D, float, allow_none=True)
        validate_type(val_loss_D, float, allow_none=True)

        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)

        if self._adverserial_logger and train_loss_D is not None:
            self.history["train_loss_D"].append(train_loss_D)
            self.history["val_loss_D"].append(val_loss_D)

        # Update epoch number
        if len(self.history["epoch"]) == 0:
            self.history["epoch"].append(1)
        else:
            self.history["epoch"].append(self.history["epoch"][-1] + 1)

        # Update best validation loss
        if self.check_best_val_loss(best_val_loss):
            self.history["best_val_loss"] = best_val_loss

        # Save optimizer state
        self.optimizer_state = optimizer.state_dict()
        self.optimizer_state2 = optimizer2.state_dict() if optimizer2 else None

        info = f"Epoch: {self.history['epoch'][-1]}"
        info += f"\nTrain Loss: {train_loss:.4f}"
        if train_loss_D is not None:
            info += f"\nTrain Loss (Discriminator): {train_loss_D:.4f}"
        info += f"\nValidation Loss: {val_loss:.4f}"
        info += f"\nBest Validation Loss: {best_val_loss:.4f}"
        print_box(info)

        if save_history:
            self.save_history()
            print_box("History saved successfully!")

    def get_optimizer_state(self):
        """
        Returns the state of the optimizer used for training.

        :return: The state of the optimizer.
        :rtype: dict
        """
        return self.optimizer_state
    
    def get_optimizer2_state(self):
        """
        Returns the state of the second optimizer (if it exists),
        typically used in GANs and is usually the discriminator's optimzier.

        :return: The state of the second optimizer or None if it does not exist.
        :rtype: dict|None
        """
        return self.optimizer_state2

    def check_best_val_loss(self, val_loss):
        """
        Checks if the current validation loss is the best observed so far.

        :param val_loss: The current validation loss to check.
        :type val_loss: float
        :return: True if the current validation loss is better than the best observed,
        False otherwise.
        :rtype: bool
        """
        return val_loss < self.history["best_val_loss"]
    
    def get_best_val_loss(self):
        """
        Returns the best validation loss observed during training.

        :return: The best validation loss.
        :rtype: float
        """
        return self.history["best_val_loss"]

    def save_history(self):
        """
        Saves the training history to a JSON file and
        the optimizer states to a .pth file. The history includes training
        and validation losses, best validation loss, and epoch numbers.
        If adversarial losses are logged, they will also be saved.
        The optimizer states are saved in a format that can be loaded
        later for resuming training.
        """
        history_copy = self.history.copy()
        with open(self.save_dir / "history.json", "w") as f:
            json.dump(history_copy, f, indent=2)

        # Optimizer state not json sterializable
        torch.save(self.optimizer_state, self.save_dir / "optimizer_state.pth")

        # Save second optimizer state if it exists
        if self.optimizer_state2 is not None:
            torch.save(self.optimizer_state2, self.save_dir / "optimizer_state2.pth")

    def _load_history(self):
        """
        Loads the training history from a JSON file and the optimizer states
        from .pth files if they exists in the `save_dir` specified in the instantiating
        of this instance. If the history file does not exist, it initializes a new
        history with empty lists for training and validation losses, best validation
        loss, and epoch numbers. If adversarial losses are logged, they will also
        be initialized. This method is called during the initialization of the TrainingLogger
        to ensure that the training history is available for logging and analysis.
        """
        # Check if the history file exists
        if not (self.save_dir / "history.json").exists():
            self.history = {
                "train_loss": [],
                "val_loss": [],
                "best_val_loss": float("inf"),
                "epoch": [],
            }

            if self._adverserial_logger:
                self.history["train_loss_D"] = []
                self.history["val_loss_D"] = []

            self.optimizer_state = None
            self.optimizer_state2 = None
            info = f"History file not found. Starting a new training session."
            print_box(info)
            return
        
        with open(self.save_dir / "history.json", "r") as f:
            history = json.load(f)

        self.history = {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "best_val_loss": history["best_val_loss"],
            "epoch": history["epoch"],
        }

        if self._adverserial_logger:
            self.history["train_loss_D"] = history.get("train_loss_D", [])
            self.history["val_loss_D"] = history.get("val_loss_D", [])
        
        # Load optimizer state
        opt_state_path = self.save_dir / "optimizer_state.pth"
        if opt_state_path.exists():
            self.optimizer_state = torch.load(opt_state_path, map_location=self.device)
        
        # Load optimizer 2 state
        opt_state_path2 = self.save_dir / "optimizer_state2.pth"
        if opt_state_path2.exists():
            self.optimizer_state2 = torch.load(opt_state_path2, map_location=self.device)

        info = f"\nPath to history file: {self.save_dir}"
        print_box(info)

        info = f"Summary of training history:"
        info += f"\nEpochs: {len(self.history['epoch'])}"
        info += f"\nBest validation loss: {self.history['best_val_loss']}"
        print_box(info)
