from pathlib import Path

from utils import print_box
from utils.device import get_device
from utils.log_utils import log_execution
from utils.persistence import load_json_file, save_json_file
from utils.validation import validate_type


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
    def __init__(self, save_dir, verbose: bool = False):
        """
        Initializes the TrainingLogger. This class is responsible for logging the training history,
        including training and validation losses, best validation loss, and optimizer states.
        Once a logger is initialized, it will load the existing training history from the specified directory.
        If the history file does not exist, it will create a new one.

        :param save_dir: Directory where the training history will be saved.
        :type save_dir: str
        :param verbose: Whether to print verbose logging information.
        :type verbose: bool
        """
        self.verbose = verbose

        self.device = get_device()
        self.save_dir = Path(save_dir)

        self._load_epoch_history()
        self._load_step_history()

    def log_step(self, step: int, metrics: dict):
        """
        Logs the training step metrics.

        :param step: Current training step.
        :type step: int
        :param loss: Training loss for the current step.
        :type loss: float
        :param metrics: Additional metrics to log.
        :type metrics: dict
        """
        validate_type(metrics, dict)

        # Log step information
        step_info = {
            "step": step,
            **metrics
        }

        info = "Step information: {"
        info += f"step: {int(step)}; "
        for key, value in step_info.items():
            if key not in self.step_history:
                self.step_history[key] = []
            self.step_history[key].append(value)

            if key != "step":
                info += f"{key}: {value:.4f}; "
        info = info[:-2] + "}"

        if self.verbose:
            print(info)

        self._save_history(self.step_history, "step_history")

    def log_epoch(self, metrics: dict) -> None:
        """
        Logs the current epoch.

        :param metrics: Dictionary containing metrics for the current epoch.
        :type metrics: dict
        """
        validate_type(metrics, dict)

        # Update epoch number
        if "epoch" not in self.epoch_history:
            self.epoch_history["epoch"] = []
        if len(self.epoch_history["epoch"]) == 0:
            self.epoch_history["epoch"].append(1)
        else:
            self.epoch_history["epoch"].append(self.epoch_history["epoch"][-1] + 1)

        info = "Epoch information:\n"
        info += f"epoch: {self.epoch_history['epoch'][-1]}\n"
        # Log training metrics
        first_val_metric = False
        for key, value in metrics.items():
            if key not in self.epoch_history:
                self.epoch_history[key] = []
            self.epoch_history[key].append(value)
            info += f"{key}: {value:.4f}\n"

            # Check if this is the first validation metric and if it 
            # is smaller than the best val loss, assumes first val metric is the priority one
            if (
                "val_loss" in key
                and not first_val_metric
                and value < self.epoch_history.get("best_val_loss", float("inf"))
                ):
                self.epoch_history["best_val_loss"] = value
                first_val_metric = True
            
        if self.verbose:
            print_box(info)

        self._save_history(self.epoch_history, "epoch_history")

    def get_current_epoch(self):
        """
        Returns the current epoch number based on the logged history.

        :return: The current epoch number.
        :rtype: int
        """
        if len(self.epoch_history["epoch"]) == 0:
            return 0
        return self.epoch_history["epoch"][-1]
    
    def get_best_val_loss(self):
        """
        Returns the best validation loss observed during training.

        :return: The best validation loss.
        :rtype: float
        """
        return self.epoch_history["best_val_loss"]
    
    def _save_history(self, hist: dict, name: str):
        """
        Saves the history to a JSON file.
        """
        history_filepath = self.save_dir / f"{name}.json"
        save_json_file(hist, history_filepath)

    def _load_epoch_history(self):
        """
        Loads the epoch history from a JSON file and the optimizer states
        from .pth files if they exists in the `save_dir` specified in the instantiating
        of this instance. If the history file does not exist, it initializes a new
        history. This method is called during the initialization of the TrainingLogger.
        """
        epoch_history_path = self.save_dir / "epoch_history.json"
        if not epoch_history_path.exists():
            self.epoch_history = {}
            self.epoch_history["best_val_loss"] = float("inf")
            return

        self.epoch_history = load_json_file(epoch_history_path)

        if self.verbose:
            info = f"Summary of training history:"
            info += f"\nEpochs: {len(self.epoch_history['epoch'])}"
            info += f"\nBest validation loss: {self.epoch_history['best_val_loss']}"
            print_box(info)

    def _load_step_history(self):
        """
        Loads the step history from a JSON file if it exists in the `save_dir`
        specified in the instantiating of this instance. If the step history file
        does not exist, it initializes a new history. This method is called during
        the initialization of the TrainingLogger.
        """
        step_history_path = self.save_dir / "step_history.json"
        if not step_history_path.exists():
            self.step_history = {}
            return

        self.step_history = load_json_file(step_history_path)

        if self.verbose:
            info = f"Summary of step history:"
            info += f"\nSteps: {len(self.step_history['step'])}"
            print_box(info)
