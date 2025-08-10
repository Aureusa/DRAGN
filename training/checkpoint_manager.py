import torch
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
from datetime import datetime

from networks.models import BaseModel
from utils.log_utils import log_execution
from utils.persistence import save_json_file, load_json_file
from utils import print_box


class CheckpointManager:
    """
    A general-purpose checkpoint manager for training state persistence and recovery.
    
    Handles saving and loading of models, optimizers, schedulers, random states,
    and custom metadata. Supports both single and multiple optimizers, automatic
    cleanup of old checkpoints, and best model tracking.
    """
    
    @log_execution("Initializing CheckpointManager...", "CheckpointManager initialized successfully!")
    def __init__(
        self, 
        save_dir: Union[str, Path],
        model_filename: str,
        device: torch.device,
        save_best: bool = True,
        best_model_suffix: str = "best_model",
        verbose: bool = True
    ):
        """
        Initialize the CheckpointManager.
        
        Args:
            save_dir: Directory where checkpoints will be saved
            keep_last_n: Number of recent checkpoints to keep (0 = keep all)
            save_best: Whether to automatically save best models
            checkpoint_prefix: Prefix for checkpoint filenames
            best_model_name: Filename for the best model checkpoint
            latest_model_name: Filename for the latest model checkpoint
            verbose: Whether to print verbose logging information
        """
        self.save_dir = Path(save_dir)
        self.model_filename = model_filename

        self.save_best = save_best
        self.best_model_suffix = best_model_suffix
        self.verbose = verbose

        self.device = device
        
        # Load existing metadata if available
        self._load_metadata()

    def checkpoint_exists(self):
        """
        Check if a checkpoint exists.
        """
        return self.metadata is not None

    def save_checkpoint(
        self,
        step: int,
        model: BaseModel,
        optimizers: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
        is_best: Optional[bool] = False,
        is_last: Optional[bool] = False
    ) -> None:
        if is_best:
            model_filename = f"{self.model_filename}_{self.best_model_suffix}.pth"
            checkpoint_name = f"checkpoint_{self.best_model_suffix}.pth"
        elif is_last:
            model_filename = f"{self.model_filename}_last.pth"
            checkpoint_name = f"checkpoint_last.pth"
        else:
            model_filename = f"{self.model_filename}_step_{step}.pth"
            checkpoint_name = f"checkpoint_step_{step}.pth"

        # Prepare checkpoint data
        checkpoint_data = {
            "step": step,
            "optimizer_state_dicts": self._serialize_optimizers(optimizers),
            "timestamp": datetime.now().isoformat(),
        }

        checkpoint_path = self.save_dir / checkpoint_name
        
        # Save the checkpoint
        torch.save(checkpoint_data, checkpoint_path)

        # Save the model
        model.save_checkpoint(model_filename, self.save_dir)

        if self.verbose:
            print_box(f"Checkpoint at step {step} saved!")
        
        # Update metadata
        self._update_metadata(step, model_filename, checkpoint_name)

    def load_checkpoint(
        self, 
        model: BaseModel,
        optimizers: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
    ):
        """
        Load a checkpoint and return its contents.
        
        Args:
            checkpoint_path: Specific checkpoint to load
            load_best: Load the best model checkpoint
            load_latest: Load the latest model checkpoint  
            map_location: Device to map tensors to
            
        Returns:
            Dictionary containing all checkpoint data
        """
        # Get the metadata of the last checkpoint
        last_step = self.metadata.get("last_step", 0)
        model_filename = self.metadata["model_filename"]
        checkpoint_filename = self.metadata["checkpoint_filename"]
        checkpoint_path = Path(self.save_dir / checkpoint_filename)
        
        if not (Path(self.save_dir / model_filename)).exists():
            raise FileNotFoundError(f"Model not found: {self.save_dir / model_filename}")
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model.load_checkpoint(model_filename, self.save_dir)
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)

        optimizer_state_dicts = checkpoint_data.get("optimizer_state_dicts", {})
        optimizers = self._restore_optimizers(optimizers, optimizer_state_dicts)

        if self.verbose:
            print_box(f"Checkpoint at step {last_step} loaded successfully!")

        return model, optimizers, last_step
    
    def _serialize_optimizers(
        self, 
        optimizers: Union[torch.optim.Optimizer, List[torch.optim.Optimizer], Dict[str, torch.optim.Optimizer]]
    ) -> Dict[str, Dict]:
        """Serialize optimizer(s) to a dictionary format."""
        if isinstance(optimizers, torch.optim.Optimizer):
            return {"optimizer": optimizers.state_dict()}
        elif isinstance(optimizers, dict):
            return {name: opt.state_dict() for name, opt in optimizers.items()}
        else:
            raise TypeError(f"Unsupported optimizer type: {type(optimizers)}")
    
    def _restore_optimizers(
        self,
        optimizers: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
        optimizer_states: Dict[str, Dict]
    ) -> Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]:
        """Restore optimizer states from checkpoint data."""
        if isinstance(optimizers, torch.optim.Optimizer):
            if "optimizer" in optimizer_states:
                optimizers.load_state_dict(optimizer_states["optimizer"])
        elif isinstance(optimizers, dict):
            for name, opt in optimizers.items():
                if name in optimizer_states:
                    opt.load_state_dict(optimizer_states[name])
        return optimizers
        
    def _update_metadata(self, step: int, model_filename: str, checkpoint_filename: str) -> None:
        """
        Update checkpoint metadata. The metadata is used when loading.

        :param step: The last training step.
        :type step: int
        :param model_filename: The filename of the model.
        :type model_filename: str
        :param checkpoint_filename: The filename of the checkpoint.
        :type checkpoint_filename: str
        """
        metadata_path = self.save_dir / "checkpoint_metadata.json"
        
        self.metadata = {
            "last_step": step,
            "model_filename": model_filename,
            "checkpoint_filename": checkpoint_filename,
        }

        save_json_file(self.metadata, metadata_path)

    def _load_metadata(self):
        """Load existing checkpoint metadata."""
        metadata_path = self.save_dir / "checkpoint_metadata.json"

        if metadata_path.exists():
            self.metadata = load_json_file(metadata_path)
        else:
            self.metadata = None
