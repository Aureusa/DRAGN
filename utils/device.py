"""
NOTE FOR USERS:

This file provides a utility function for standardized device selection
(CPU or GPU) across the codebase.
Use `get_device()` to consistently determine the
appropriate device for your models and tensors.

Example:
    from utils.device import get_device
    device = get_device()
    model.to(device)
    tensor = tensor.to(device)

Centralizing device selection helps ensure consistency,
maintainability, and easier configuration for all scripts and modules.
"""
from typing import Dict, Any
import torch

def get_device() -> torch.device:
    """
    Get the appropriate device for PyTorch operations.
    This function checks if a CUDA-enabled GPU is available and returns
    the device accordingly. If no GPU is available, it defaults to CPU.

    :return: A PyTorch device object representing the selected device.
    :rtype: torch.device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move batch data to the device.

    :param batch: The input batch of data. Should be a dict.
    :type batch: Dict[str, Any]
    :return: The batch data moved to the appropriate device.
    :rtype: Dict[str, Any]
    """
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
