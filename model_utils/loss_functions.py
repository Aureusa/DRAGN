import torch
import torch.nn as nn
from typing import Any
from abc import ABC, abstractmethod

# TODO:
# 1. Double check the loss functions (escpecially the PSF-Constrained ones)
# 2. Add more loss functions
def get_loss_function(loss_name: str) -> None:
    loss_functions = _get_avaliable_loss_funcstions()
    
    if loss_name not in loss_functions:
        raise ValueError(f"Loss function '{loss_name}' is not recognized.")
    
    # Set the loss function based on the provided name
    loss_func = loss_functions[loss_name]

    return loss_func


# LAST UPDATE: 2025-13-04
# Make sure to update this once new loss functions are added
def _get_avaliable_loss_funcstions() -> dict:
    loss_functions = {
        'PSF-Constrained MSE Loss': PSFConstrainedMSELoss(),
        'PSF-Constrained Smooth L1 Loss': PSFConstrainedSmoothL1Loss(),
        'MSE Loss': MSELoss(),
        'L1 Loss': L1Loss(),
        'Smooth L1 Loss': SmoothL1Loss(),
        'PSF-MSE Loss': PSFMSELoss(),
        "Masked PSF-MSE Loss": MaskedPSFMSELoss(),
        "Weighted PSF-MSE Loss": WeightedPSFMSELoss(),
        "Weighted MSE Loss": WeightedMSELoss(),
    }
    return loss_functions

    
class Loss(ABC):
    """
    Abstract Loss class that enforces the characteristic
    forward pass for all loss functions.
    """
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        psf: torch.Tensor
    ) -> Any:
        """
        Forward pass for the loss function with its specific parameters
        to be implemented by the subclasses.

        :param x: the input image
        :type x: torch.Tensor
        :param y_pred: the predicted image
        :type y_pred: torch.Tensor
        :param y_true: the true image
        :type y_true: torch.Tensor
        :param psf: the point spread function
        :type psf: torch.Tensor
        """
        pass
    
    
class PSFConstrainedMSELoss(nn.Module, Loss):
    def __init__(self):
        super(PSFConstrainedMSELoss, self).__init__()
        self._loss_func = nn.MSELoss()

    def forward(self, x, y_pred, y_true, psf):
        # Calculate the difference between the predicted and original images
        diff = x - y_pred
        
        # Calculate the loss for the predicted and true images
        loss1 = self._loss_func(y_pred, y_true)

        # Calculate the loss for the difference between
        # the predicted-clean difference and the PSF
        loss2 = self._loss_func(diff, psf)
        return loss1 + loss2
    

class PSFConstrainedSmoothL1Loss(nn.Module, Loss):
    def __init__(self):
        super(PSFConstrainedSmoothL1Loss, self).__init__()
        self._loss_func = nn.SmoothL1Loss()

    def __str__(self):
        return "PSF-Constrained Smooth L1 Loss"

    def forward(self, x, y_pred, y_true, psf):
        # Calculate the difference between the predicted and true images
        diff = x - y_pred
        
        # Calculate the loss for the predicted and true images
        loss1 = self._loss_func(y_pred, y_true)

        # Calculate the loss for the difference between
        # the predicted-clean difference and the PSF
        loss2 = self._loss_func(diff, psf)
        return loss1 + loss2


class MSELoss(nn.Module, Loss):
    def __init__(self):
        super(MSELoss, self).__init__()
        self._loss_func = nn.MSELoss()

    def __str__(self):
        return "MSE Loss"

    def forward(self, x, y_pred, y_true, psf):
        return self._loss_func(y_pred, y_true)
    

class PSFMSELoss(nn.Module, Loss):
    def __init__(self):
        super(PSFMSELoss, self).__init__()
        self._loss_func = nn.MSELoss()

    def __str__(self):
        return "PSF-MSE Loss"

    def forward(self, x, y_pred, y_true, psf):
        diff = x - y_pred
        return self._loss_func(diff, psf)
    

class MaskedPSFMSELoss(nn.Module, Loss):
    def __init__(self):
        super(MaskedPSFMSELoss, self).__init__()
        self._loss_func = nn.MSELoss()

    def __str__(self):
        return "Masked PSF-MSE Loss"

    def forward(self, x, y_pred, y_true, psf):
        # Create a mask for the pixels where the PSF is non-zero
        mask = psf > 0

        # Apply the mask to the difference between the predicted and original images
        diff = x - y_pred
        diff = diff[mask]
        return self._loss_func(diff, psf)
    
    
class WeightedPSFMSELoss(nn.Module, Loss):
    def __init__(self):
        super(WeightedPSFMSELoss, self).__init__()

    def __str__(self):
        return "Weighted PSF-MSE Loss"

    def forward(self, x, y_pred, y_true, psf):
        # Create weights based on the PSF
        weights = psf / torch.max(psf)

        diff = x - y_pred
        return nn.functional.mse_loss(diff, psf, weight=weights)
    

class WeightedMSELoss(nn.Module, Loss):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def __str__(self):
        return "Weighted MSE Loss"

    def forward(self, x, y_pred, y_true, psf):
        # Create weights based on the PSF
        weights = psf / torch.max(psf)
        return nn.functional.mse_loss(y_pred, y_true, weight=weights)
    

class L1Loss(nn.Module, Loss):
    def __init__(self):
        super(L1Loss, self).__init__()
        self._loss_func = nn.L1Loss()

    def __str__(self):
        return "L1 Loss"

    def forward(self, x, y_pred, y_true, psf):
        return self._loss_func(y_pred, y_true)
    

class SmoothL1Loss(nn.Module, Loss):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self._loss_func = nn.SmoothL1Loss()

    def __str__(self):
        return "Smooth L1 Loss"

    def forward(self, x, y_pred, y_true, psf):
        return self._loss_func(y_pred, y_true)
    