import torch
import torch.nn as nn
from typing import Any
from abc import ABC, abstractmethod


class LossFactory:
    """
    Factory class for creating loss functions based on the provided
    loss function name.
    """
    def __init__(self, loss_name: str) -> None:
        """
        Initialize the LossFactory with the provided loss function name.

        :param loss_name: the name of the loss function
        :type loss_name: str
        :raises ValueError: if the loss function name is not recognized
        """
        self.loss_functions = {
            'PSF-Constrained MSE Loss': PSFConstrainedMSELoss(),
            'PSF-Constrained Smooth L1 Loss': PSFConstrainedSmoothL1Loss(),
            'MSE Loss': MSELoss(),
            'L1 Loss': L1Loss(),
            'Smooth L1 Loss': SmoothL1Loss(),
        }
        
        if loss_name not in self.loss_functions:
            raise ValueError(f"Loss function '{loss_name}' is not recognized.")
        
        # Set the loss function based on the provided name
        self.loss_func = self.loss_functions[loss_name]

    def get_loss(self) -> nn.Module:
        """
        Get the loss function based on the provided name.

        :return: the loss function
        :rtype: nn.Module
        """
        return self.loss_func
    
    
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

    def forward(self, x, y_pred, y_true, psf):
        return self._loss_func(y_pred, y_true)
    

class L1Loss(nn.Module, Loss):
    def __init__(self):
        super(L1Loss, self).__init__()
        self._loss_func = nn.L1Loss()

    def forward(self, x, y_pred, y_true, psf):
        return self._loss_func(y_pred, y_true)
    

class SmoothL1Loss(nn.Module, Loss):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self._loss_func = nn.SmoothL1Loss()

    def forward(self, x, y_pred, y_true, psf):
        return self._loss_func(y_pred, y_true)
    