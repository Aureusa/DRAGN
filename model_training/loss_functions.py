import torch
import torch.nn as nn
from typing import Any
from abc import ABC, abstractmethod
import torch.nn.functional as F

from utils_utils.validation import check_4tensor_inputs


def get_loss_function(loss_name: str) -> None:
    loss_functions = _get_avaliable_loss_funcstions()
    
    if loss_name not in loss_functions:
        raise ValueError(f"Loss function '{loss_name}' is not recognized.")
    
    # Set the loss function based on the provided name
    loss_func = loss_functions[loss_name]

    return loss_func


# LAST UPDATE: 2025-07-05
# Make sure to update this once new loss functions are added
def _get_avaliable_loss_funcstions() -> dict:
    loss_functions = {
        'L1 + Weighted L2 Loss': L1plusWeightedL2(),
        'MSE Loss': MSELoss(),
        'L1 Loss': L1Loss(),
        'Perceptual Loss': PerceptualLoss(),
        'Smooth L1 Loss': SmoothL1Loss(),
        "Weighted Squared MSE Loss": WeightedSquaredMSELoss(),
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


class PerceptualLoss(nn.Module, Loss):
    """
    Perceptual Loss based on VGG16 features.
    This loss function computes the perceptual difference
    between two images by comparing their feature maps
    extracted from a pre-trained VGG16 model."""
    def __init__(self, layers: list[str] = None):
        """
        Initialize the PerceptualLoss class.
        
        :param layers: List of layer names to use for perceptual loss.
                      If None, defaults to ['3', '8', '15', '22'].
        :type layers: list[str], optional
        """
        super(PerceptualLoss, self).__init__()
        import torchvision.models as models
        import torchvision.transforms as transforms

        # Define the loss function
        self.l1_loss = nn.L1Loss()

        # Select layers for perceptual loss
        if layers is None:
            layers = ['3', '8', '15', '22']  # relu1_2, relu2_2, relu3_3, relu4_3
        self.selected_layers = layers

        # Load pre-trained VGG16
        self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        for param in self.vgg.parameters():
            param.requires_grad = False

    def __str__(self):
        return "Perceptual Loss"
    
    @check_4tensor_inputs
    def forward(
            self,
            x: torch.Tensor,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            psf: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass for the perceptual loss function.

        :param x: Input image tensor.
        :type x: torch.Tensor
        :param y_pred: Predicted image tensor.
        :type y_pred: torch.Tensor
        :param y_true: True image tensor.
        :type y_true: torch.Tensor
        :param psf: Point spread function tensor (not used in this loss).
        :type psf: torch.Tensor
        :return: Computed perceptual loss.
        :rtype: torch.Tensor
        """
        return self._perceptual_loss(y_true, y_pred)
    
    def _perceptual_loss(
            self,
            x: torch.Tensor,
            y: torch.Tensor
        ) -> torch.Tensor:
        """
        Compute the perceptual loss between two images.
        
        :param x: First image tensor.
        :type x: torch.Tensor
        :param y: Second image tensor.
        :type y: torch.Tensor
        :return: Perceptual loss.
        :rtype: torch.Tensor
        """
        # Preprocess the images
        x = self._preprocess_batch(x)
        y = self._preprocess_batch(y)

        # Compute perceptual difference
        features1 = self._forward_vgg(x)
        features2 = self._forward_vgg(y)

        # Compute L1 distance across feature maps
        loss = 0
        for f1, f2 in zip(features1, features2):
            loss += self.l1_loss(f1, f2)
        return loss
    
    def _forward_vgg(
            self,
            x: torch.Tensor
        ) -> list[torch.Tensor]:
        """
        Forward pass of the VGG16 model.
        
        :param x: Input tensor.
        :type x: torch.Tensor
        :return: List of feature maps from the selected layers.
        :rtype: list[torch.Tensor]
        """
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.selected_layers:
                features.append(x)
        return features
    
    def _preprocess_batch(
            self,
            batch_tensor: torch.Tensor
        ) -> torch.Tensor:
        """
        Preprocess the input batch tensor for VGG16.
        
        :param batch_tensor: Input batch tensor.
        :type batch_tensor: torch.Tensor
        :return: Preprocessed batch tensor.
        :rtype: torch.Tensor
        """
        if batch_tensor.shape[1] == 1:  # Grayscale
            batch_tensor = batch_tensor.repeat(1, 3, 1, 1)  # Convert to RGB

        # Resize
        batch_tensor = F.interpolate(batch_tensor, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=batch_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=batch_tensor.device).view(1, 3, 1, 1)
        batch_tensor = (batch_tensor - mean) / std

        return batch_tensor


class MSELoss(nn.Module, Loss):
    """
    Mean Squared Error (MSE) Loss function.
    This loss function computes the mean squared error
    between the predicted and true images.
    """
    def __init__(self):
        """
        Initialize the MSELoss class.
        """
        super(MSELoss, self).__init__()
        self._loss_func = nn.MSELoss()

    def __str__(self):
        return "MSE Loss"

    @check_4tensor_inputs
    def forward(
            self,
            x: torch.Tensor,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            psf: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass for the MSE loss function.

        :param x: Input image tensor.
        :type x: torch.Tensor
        :param y_pred: Predicted image tensor.
        :type y_pred: torch.Tensor
        :param y_true: True image tensor.
        :type y_true: torch.Tensor
        :param psf: Point spread function tensor (not used in this loss).
        :type psf: torch.Tensor
        :return: Computed MSE loss.
        :rtype: torch.Tensor
        """
        return self._loss_func(y_pred, y_true)


class WeightedSquaredMSELoss(nn.Module, Loss):
    """
    Weighted Squared Mean Squared Error (MSE) Loss function.
    This loss function computes the mean squared error
    between the predicted and true images, weighted by the squared
    point spread function (PSF).
    
    Weigted MSE Loss is defined as:
    .. math::
        L_{wMSE}(y_{pred}, y_{true}, PSF) = \frac{1}{N} \sum_{i=1}^{N} w_i (y_{pred,i} - y_{true,i})^2
        where :math:`w_i = \frac{PSF_i^2}{max(PSF^2) + \epsilon}` and :math:`\epsilon` is a small constant to avoid division by zero.
    """
    def __init__(self):
        """
        Initialize the WeightedSquaredMSELoss class.
        """
        super(WeightedSquaredMSELoss, self).__init__()

    def __str__(self):
        return "Weighted Squared MSE Loss"

    @check_4tensor_inputs
    def forward(
            self,
            x: torch.Tensor,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            psf: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass for the weighted squared MSE loss function.
        :param x: Input image tensor (not used in this loss).
        :type x: torch.Tensor
        :param y_pred: Predicted image tensor.
        :type y_pred: torch.Tensor
        :param y_true: True image tensor.
        :type y_true: torch.Tensor
        :param psf: Point spread function tensor.
        :type psf: torch.Tensor
        :return: Computed weighted squared MSE loss.
        :rtype: torch.Tensor
        """
        # Square the PSF
        psf = psf ** 2
        
        # Create weights based on the PSF
        max_psf = torch.amax(psf, dim=(1, 2, 3), keepdim=True)
        weights = psf / (max_psf + 1e-8)

        # Find batch indices where max == 0
        zero_mask = (max_psf == 0).squeeze(-1).squeeze(-1).squeeze(-1)  # shape (B,)
        # Set all weights for those batches to 1
        weights[zero_mask] = 1.0

        weights = torch.clamp(weights, min=1e-6)

        # Compute the squared difference and apply weights
        squared_diff = (y_pred - y_true) ** 2
        weighted_squared_diff = squared_diff * weights

        loss = weighted_squared_diff.mean()
        return loss
        

class L1Loss(nn.Module, Loss):
    """
    L1 Loss function.
    """
    def __init__(self):
        """
        Initialize the L1Loss class.
        """
        super(L1Loss, self).__init__()

    def __str__(self):
        return "L1 Loss"

    @check_4tensor_inputs
    def forward(
            self,
            x: torch.Tensor,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            psf: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass for the L1 loss function.

        :param x: Input image tensor (not used in this loss).
        :type x: torch.Tensor
        :param y_pred: Predicted image tensor.
        :type y_pred: torch.Tensor
        :param y_true: True image tensor.
        :type y_true: torch.Tensor
        :param psf: Point spread function tensor (not used in this loss).
        :type psf: torch.Tensor
        :return: Computed L1 loss.
        :rtype: torch.Tensor
        """
        return torch.mean(torch.abs(y_pred - y_true))
    

class SmoothL1Loss(nn.Module, Loss):
    """
    Smooth L1 Loss function.
    This loss function is less sensitive to outliers than L1 Loss
    and is defined as:
    .. math::
        L_{SmoothL1}(y_{pred}, y_{true}) = \begin{cases}
            0.5 * (y_{pred} - y_{true})^2 / beta & \text{if } |y_{pred} - y_{true}| < beta \\
            |y_{pred} - y_{true}| - 0.5 * beta & \text{otherwise}
        \end{cases}
    """
    def __init__(self):
        """
        Initialize the SmoothL1Loss class.
        """
        super(SmoothL1Loss, self).__init__()
        self._loss_func = nn.SmoothL1Loss()

    def __str__(self):
        return "Smooth L1 Loss"

    @check_4tensor_inputs
    def forward(
            self,
            x: torch.Tensor,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            psf: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass for the Smooth L1 loss function.

        :param x: Input image tensor (not used in this loss).
        :type x: torch.Tensor
        :param y_pred: Predicted image tensor.
        :type y_pred: torch.Tensor
        :param y_true: True image tensor.
        :type y_true: torch.Tensor
        :param psf: Point spread function tensor (not used in this loss).
        :type psf: torch.Tensor
        :return: Computed Smooth L1 loss.
        :rtype: torch.Tensor
        """
        return self._loss_func(y_pred, y_true)
    

class L1plusWeightedL2(nn.Module, Loss):
    """
    L1 Loss combined with Weighted Squared MSE Loss.
    It is defined as:
    .. math::
        L_{L1 + wL2}(y_{pred}, y_{true}, PSF) = \alpha * L_{L1}(y_{pred}, y_{true}) + \beta * L_{wMSE}(y_{pred}, y_{true}, PSF)
    
    where :math:`\alpha` and :math:`\beta` are hyperparameters that control the contribution of each loss.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.2):
        super(L1plusWeightedL2, self).__init__()
        self.alpha = alpha
        self.beta = beta

        # Initialize the loss functions
        self.l1_loss = nn.L1Loss()
        self.weighted_mse_loss = WeightedSquaredMSELoss()

    def __str__(self):
        return "L1 + Weighted L2 Loss"

    @check_4tensor_inputs
    def forward(
            self,
            x: torch.Tensor,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            psf: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass for the combined L1 and Weighted Squared MSE loss function.
        
        :param x: Input image tensor (not used in this loss).
        :type x: torch.Tensor
        :param y_pred: Predicted image tensor.
        :type y_pred: torch.Tensor
        :param y_true: True image tensor.
        :type y_true: torch.Tensor
        :param psf: Point spread function tensor.
        :type psf: torch.Tensor
        :return: Computed combined loss.
        :rtype: torch.Tensor
        """
        l1_loss = self.l1_loss(y_pred, y_true)
        weighted_mse_loss = self.weighted_mse_loss(x, y_pred, y_true, psf)
        
        loss = self.alpha * l1_loss + self.beta * weighted_mse_loss
        return loss
    