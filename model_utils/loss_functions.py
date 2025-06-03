import torch
import torch.nn as nn
from typing import Any
from abc import ABC, abstractmethod
import torch.nn.functional as F


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
        'PSF-Constrained MSE Loss': PSFConstrainedMSELoss(), # Depricated
        'PSF-Constrained Smooth L1 Loss': PSFConstrainedSmoothL1Loss(), # Depricated
        'L1 + Weighted L2 Loss': L1plusWeightedL2(),
        'MSE Loss': MSELoss(),
        'L1 Loss': L1Loss(),
        'Perceptual Loss': PerceptualLoss(),
        'Smooth L1 Loss': SmoothL1Loss(),
        'PSF-MSE Loss': PSFMSELoss(), # Depricated
        'PSF-L1 Loss': PSFL1Loss(), # Depricated
        "Masked PSF-MSE Loss": MaskedPSFMSELoss(), # Depricated
        "Weighted PSF-MSE Loss": WeightedPSFMSELoss(), # Depricated
        "Weighted MSE Loss": WeightedMSELoss(),
        "Weighted MSE Loss Torch": WeightedMSELoss_Torch(),
        "MSE Loss for PSF": MSELoss_for_PSF(),
        "Weighted Squared MSE Loss": WeightedSquaredMSELoss(),
        "Weighted PSF-MSE Loss for PSF": WeightedPSFMSELoss_for_PSF(), # Depricated
    }
    return loss_functions


def check_loss_inputs(func):
    def wrapper(self, x, y_pred, y_true, psf, *args, **kwargs):
        for t, name in zip([x, y_pred, y_true, psf], ['x', 'y_pred', 'y_true', 'psf']):
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"Input '{name}' is not a torch.Tensor")
            if t.dim() != 4:
                raise ValueError(f"Input '{name}' must be 4D (B, C, H, W), got shape {t.shape}")
        return func(self, x, y_pred, y_true, psf, *args, **kwargs)
    return wrapper

    
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
    def __init__(self, layers: list[str] = None):
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
    
    def forward(self, x, y_pred, y_true, psf) -> torch.Tensor:
        return self._perceptual_loss(y_true, y_pred)
    
    def _perceptual_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
    
    def _forward_vgg(self, x: torch.Tensor) -> list[torch.Tensor]:
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
    
    def _preprocess_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
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


class PSFL1Loss(nn.Module, Loss):
    def __init__(self):
        super(PSFL1Loss, self).__init__()
        self._loss_func = nn.L1Loss()

    def __str__(self):
        return "PSF-L1 Loss"

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
    

class WeightedPSFMSELoss_for_PSF(nn.Module, Loss):
    def __init__(self):
        super(WeightedPSFMSELoss_for_PSF, self).__init__()

    def __str__(self):
        return "Weighted PSF-MSE Loss for PSF"

    def forward(self, x, y_pred, psf, y_true):
        # Create weights based on the PSF
        weights = psf / torch.max(psf)
        return nn.functional.mse_loss(y_pred, psf, weight=weights)
    

class MSELoss_for_PSF(nn.Module, Loss):
    def __init__(self):
        super(MSELoss_for_PSF, self).__init__()
        self._loss_func = nn.MSELoss()

    def __str__(self):
        return "MSE Loss for PSF"

    def forward(self, x, y_pred, psf, y_true):
        return self._loss_func(y_pred, psf)
    
    
class WeightedMSELoss(nn.Module, Loss):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self._loss_func = nn.MSELoss()

    def __str__(self):
        return "Weighted MSE Loss"

    def forward(self, x, y_pred, y_true, psf):
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

        # Sum over spatial dims and average over batch
        loss = weighted_squared_diff.sum(dim=(1, 2, 3))  # shape: (B,)
        return loss.mean()
    

class WeightedSquaredMSELoss(nn.Module, Loss):
    def __init__(self):
        super(WeightedSquaredMSELoss, self).__init__()

    def __str__(self):
        return "Weighted Squared MSE Loss"

    def forward(self, x, y_pred, y_true, psf):
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
        

class WeightedMSELoss_Torch(nn.Module, Loss):
    def __init__(self):
        super(WeightedMSELoss_Torch, self).__init__()
        self._loss_func = nn.MSELoss()

    def __str__(self):
        return "Weighted MSE Loss Torch"

    def forward(self, x, y_pred, y_true, psf):
        # Create weights based on the PSF
        max_psf = torch.amax(psf, dim=(1, 2, 3), keepdim=True)
        weights = psf / (max_psf + 1e-8)

        # Find batch indices where max == 0
        zero_mask = (max_psf == 0).squeeze(-1).squeeze(-1).squeeze(-1)  # shape (B,)

        # Set all weights for those batches to 1
        weights[zero_mask] = 1.0
        weights = torch.clamp(weights, min=1e-6)

        weighted_y_pred = y_pred * (weights ** 0.5)
        weighted_y_true = y_true * (weights ** 0.5)

        return self._loss_func(weighted_y_pred, weighted_y_true)
    

class L1Loss(nn.Module, Loss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def __str__(self):
        return "L1 Loss"

    def forward(self, x, y_pred, y_true, psf):
        return torch.mean(torch.abs(y_pred - y_true))
    

class SmoothL1Loss(nn.Module, Loss):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self._loss_func = nn.SmoothL1Loss()

    def __str__(self):
        return "Smooth L1 Loss"

    def forward(self, x, y_pred, y_true, psf):
        return self._loss_func(y_pred, y_true)
    

class L1plusWeightedL2(nn.Module, Loss):
    def __init__(self, alpha: float = 1.0, beta: float = 0.2):
        super(L1plusWeightedL2, self).__init__()
        self.alpha = alpha
        self.beta = beta

        # Initialize the loss functions
        self.l1_loss = nn.L1Loss()
        self.weighted_mse_loss = WeightedSquaredMSELoss()

    def __str__(self):
        return "L1 + Weighted L2 Loss"

    def forward(self, x, y_pred, y_true, psf):
        l1_loss = self.l1_loss(y_pred, y_true)
        weighted_mse_loss = self.weighted_mse_loss(x, y_pred, y_true, psf)
        
        loss = self.alpha * l1_loss + self.beta * weighted_mse_loss
        return loss
    