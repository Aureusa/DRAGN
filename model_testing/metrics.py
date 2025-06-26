from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from utils import print_box
from utils_utils.validation import check_4tensor_inputs


def get_metrics(metrics: list[str], verbose: bool = True) -> list:
    """
    Retrieve the specified metrics from the available metrics dictionary.

    :param metrics: List of metric names to retrieve.
    :type metrics: list[str]
    :param verbose: If True, print the retrieved metrics.
    :type verbose: bool
    :return: List of metric instances corresponding to the provided names.
    :rtype: list[Metric]
    :raises ValueError: If no valid metrics are provided.
    """
    metrics_dict = _get_avaliable_metrics()

    metrics_list = []

    info = "Retrieved metrics:\n"
    for m in metrics:
        if m not in metrics_dict:
            info += f" - Warning retrival of `{m}` unsuccessful, no such metric!\n"
            continue
        
        metric = metrics_dict[m]
        metrics_list.append(metric)
        info += f" - {m}\n"

    if verbose:
        print_box(info)
    
    if not metrics_list:
        raise ValueError("No valid metrics provided. Please check the metric names.")
    
    return metrics_list

def _get_avaliable_metrics() -> dict:
    metrics_dict = {
            "PSNR": PSNR(),
            "PSNR PSF": PSNR_PSF(),
            "SSIM": SSIM(),
            "SSIM PSF": SSIM_PSF(),
            "Centroid Error": CentroidError(),
            "FRF": FluxResidualFraction(),
            "FRF PSF": FluxResidualFraction_PSF(),
            "Reconstruction Loss": ReconstructionLoss(),
        }
    return metrics_dict


class Metric(ABC):
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
        Forward pass for the metric function with its specific parameters
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

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata about the metric.
        This method should be implemented by subclasses to provide
        information about the metric such as its name, description, etc.
        
        :return: Metadata dictionary containing information about the metric.
        :rtype: dict[str, Any]
        """
        pass


class PSNR(nn.Module, Metric):
    """
    Peak Signal-to-Noise Ratio (PSNR) metric.
    This class computes the PSNR metric between two images.
    PSNR is a common metric for measuring the quality of reconstructed images.
    It is defined as the ratio between the maximum possible power of a signal
    and the power of corrupting noise that affects the fidelity of its representation.
    The reduction is done over the batch dimension, and the output is a tensor
    containing the PSNR values of the batch.
    """
    def __init__(self):
        super(PSNR, self).__init__()
        self._psnr = PeakSignalNoiseRatio()

    def __str__(self):
        return "PSNR"
    
    @check_4tensor_inputs
    def forward(
            self,
            x: torch.Tensor,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            psf: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass for the PSNR metric.

        :param x: the input image (not used in PSNR calculation)
        :type x: torch.Tensor
        :param y_pred: the predicted image
        :type y_pred: torch.Tensor
        :param y_true: the true image
        :type y_true: torch.Tensor
        :param psf: the point spread function (not used in PSNR calculation)
        :type psf: torch.Tensor
        :return: PSNR value as a tensor.
        :rtype: torch.Tensor
        """
        psnr_value = self._psnr(y_pred, y_true)
        return psnr_value
        
    def to(self, device):
        """
        Move the PSNR metric to the specified device.
        """
        self._psnr.to(device)
        return self
    
    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata about the PSNR metric.
        
        :return: Metadata dictionary containing information about the PSNR metric.
        :rtype: dict[str, Any]
        """
        return {
            "name": "PSNR",
            "description": "Peak Signal-to-Noise Ratio (PSNR) metric for image quality assessment.",
            "type": "image_quality",
            "units": "dB",
        }
    

class PSNR_PSF(PSNR):
    """
    Peak Signal-to-Noise Ratio (PSNR) metric for PSF.
    """
    def __str__(self):
        return "PSNR PSF"

    @check_4tensor_inputs
    def forward(
            self,
            x: torch.Tensor,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            psf: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass for the PSNR PSF metric.

        :param x: the input image (not used in PSNR calculation)
        :type x: torch.Tensor
        :param y_pred: the predicted image
        :type y_pred: torch.Tensor
        :param y_true: the true image
        :type y_true: torch.Tensor
        :param psf: the point spread function
        :type psf: torch.Tensor
        :return: PSNR value as a tensor.
        :rtype: torch.Tensor
        """
        diff = x - y_pred
        psnr_value = self._psnr(diff, psf)
        return psnr_value
    
    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata about the PSNR PSF metric.
        
        :return: Metadata dictionary containing information about the PSNR PSF metric.
        :rtype: dict[str, Any]
        """
        return {
            "name": "PSNR PSF",
            "description": "Peak Signal-to-Noise Ratio (PSNR) metric for Point Spread Function (PSF) assessment.",
            "type": "psf_quality",
            "units": "dB",
        }
    

class SSIM(nn.Module, Metric):
    """
    Structural Similarity Index Measure (SSIM) metric.
    This class computes the SSIM metric between two images.
    SSIM is a perception-based model that considers image degradation
    as perceived change in structural information,
    while also incorporating important perceptual phenomena.
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self._ssim = StructuralSimilarityIndexMeasure(reduction='none')

    def __str__(self):
        return "SSIM"

    @check_4tensor_inputs
    def forward(
            self,
            x: torch.Tensor,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            psf: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass for the SSIM metric.

        :param x: the input image (not used in SSIM calculation)
        :type x: torch.Tensor
        :param y_pred: the predicted image
        :type y_pred: torch.Tensor
        :param y_true: the true image
        :type y_true: torch.Tensor
        :param psf: the point spread function (not used in SSIM calculation)
        :type psf: torch.Tensor
        :return: SSIM value as a tensor.
        :rtype: torch.Tensor
        """
        ssim_value = self._ssim(y_pred, y_true)
        return ssim_value
    
    def to(self, device):
        """
        Move the SSIM metric to the specified device.
        """
        self._ssim.to(device)
        return self
    
    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata about the SSIM metric.
        
        :return: Metadata dictionary containing information about the SSIM metric.
        :rtype: dict[str, Any]
        """
        return {
            "name": "SSIM",
            "description": "Structural Similarity Index Measure (SSIM) for image quality assessment.",
            "type": "image_quality",
            "best_value": 1.0,
            "worst_value": 0.0,
            "higher_is_better": True,
        }
    

class SSIM_PSF(SSIM):
    """
    Structural Similarity Index Measure (SSIM) metric for PSF.
    """
    def __str__(self):
        return "SSIM PSF"

    @check_4tensor_inputs
    def forward(
            self,
            x: torch.Tensor,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            psf: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass for the SSIM PSF metric.
        
        :param x: the input image (not used in SSIM calculation)
        :type x: torch.Tensor
        :param y_pred: the predicted image
        :type y_pred: torch.Tensor
        :param y_true: the true image (not used in SSIM calculation)
        :type y_true: torch.Tensor
        :param psf: the point spread function
        :type psf: torch.Tensor
        :return: SSIM value as a tensor.
        :rtype: torch.Tensor
        """
        diff = x - y_pred
        ssim_value = self._ssim(diff, psf)
        return ssim_value
    
    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata about the SSIM PSF metric.
        
        :return: Metadata dictionary containing information about the SSIM PSF metric.
        :rtype: dict[str, Any]
        """
        return {
            "name": "SSIM PSF",
            "description": "Structural Similarity Index Measure (SSIM) for Point Spread Function (PSF) assessment.",
            "type": "psf_quality",
            "best_value": 1.0,
            "worst_value": 0.0,
            "higher_is_better": True,
        }
    

class CentroidError(nn.Module, Metric):
    """
    Centroid Error metric.
    This class computes the centroid error between two images.
    The centroid error is a measure of the difference in the centroids
    of two images.
    """
    def __init__(self):
        super(CentroidError, self).__init__()

    def __str__(self):
        return "Centroid Error"

    @check_4tensor_inputs
    def forward(
            self,
            x: torch.Tensor,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            psf: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass for the Centroid Error metric.

        :param x: the input image
        :type x: torch.Tensor
        :param y_pred: the predicted image
        :type y_pred: torch.Tensor
        :param y_true: the true image (not used in Centroid Error calculation)
        :type y_true: torch.Tensor
        :param psf: the point spread function
        :type psf: torch.Tensor
        :return: Centroid Error value as a tensor.
        :rtype: torch.Tensor
        """
        # Compute the estimated PSF from the 
        # input image and the predicted image
        psf_est = x - y_pred

        # Compute centroids of the PSF and the estimated PSF
        c_true = self._compute_centroid(psf)
        c_est  = self._compute_centroid(psf_est)

        # L2 distance per sample
        err = torch.norm(c_true - c_est, dim=1) # shape: (B,)

        # Normalize the error by the image dimensions
        _, _, H, W = x.shape
        norm_factor = torch.sqrt(torch.tensor(H**2 + W**2, dtype=x.dtype, device=x.device))
        err = err / norm_factor # shape: (B,)

        return err
    
    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata about the Centroid Error metric.
        
        :return: Metadata dictionary containing information about the Centroid Error metric.
        :rtype: dict[str, Any]
        """
        return {
            "name": "Centroid Error",
            "description": "Centroid Error metric for measuring the difference in centroids of two images.",
            "type": "image_quality",
            "units": "pixels",
            "best_value": 0.0,
            "worst_value": float('inf'),
            "higher_is_better": False,
        }
    
    def _compute_centroid(self, img: torch.Tensor) -> torch.Tensor:
        """
        Computes the centroid of a single-channel image.

        The centroid is computed as the weighted average of the pixel coordinates,
        where the weights are the pixel values in the image.
        The input image is expected to be of shape (B, C, H, W), where:
        - B is the batch size
        - C is the number of channels (should be 1 for this metric)
        - H is the height of the image
        - W is the width of the image
        The output is a tensor of shape (B, 2), where the first column contains
        the x-coordinates of the centroids and the second column contains
        the y-coordinates of the centroids.

        :param img: Input image tensor of shape (B, C, H, W).
        :type img: torch.Tensor
        :return: Tensor of shape (B, 2) containing the centroids.
        :rtype: torch.Tensor
        """
        # Extract the batch size, number of channels, height, and width
        B, C, H, W = img.shape

        # Assert that the number of channels is 1
        assert C == 1, "CentroidError expects single-channel images (C=1)"

        # Get the device of the input image
        device = img.device

        # Ommit the channel dimension
        img = img[:, 0, :, :]  # (B, H, W)

        # Create coordinate grids for the image
        # y_coords and x_coords are created to
        # match the spatial dimensions of the image
        y_coords = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)
        x_coords = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)

        # Calculate the centroids
        # The centroids are computed as the weighted average of the coordinates
        # where the weights are the pixel values in the image
        # The sum is computed over the spatial dimensions (H, W)
        # to get the total weight for each image in the batch
        total = img.sum(dim=(1, 2), keepdim=True) + 1e-8
        x_centroid = (x_coords * img).sum(dim=(1, 2)) / total.view(B)
        y_centroid = (y_coords * img).sum(dim=(1, 2)) / total.view(B)

        return torch.stack([x_centroid, y_centroid], dim=1)  # (B, 2)
    

class FluxResidualFraction(nn.Module, Metric):
    """
    Computes Flux Residual Fraction (FRF) of the galaxy images:
        FRF = [sum(estimated PSF) / sum(true PSF)] - 1
    The FRF is a measure of how well the model recovers the flux of the PSF.
    The value is is expected to be close to 0.0 meaning perfect recovery of the flux.
    A positive value indicates that the model is recovering more flux than it should,
    while a negative value indicates that the model is recovering less flux than it should.
    """
    def __init__(self):
        super(FluxResidualFraction, self).__init__()

    def __str__(self):
        return "FRF"

    @check_4tensor_inputs
    def forward(
            self,
            x: torch.Tensor,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            psf: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass for the Flux Residual Fraction metric.

        :param x: the input image (not used in FRF calculation)
        :type x: torch.Tensor
        :param y_pred: the predicted image
        :type y_pred: torch.Tensor
        :param y_true: the true image
        :type y_true: torch.Tensor
        :param psf: the point spread function
        :type psf: torch.Tensor
        :return: Flux Residual Fraction value as a tensor.
        :rtype: torch.Tensor
        """
        return self._flux_residual_fraction(y_pred, y_true)
    
    def _flux_residual_fraction(
            self,
            pred: torch.Tensor,
            true: torch.Tensor
        ) -> torch.Tensor:
        """
        Computes the FRF for the estimate and the ground truth.

        :param pred: the predicted image
        :type pred: torch.Tensor
        :param true: the true image
        :type true: torch.Tensor
        :return: Flux Residual Fraction value as a tensor.
        :rtype: torch.Tensor
        """
        # Compute the fluxes
        recovered_flux = pred.sum(dim=(1, 2, 3)) # shape: (B,)
        true_flux = true.sum(dim=(1, 2, 3)) # shape: (B,)

        frf = recovered_flux / (true_flux + 1e-8) - 1   # avoid div/0; shape: (B,)
        return frf
    
    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata about the Flux Residual Fraction metric.
        
        :return: Metadata dictionary containing information about the FRF metric.
        :rtype: dict[str, Any]
        """
        return {
            "name": "Flux Residual Fraction",
            "description": "Flux Residual Fraction (FRF) for measuring the flux recovery of the PSF.",
            "type": "flux_quality",
            "best_value": 0.0,
            "worst_value": float('inf'),
            "higher_is_better": False,
        }

class FluxResidualFraction_PSF(FluxResidualFraction):
    """
    Computes Flux Residual Fraction (FRF) of the PSF:
    """
    def __init__(self):
        super(FluxResidualFraction_PSF, self).__init__()

    def __str__(self):
        return "FRF PSF"

    @check_4tensor_inputs
    def forward(
            self,
            x: torch.Tensor,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            psf: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass for the Flux Residual Fraction PSF metric.

        :param x: the input image (not used in FRF PSF calculation)
        :type x: torch.Tensor
        :param y_pred: the predicted image
        :type y_pred: torch.Tensor
        :param y_true: the true image (not used in FRF PSF calculation)
        :type y_true: torch.Tensor
        :param psf: the point spread function
        :type psf: torch.Tensor
        :return: Flux Residual Fraction PSF value as a tensor.
        :rtype: torch.Tensor
        """
        psf_est = x - y_pred
        return self._flux_residual_fraction(psf_est, psf)
    

class ReconstructionLoss(nn.Module, Metric):
    """
    Base class for reconstruction loss.
    This class is used to compute the reconstruction loss between the input image and the predicted image.
    """
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self._loss_fn = nn.L1Loss(reduction="none")

    def __str__(self):
        return "Reconstruction Loss"

    @check_4tensor_inputs
    def forward(
            self,
            x: torch.Tensor,
            y_pred: torch.Tensor,
            y_true: torch.Tensor,
            psf: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass for the Reconstruction Loss metric.

        :param x: the input image (not used in Reconstruction Loss calculation)
        :type x: torch.Tensor
        :param y_pred: the predicted image
        :type y_pred: torch.Tensor
        :param y_true: the true image
        :type y_true: torch.Tensor
        :param psf: the point spread function (not used in Reconstruction Loss calculation)
        :type psf: torch.Tensor
        :return: Reconstruction Loss value as a tensor.
        :rtype: torch.Tensor
        """
        loss = self._loss_fn(y_pred, y_true) # shape: (B, C, H, W)
        loss_per_sample = loss.view(loss.shape[0], -1).mean(dim=1) # shape: (B,)
        return loss_per_sample
    
    def to(self, device):
        """
        Move the reconstruction loss to the specified device.
        """
        self._loss_fn.to(device)
        return self
    
    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata about the Reconstruction Loss metric.
        
        :return: Metadata dictionary containing information about the Reconstruction Loss metric.
        :rtype: dict[str, Any]
        """
        return {
            "name": "Reconstruction Loss",
            "description": "Reconstruction Loss for measuring the difference between predicted and true images.",
            "type": "reconstruction_quality",
            "best_value": 0.0,
            "worst_value": float('inf'),
            "higher_is_better": False,
        }
