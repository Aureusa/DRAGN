import torch
import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from model_utils.loss_functions import get_loss_function, check_loss_inputs, Loss
from utils import print_box


def get_metrics(metrics: list[str]) -> None:
    metrics_dict = {
            "PSNR": PSNR(),
            "PSNR PSF": PSNR_PSF(),
            "SSIM": SSIM(),
            "SSIM PSF": SSIM_PSF(),
            "Centroid Error": CentroidError(),
            "FRF": FluxRecoveryFraction(),
            "FRF PSF": FluxRecoveryFraction_PSF(),
            "RFC": RelativeFluxChange(),
            "RFC PSF": RelativeFluxChange_PSF(),
            "Reconstruction Loss": ReconstructionLoss(),
        }
    
    metrics_list = []

    info = "Retrieved metrics:\n"
    for m in metrics:
        try:
            metric = get_loss_function(m)
            metrics_list.append(metric)
            info += f" - {m}\n"
        except Exception as e:
            if m not in metrics_dict:
                continue
            else:
                metric = metrics_dict[m]
                metrics_list.append(metric)
                info += f" - {m}\n"

    print_box(info)

    return metrics_list

class PSNR(nn.Module, Loss):
    """
    Peak Signal-to-Noise Ratio (PSNR) metric.
    This class computes the PSNR metric between two images.
    PSNR is a common metric for measuring the quality of reconstructed images.
    It is defined as the ratio between the maximum possible power of a signal
    and the power of corrupting noise that affects the fidelity of its representation.
    """
    def __init__(self):
        super(PSNR, self).__init__()
        self._psnr = PeakSignalNoiseRatio()

    def __str__(self):
        return "PSNR"
    
    @check_loss_inputs
    def forward(self, x, y_pred, y_true, psf):
        psnr_value = self._psnr(y_pred, y_true)
        return psnr_value
        
    def to(self, device):
        """
        Move the PSNR metric to the specified device.
        """
        self._psnr.to(device)
        return self
    

class PSNR_PSF(PSNR):
    """
    Peak Signal-to-Noise Ratio (PSNR) metric for PSF.
    """
    def __str__(self):
        return "PSNR PSF"

    @check_loss_inputs
    def forward(self, x, y_pred, y_true, psf):
        diff = x - y_pred
        psnr_value = self._psnr(diff, psf)
        return psnr_value
    

class SSIM(nn.Module, Loss):
    """
    Structural Similarity Index Measure (SSIM) metric.
    This class computes the SSIM metric between two images.
    SSIM is a perception-based model that considers image degradation
    as perceived change in structural information,
    while also incorporating important perceptual phenomena.
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self._ssim = StructuralSimilarityIndexMeasure()

    def __str__(self):
        return "SSIM"

    @check_loss_inputs
    def forward(self, x, y_pred, y_true, psf):
        ssim_value = self._ssim(y_pred, y_true)
        return ssim_value
    
    def to(self, device):
        """
        Move the SSIM metric to the specified device.
        """
        self._ssim.to(device)
        return self
    

class SSIM_PSF(SSIM):
    """
    Structural Similarity Index Measure (SSIM) metric for PSF.
    """
    def __str__(self):
        return "SSIM PSF"

    @check_loss_inputs
    def forward(self, x, y_pred, y_true, psf):
        diff = x - y_pred
        ssim_value = self._ssim(diff, psf)
        return ssim_value
    

class CentroidError(nn.Module, Loss):
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

    @check_loss_inputs
    def forward(self, x, y_pred, y_true, psf):
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
        err = err / norm_factor

        return err.mean() # The mean error across the batch
    
    def _compute_centroid(self, img):
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
    

class FluxRecoveryFraction(nn.Module, Loss):
    """
    Computes Flux Recovery Fraction (FRF) of the galaxy images:
        FRF = sum(estimated PSF) / sum(true PSF)
    The FRF is a measure of how well the model recovers the flux of the PSF.
    The value is computed as the mean across the batch and is expected to be close to 1.0
    if the model is performing well and 0.0 if the model is not recovering the flux at all.
    """
    def __init__(self):
        super(FluxRecoveryFraction, self).__init__()

    def __str__(self):
        return "FRF"

    @check_loss_inputs
    def forward(self, x, y_pred, y_true, psf):
        return self._flux_recovery_fraction(y_pred, y_true)
    
    def _flux_recovery_fraction(self, pred, true):
        """
        Computes the flux recovery fraction for the estimated and true PSF.
        """
        # Compute the fluxes
        recovered_flux = pred.sum(dim=(1, 2, 3)) # shape: (B,)
        true_flux = true.sum(dim=(1, 2, 3)) # shape: (B,)

        frf = recovered_flux / (true_flux + 1e-8) - 1   # avoid div/0; shape: (B,)
        return frf.mean()  # The mean FRF across the batch
    

class FluxRecoveryFraction_PSF(FluxRecoveryFraction):
    """
    Computes Flux Recovery Fraction (FRF) of the PSF:
    """
    def __init__(self):
        super(FluxRecoveryFraction_PSF, self).__init__()

    def __str__(self):
        return "FRF PSF"

    @check_loss_inputs
    def forward(self, x, y_pred, y_true, psf):
        psf_est = x - y_pred
        return self._flux_recovery_fraction(psf_est, psf)
    

class RelativeFluxChange(nn.Module, Loss):
    """
    Computes the relative change in flux between predicted and true restored images:
        RFC = (Flux_pred - Flux_true) / Flux_true
    > 0: model didn't remove enough flux
    < 0: model removed too much flux
    = 0: perfect restoration
    """
    def __init__(self):
        super(RelativeFluxChange, self).__init__()

    def __str__(self):
        return "RFC"

    @check_loss_inputs
    def forward(self, x, y_pred, y_true, psf):
        return self._relative_flux_change(y_pred, y_true)
    
    def _relative_flux_change(self, pred, true):
        # Compute the fluxes
        flux_pred = pred.sum(dim=(1, 2, 3))
        flux_true = true.sum(dim=(1, 2, 3))

        # avoid div/0; shape: (B,)
        relative_change = (flux_pred - flux_true) / (flux_true + 1e-8)

        return relative_change.mean()  # The mean relative change across the batch
    

class RelativeFluxChange_PSF(RelativeFluxChange):
    """
    Computes the RFC of the true and predicted PSF.
    """
    def __init__(self):
        super(RelativeFluxChange_PSF, self).__init__()

    def __str__(self):
        return "RFC PSF"

    @check_loss_inputs
    def forward(self, x, y_pred, y_true, psf):
        psf_est = x - y_pred
        return self._relative_flux_change(psf, psf_est)
    

class ReconstructionLoss(nn.Module, Loss):
    """
    Base class for reconstruction loss.
    This class is used to compute the reconstruction loss between the input image and the predicted image.
    """
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self._loss_fn = nn.L1Loss()

    def __str__(self):
        return "Reconstruction Loss"

    @check_loss_inputs
    def forward(self, x, y_pred, y_true, psf):
        return self._loss_fn(y_pred, y_true)
    
    def to(self, device):
        """
        Move the reconstruction loss to the specified device.
        """
        self._loss_fn.to(device)
        return self
