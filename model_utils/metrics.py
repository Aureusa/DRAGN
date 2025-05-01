import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from model_utils.loss_functions import get_loss_function, Loss
from utils import print_box


def get_metrics(metrics: list[str]) -> None:
    metrics_dict = {
            "PSNR": PSNR(),
            "PSF PSNR": PSF_PSNR(),
            "SSIM": SSIM(),
            "PSF SSIM": PSF_SSIM(),

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

    def forward(self, x, y_pred, y_true, psf):
        psnr_value = self._psnr(y_pred, y_true)
        return psnr_value
        
    def to(self, device):
        """
        Move the PSNR metric to the specified device.
        """
        self._psnr.to(device)
        return self
    

class PSF_PSNR(PSNR):
    """
    Peak Signal-to-Noise Ratio (PSNR) metric for PSF.
    """
    def __str__(self):
        return "PSF PSNR"

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

    def forward(self, x, y_pred, y_true, psf):
        ssim_value = self._ssim(y_pred, y_true)
        return ssim_value
    
    def to(self, device):
        """
        Move the SSIM metric to the specified device.
        """
        self._ssim.to(device)
        return self
    

class PSF_SSIM(SSIM):
    """
    Structural Similarity Index Measure (SSIM) metric for PSF.
    """
    def __str__(self):
        return "PSF SSIM"

    def forward(self, x, y_pred, y_true, psf):
        diff = x - y_pred
        ssim_value = self._ssim(diff, psf)
        return ssim_value
    