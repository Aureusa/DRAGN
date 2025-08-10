from typing import Tuple
import torch

from core.registry import TRANSFORM_REGISTRY
from .base_transforms import NormalizationParams, _BaseTransform


@TRANSFORM_REGISTRY.register("per_image_asinh_normalize")
class PerImageAsinhNormalize(_BaseTransform):
    """
    Asinh (inverse hyperbolic sine) normalization for astronomical images.
    This is very popular in astronomy as it compresses bright features while 
    preserving faint details much better than linear scaling.
    
    The asinh function: asinh(x) = ln(x + sqrt(x^2 + 1))
    - Linear for small values (preserves faint details)
    - Logarithmic for large values (compresses bright spikes)
    - Smooth transition between the two regimes
    """
    
    def __init__(self, stretch_factor: float = 1.0, min_range: float = -1.0, max_range: float = 1.0):
        """
        Initialize asinh normalization.
        
        :param stretch_factor: Controls the transition point between linear and log behavior.
                              Higher values = more linear (preserves more faint detail)
                              Lower values = more logarithmic (more compression of bright features)
                              Typical values: 0.1 to 10.0
        :param min_range: Target minimum value
        :param max_range: Target maximum value
        """
        self.stretch_factor = stretch_factor
        self.min_range = min_range
        self.max_range = max_range
    
    def __call__(self, input: torch.Tensor) -> Tuple[torch.Tensor, NormalizationParams]:
        """
        Apply asinh normalization.
        
        :param input: Input tensor of shape (B, C, H, W)
        :return: Normalized tensor and normalization parameters
        """
        # Store original min for shifting to positive values
        original_min = input.amin(dim=(1, 2, 3), keepdim=True)
        
        # Shift to make all values non-negative
        shifted = input - original_min
        
        # Apply asinh transformation with stretch factor
        # asinh(x/stretch) scales the input before applying asinh
        asinh_transformed = torch.asinh(shifted / self.stretch_factor)
        
        # Get min/max of asinh-transformed data for normalization
        asinh_min = asinh_transformed.amin(dim=(1, 2, 3), keepdim=True)
        asinh_max = asinh_transformed.amax(dim=(1, 2, 3), keepdim=True)
        
        # Normalize to [0, 1]
        normalized_01 = (asinh_transformed - asinh_min) / (asinh_max - asinh_min + 1e-8)
        
        # Scale to target range
        normalized = normalized_01 * (self.max_range - self.min_range) + self.min_range
        
        return normalized, NormalizationParams(
            original_min=original_min,
            asinh_min=asinh_min,
            asinh_max=asinh_max,
            stretch_factor=self.stretch_factor,
            min_range=self.min_range,
            max_range=self.max_range
        )
    
    def inverse(self, input: torch.Tensor, params: NormalizationParams) -> torch.Tensor:
        """
        Inverse the asinh normalization.
        
        :param input: Normalized input
        :param params: Normalization parameters
        :return: Denormalized tensor
        """
        original_min = params.get("original_min").to(input.device)
        asinh_min = params.get("asinh_min").to(input.device)
        asinh_max = params.get("asinh_max").to(input.device)
        stretch_factor = params.get("stretch_factor")
        min_range = params.get("min_range")
        max_range = params.get("max_range")
        
        # Convert back from target range to [0, 1]
        normalized_01 = (input - min_range) / (max_range - min_range)
        
        # Scale back to asinh range
        asinh_restored = normalized_01 * (asinh_max - asinh_min) + asinh_min
        
        # Apply inverse asinh transformation (sinh)
        shifted_restored = torch.sinh(asinh_restored) * stretch_factor
        
        # Restore original range
        denormalized = shifted_restored + original_min
        
        return denormalized
