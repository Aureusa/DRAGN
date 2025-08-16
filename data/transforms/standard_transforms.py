from typing import List, Tuple
import torch

from core.registry import TRANSFORM_REGISTRY
from .base_transforms import NormalizationParams, _BaseTransform


@TRANSFORM_REGISTRY.register("per_image_normalize")
class PerImageNormalize(_BaseTransform):
    """
    Normalize the by subtracting the mean and dividing by the standard deviation:
        normalized = (input - mean) / std
    """
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input, this method assumes that it has a shape (B, C, H, W).

        :param input: input to be normalized.
        :type input: torch.Tensor
        :return: Normalized and NormalizationParams containing mean and std.
        :rtype: Tuple[torch.Tensor, NormalizationParams]
        """
        mean = torch.mean(input, dim=(1,2,3), keepdim=True) # shape (B, C, 1, 1)
        std = torch.std(input, dim=(1,2,3), keepdim=True) # shape (B, C, 1, 1)

        normalized = (input - mean) / (std + 1e-8) # shape (B, C, H, W)
        return normalized, NormalizationParams(mean=mean, std=std)
    
    def inverse(self, input: torch.Tensor, params: NormalizationParams) -> torch.Tensor:
        """
        Inverse the normalization by multiplying by std and adding mean.
        This method assumes that the input is in the format (B, C, H, W).

        :param input: Normalized input to be denormalized.
        :type input: torch.Tensor
        :param params: NormalizationParams containing mean and std for the image.
        :type params: NormalizationParams
        :return: Denormalized images.
        :rtype: torch.Tensor
        """
        std = params.get("std").to(input.device)
        mean = params.get("mean").to(input.device)
        denormalized = input * std + mean
        return denormalized
    

@TRANSFORM_REGISTRY.register("per_image_min_max")
class PerImageMinMax(_BaseTransform):
    """
    Normalize the input by scaling it to the range [0, 1]:
        normalized = (input - min) / (max - min)
    """
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input, this method assumes that it has a shape (B, C, H, W).

        :param input: input to be normalized.
        :type input: torch.Tensor
        :return: Normalized and NormalizationParams containing min and max.
        :rtype: Tuple[torch.Tensor, NormalizationParams]
        """
        min_val = input.amin(dim=(1, 2, 3), keepdim=True)
        max_val = input.amax(dim=(1, 2, 3), keepdim=True)
        normalized = (input - min_val) / (max_val - min_val + 1e-8)
        return normalized, NormalizationParams(min=min_val, max=max_val)
    
    def inverse(self, input: torch.Tensor, params: NormalizationParams) -> torch.Tensor:
        """
        Normalize the input, this method assumes that it has a shape (B, C, H, W).

        :param input: Normalized input to be denormalized.
        :type input: torch.Tensor
        :param params: NormalizationParams containing min and max for the image.
        :type params: NormalizationParams
        :return: Denormalized images.
        :rtype: torch.Tensor
        """
        min_ = params.get("min").to(input.device)
        max_ = params.get("max").to(input.device)
        denormalized = input * (max_ - min_) + min_
        return denormalized


@TRANSFORM_REGISTRY.register("per_image_normalize_to_range")
class PerImageNormalizeToRange(_BaseTransform):
    """
    Normalize the input to a specified range (default [-1, 1]) by first applying
    min-max scaling to [0, 1], then scaling to the target range:
        Step 1: minmax_normalized = (input - min) / (max - min)
        Step 2: range_normalized = minmax_normalized * (max_range - min_range) + min_range
    
    This is useful for preparing data for GANs and other models that expect
    inputs in the [-1, 1] range.
    """
    
    def __init__(self, min_range: float = -1.0, max_range: float = 1.0):
        """
        Initialize the transform with target range.
        
        :param min_range: Minimum value of the target range.
        :type min_range: float
        :param max_range: Maximum value of the target range.
        :type max_range: float
        """
        self.min_range = min_range
        self.max_range = max_range
    
    def __call__(self, input: torch.Tensor) -> Tuple[torch.Tensor, NormalizationParams]:
        """
        Normalize the input to the specified range, this method assumes that it has a shape (B, C, H, W).

        :param input: input to be normalized.
        :type input: torch.Tensor
        :return: Normalized tensor and NormalizationParams containing original min, max, and range parameters.
        :rtype: Tuple[torch.Tensor, NormalizationParams]
        """
        # Get original min and max for each image
        original_min = input.amin(dim=(1, 2, 3), keepdim=True)
        original_max = input.amax(dim=(1, 2, 3), keepdim=True)
        
        # First normalize to [0, 1]
        minmax_normalized = (input - original_min) / (original_max - original_min + 1e-8)
        
        # Then scale to target range
        range_normalized = minmax_normalized * (self.max_range - self.min_range) + self.min_range
        
        return range_normalized, NormalizationParams(
            original_min=original_min, 
            original_max=original_max,
            min_range=self.min_range,
            max_range=self.max_range
        )
    
    def inverse(self, input: torch.Tensor, params: NormalizationParams) -> torch.Tensor:
        """
        Inverse the normalization to restore original range.

        :param input: Normalized input to be denormalized.
        :type input: torch.Tensor
        :param params: NormalizationParams containing original min, max, and range parameters.
        :type params: NormalizationParams
        :return: Denormalized images in original range.
        :rtype: torch.Tensor
        """
        original_min = params.get("original_min").to(input.device)
        original_max = params.get("original_max").to(input.device)
        min_range = params.get("min_range")
        max_range = params.get("max_range")
        
        # First convert back from target range to [0, 1]
        minmax_restored = (input - min_range) / (max_range - min_range)
        
        # Then restore to original range
        denormalized = minmax_restored * (original_max - original_min) + original_min
        
        return denormalized
    

@TRANSFORM_REGISTRY.register("per_image_adaptive_normalize")
class PerImageRobustNormalize(_BaseTransform):
    """
    Robust normalization using percentiles instead of min/max.
    This preserves fine details by not letting extreme outliers dominate the normalization.
    """
    
    def __init__(self, lower_percentile: float = 1.0, upper_percentile: float = 99.0, 
                 min_range: float = -1.0, max_range: float = 1.0):
        """
        Initialize robust normalization.
        
        :param lower_percentile: Lower percentile for clipping (e.g., 1.0 for 1st percentile)
        :param upper_percentile: Upper percentile for clipping (e.g., 99.0 for 99th percentile)
        :param min_range: Target minimum value
        :param max_range: Target maximum value
        """
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.min_range = min_range
        self.max_range = max_range
    
    def __call__(self, input: torch.Tensor) -> Tuple[torch.Tensor, NormalizationParams]:
        """
        Apply robust normalization using percentiles.
        
        :param input: Input tensor of shape (B, C, H, W)
        :return: Normalized tensor and normalization parameters
        """
        # Flatten spatial dimensions for percentile calculation
        batch_size, channels = input.shape[:2]
        flattened = input.view(batch_size, channels, -1)
        
        # Calculate percentiles for each image in the batch
        lower_vals = torch.quantile(flattened, self.lower_percentile/100.0, dim=2, keepdim=True)
        upper_vals = torch.quantile(flattened, self.upper_percentile/100.0, dim=2, keepdim=True)
        
        # Reshape back to original spatial dimensions
        lower_vals = lower_vals.unsqueeze(-1)  # (B, C, 1, 1)
        upper_vals = upper_vals.unsqueeze(-1)  # (B, C, 1, 1)
        
        # Clip extreme values
        clipped = torch.clamp(input, min=lower_vals, max=upper_vals)
        
        # Normalize to [0, 1] using robust min/max
        normalized_01 = (clipped - lower_vals) / (upper_vals - lower_vals + 1e-8)
        
        # Scale to target range
        normalized = normalized_01 * (self.max_range - self.min_range) + self.min_range
        
        return normalized, NormalizationParams(
            lower_vals=lower_vals,
            upper_vals=upper_vals,
            min_range=self.min_range,
            max_range=self.max_range,
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile
        )
    
    def inverse(self, input: torch.Tensor, params: NormalizationParams) -> torch.Tensor:
        """
        Inverse the robust normalization.
        
        :param input: Normalized input
        :param params: Normalization parameters
        :return: Denormalized tensor
        """
        lower_vals = params.get("lower_vals").to(input.device)
        upper_vals = params.get("upper_vals").to(input.device)
        min_range = params.get("min_range")
        max_range = params.get("max_range")
        
        # Convert back from target range to [0, 1]
        normalized_01 = (input - min_range) / (max_range - min_range)
        
        # Scale back to clipped range
        denormalized = normalized_01 * (upper_vals - lower_vals) + lower_vals
        
        return denormalized


@TRANSFORM_REGISTRY.register("per_image_log_normalize")
class PerImageLogNormalize(_BaseTransform):
    """
    Log-based normalization for images with extreme dynamic range.
    Uses log(1 + x) transformation to compress dynamic range while preserving details.
    """
    
    def __init__(self, min_range: float = -1.0, max_range: float = 1.0):
        """
        Initialize log normalization.
        
        :param min_range: Target minimum value
        :param max_range: Target maximum value
        """
        self.min_range = min_range
        self.max_range = max_range
    
    def __call__(self, input: torch.Tensor) -> Tuple[torch.Tensor, NormalizationParams]:
        """
        Apply log normalization.
        
        :param input: Input tensor of shape (B, C, H, W)
        :return: Normalized tensor and normalization parameters
        """
        # Store original min for shifting
        original_min = input.amin(dim=(1, 2, 3), keepdim=True)
        
        # Shift to make all values positive (add small epsilon for safety)
        shifted = input - original_min + 1e-8
        
        # Apply log transformation
        log_transformed = torch.log1p(shifted)  # log(1 + x)
        
        # Get min/max of log-transformed data
        log_min = log_transformed.amin(dim=(1, 2, 3), keepdim=True)
        log_max = log_transformed.amax(dim=(1, 2, 3), keepdim=True)
        
        # Normalize to [0, 1]
        normalized_01 = (log_transformed - log_min) / (log_max - log_min + 1e-8)
        
        # Scale to target range
        normalized = normalized_01 * (self.max_range - self.min_range) + self.min_range
        
        return normalized, NormalizationParams(
            original_min=original_min,
            log_min=log_min,
            log_max=log_max,
            min_range=self.min_range,
            max_range=self.max_range
        )
    
    def inverse(self, input: torch.Tensor, params: NormalizationParams) -> torch.Tensor:
        """
        Inverse the log normalization.
        
        :param input: Normalized input
        :param params: Normalization parameters
        :return: Denormalized tensor
        """
        original_min = params.get("original_min").to(input.device)
        log_min = params.get("log_min").to(input.device)
        log_max = params.get("log_max").to(input.device)
        min_range = params.get("min_range")
        max_range = params.get("max_range")
        
        # Convert back from target range to [0, 1]
        normalized_01 = (input - min_range) / (max_range - min_range)
        
        # Scale back to log range
        log_restored = normalized_01 * (log_max - log_min) + log_min
        
        # Apply inverse log transformation
        shifted_restored = torch.expm1(log_restored)  # exp(x) - 1
        
        # Restore original range
        denormalized = shifted_restored + original_min - 1e-8
        
        return denormalized


@TRANSFORM_REGISTRY.register("compose")
class TransformCompose(_BaseTransform):
    """
    Compose multiple transforms
    into a single transform that applies them sequentially.
    This allows you to chain multiple normalization strategies together.
    """
    def __init__(self, transforms: List[_BaseTransform]):
        self.transforms = transforms
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, NormalizationParams]:
        combined_params = {}
        current_data = x
        
        for i, transform in enumerate(self.transforms):
            current_data, params = transform(current_data)
            combined_params[f"transform_{i}_{type(transform).__name__}"] = params

        return current_data, NormalizationParams(combined_params)

    def inverse(self, x: torch.Tensor, params: NormalizationParams) -> torch.Tensor:
        current_data = x
        # Apply inverse transforms in reverse order
        for i in reversed(range(len(self.transforms))):
            transform = self.transforms[i]
            transform_key = f"transform_{i}_{type(transform).__name__}"
            transform_params = params.get(transform_key)
        
            if transform_params is not None:
                current_data = transform.inverse(current_data, transform_params)
        return current_data
    