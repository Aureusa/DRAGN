"""
This module provides transformation and normalization
utilities for AGN/galaxy datasets. The main classes here implement
per-image normalization transforms that operate on PyTorch tensors 
with shape (B, C, H, W), where B is batch size, C is channels,
and H, W are spatial dimensions.

**How these methods work:**
- Each normalization transform is a callable class that takes
  a tensor input of shape (B, C, H, W)
  and returns a tuple: (normalized_tensor, NormalizationParams).
- The `NormalizationParams` class stores the parameters
  (mean/std or min/max) used for normalization, allowing you to later
  invert (denormalize) the transformation using the `inverse` method.

**Available normalizations:**
- `PerImageNormalize`: Normalizes each image in the batch by subtracting
  its mean and dividing by its standard deviation.
- `PerImageMinMax`: Normalizes each image in the batch to the [0, 1]
  range using its own min and max.
- `PerImageNormalizeToRange`: Normalizes each image in the batch to a 
  specified range (default [-1, 1]) using min-max scaling. Useful for
  preparing data for GANs and models expecting [-1, 1] input range.
- `PerImageRobustNormalize`: Uses percentiles instead of min/max to handle
  extreme outliers while preserving fine details.
- `PerImageLogNormalize`: Uses log(1+x) transformation for extreme dynamic range.
- `PerImageAsinhNormalize`: Uses asinh transformation - ideal for astronomical
  images as it's linear for faint features and logarithmic for bright spikes.
- `PerImageAdaptiveNormalize`: Combines different strategies for different
  brightness regions.

**NormalizationParams:**
- This is a simple container for the normalization parameters
  (mean, std, min, max, etc.) for each image.
- It allows you to easily retrieve these parameters for denormalization.

**Input shape:**
- All transforms expect input tensors of shape (B, C, H, W).

**Example usage:**
    from data_pipeline.transforms import PerImageNormalize

    transform = PerImageNormalize()

    # images: torch.Tensor of shape (B, C, H, W)
    normalized, norm_params = transform(images)
    restored = transform.inverse(normalized, norm_params)

Use these transforms to preprocess your data before training or evaluation,
and to invert normalization for visualization or metric calculation.
"""
from .base_transforms import _BaseTransform, NormalizationParams
from .astro_transforms import PerImageAsinhNormalize
from .standard_transforms import (
    PerImageNormalize,
    PerImageMinMax,
    PerImageNormalizeToRange,
    PerImageRobustNormalize,
    PerImageLogNormalize,
    TransformCompose
) 

__all__ = ["NormalizationParams", "PerImageAsinhNormalize", "PerImageNormalize", "PerImageMinMax", "PerImageNormalizeToRange", "PerImageRobustNormalize", "PerImageLogNormalize", "TransformCompose"]
