"""
NOTE FOR USERS:

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
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import torch


class NormalizationParams:
    """
    Class to hold normalization parameters.
    This class is used to store parameters such as mean, std, min, and max
    for normalization and denormalization of images.
    It provides a simple interface to access these parameters for an
    arbitrary normalization.
    """
    def __init__(self, **kwargs):
        """
        Initialize the NormalizationParams with given parameters.

        :param kwargs: Arbitrary keyword arguments representing
        normalization parameters.
        :type kwargs: dict
        """
        self.params = kwargs

    def __str__(self):
        """
        String representation of the NormalizationParams object.

        :return: String representation of the NormalizationParams.
        :rtype: str
        """
        return str(self.params)

    def get(self, key: str, default: Any = None):
        """
        Get the value of a parameter by key.
        If the key does not exist, return the default value.

        :param key: The key of the parameter to retrieve.
        :type key: str
        :param default: The default value to return if the key does not exist.
        :type default: Any
        :return: The value of the parameter or the default value.
        :rtype: Any
        """
        return self.params.get(key, default)

    def __getitem__(self, key):
        """
        Get the value of a parameter by key using indexing.

        :param key: The key of the parameter to retrieve.
        :type key: str
        :return: The value of the parameter.
        :rtype: Any
        """
        return self.params[key]

    def __repr__(self):
        """
        String representation of the NormalizationParams object.

        :return: String representation of the NormalizationParams.
        :rtype: str
        """
        return f"NormalizationParams({self.params})"
    

class _BaseTransform(ABC):
    """
    Base class for all transforms.
    """
    @abstractmethod
    def __call__(self, input: torch.Tensor) -> tuple[torch.Tensor, NormalizationParams]:
        """
        Apply the transform to the input.

        :param input: to be transformed. The shape has to be
        (B, C, H, W).
        :type input: torch.Tensor
        :return: Transformed image.
        :rtype: torch.Tensor
        """
        pass

    @abstractmethod
    def inverse(self, input: torch.Tensor, params: NormalizationParams) -> torch.Tensor:
        """
        Inverse the transform.

        :param input: nromalized input to be inversed. The shape has to be
        (B, C, H, W).
        :type input: torch.Tensor
        :param params: Parameters used for the inverse transform.
        :type params: NormalizationParams
        :return: Inversed transform input.
        :rtype: torch.Tensor
        """
        pass


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
        :rtype: tuple[torch.Tensor, NormalizationParams]
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
        :rtype: tuple[torch.Tensor, NormalizationParams]
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
