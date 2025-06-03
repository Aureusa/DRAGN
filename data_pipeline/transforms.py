"""
NOTE FOR USERS:

This module contains image transformation and normalization
utilities for AGN/galaxy datasets.
**Deprecated:** None of the current models in this package use
these transformations.
You likely do not need to use or modify this file for standard workflows.
"""
from abc import ABC, abstractmethod
from typing import Any
import numpy as np


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
    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, NormalizationParams]:
        """
        Apply the transform to the image.

        :param img: Image to be transformed.
        :type img: np.ndarray
        :return: Transformed image.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def inverse(self, img: np.ndarray, params: NormalizationParams) -> np.ndarray:
        """
        Inverse the transform.

        :param img: Image to be inversed.
        :type img: np.ndarray
        :param params: Parameters used for the inverse transform.
        :type params: Any
        :return: Inversed image.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def batched_inverse(self, img: np.ndarray, params_list: list[NormalizationParams]) -> np.ndarray:
        """
        Inverse the transform for a batch of images.

        :param img: Batch of images to be inversed.
        :type img: np.ndarray
        :param params_list: List of parameters used for the inverse transform.
        :type params_list: list
        :return: Inversed batch of images.
        :rtype: np.ndarray
        """
        pass


class PerImageNormalize(_BaseTransform):
    """
    Normalize the image by subtracting the mean and dividing by the standard deviation:
        normalized = (img - mean) / std
    """
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize the image by subtracting the mean and dividing by the standard deviation.
        This method assumes that the input image is in the format (H, W) or (H, W, C).

        :param img: Image to be normalized.
        :type img: np.ndarray
        :return: Normalized image and NormalizationParams containing mean and std.
        :rtype: tuple[np.ndarray, NormalizationParams]
        """
        mean = img.mean()
        std = img.std()
        normalized = (img - mean) / std
        return normalized, NormalizationParams(mean=mean, std=std)
    
    def batched_inverse(self, img: np.ndarray, params_list: list[NormalizationParams]) -> np.ndarray:
        """
        Inverse normalization for a batch of images.
        This method assumes that the input image is in the format (B, H, W)

        :param img: Batch of images to denormalize.
        :type img: np.ndarray
        :param params_list: List of NormalizationParams objects
        containing mean and std for each image.
        :type params_list: list[NormalizationParams]
        :return: Denormalized batch of images.
        :rtype: np.ndarray
        """
        # Ensure params is a list of NormalizationParams
        if not isinstance(params_list, list) or not all(isinstance(p, NormalizationParams) for p in params_list):
            raise ValueError("params must be a list of NormalizationParams objects")
        
        # Ensure img is a numpy array
        if not isinstance(img, np.ndarray):
            raise ValueError("img must be a numpy array")
        
        # Denormalize each image in the batch
        # Assuming img is in the format (Batch, Height, Width)
        means = np.array([p['mean'] for p in params_list]).reshape(-1, 1, 1)
        stds = np.array([p['std'] for p in params_list]).reshape(-1, 1, 1)
        return img * stds + means
    
    def inverse(self, img: np.ndarray, params: NormalizationParams) -> np.ndarray:
        denormalized = img * params["std"] + params["mean"]
        return denormalized
    

class PerImageMinMax(_BaseTransform):
    """
    Normalize the image by scaling it to the range [0, 1]:
        normalized = (img - min) / (max - min)
    """
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize the image by scaling it to the range [0, 1].
        This method assumes that the input image is in the format
        (H, W).

        :param img: Image to be normalized.
        :type img: np.ndarray
        :return: Normalized image and NormalizationParams containing min and max.
        :rtype: tuple[np.ndarray, NormalizationParams]
        """
        min_val = img.min()
        max_val = img.max()
        normalized = (img - min_val) / (max_val - min_val)
        return normalized, NormalizationParams(min=min_val, max=max_val)
    
    def inverse(self, img: np.ndarray, params: NormalizationParams) -> np.ndarray:
        """
        Inverse normalization for a single image.
        This method assumes that the input image is in the format (H, W).

        :param img: Image to be denormalized.
        :type img: np.ndarray
        :param params: NormalizationParams containing min and max for the image.
        :type params: NormalizationParams
        :return: Denormalized image.
        :rtype: np.ndarray
        """
        denormalized = img * (params["max"] - params["min"]) + params["min"]
        return denormalized
    
    def batched_inverse(self, img: np.ndarray, params_list: list[NormalizationParams]) -> np.ndarray:
        """
        Inverse normalization for a batch of images.
        This method assumes that the input image is in the format (B, H, W)
        :param img: Batch of images to denormalize.
        :type img: np.ndarray
        :param params_list: List of NormalizationParams objects
        containing min and max for each image.
        :type params_list: list[NormalizationParams]
        :return: Denormalized batch of images.
        :rtype: np.ndarray
        """
        # Ensure params is a list of NormalizationParams
        if not isinstance(params_list, list) or not all(isinstance(p, NormalizationParams) for p in params_list):
            raise ValueError("params must be a list of NormalizationParams objects")
        
        # Ensure img is a numpy array
        if not isinstance(img, np.ndarray):
            raise ValueError("img must be a numpy array")
        
        # Denormalize each image in the batch
        # Assuming img is in the format (Batch, Height, Width)
        mins = np.array([p['min'] for p in params_list]).reshape(-1, 1, 1)
        maxs = np.array([p['max'] for p in params_list]).reshape(-1, 1, 1)
        return img * (maxs - mins) + mins
    