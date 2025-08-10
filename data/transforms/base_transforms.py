from abc import abstractmethod
from typing import Any, Tuple
import torch

from core.component import Component


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
    

class _BaseTransform(Component):
    """
    Base class for all transforms.
    """
    @classmethod
    def from_config(cls, params) -> '_BaseTransform':
        return cls(**params)

    @abstractmethod
    def __call__(self, input: torch.Tensor) -> Tuple[torch.Tensor, NormalizationParams]:
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
