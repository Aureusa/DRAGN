import numpy as np
import torch

from core.registry import DATASET_REGISTRY
from .base_dataset import _BaseDataset
from ..transforms import NormalizationParams, _BaseTransform
from ..utils import load_fits_data, center_crop
    

@DATASET_REGISTRY.register("galaxy_dataset")
class GalaxyDataset(_BaseDataset):
    def __init__(
        self,
        source: list[str],
        target: list[str],
        transform: _BaseTransform|None = None,
        **kwargs
    ):
        """
        Initialize the GalaxyDataset class.
        This class is a concrete implementation of the _BaseDataset class,
        designed to load AGN and AGN-free images from FITS files.

        :param source: The list of AGN image file paths.
        :type source: list[str]
        :param target: The list of AGN-free image file paths.
        :type target: list[str]
        :param transform: The transformation to apply to the images.
        :type transform: _BaseTransform|None
        :param training: Whether the dataset is for training or not.
        :type training: bool
        """
        super().__init__(source, target, transform)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get the source-target pair at the given index.
        The source is the AGN image and the target is the AGN-free image.

        :param idx: The index of the source-target pair to retrieve.
        :type idx: int
        :return: A tuple containing the input tensor and target tensor.
                 If training is True, it returns (input_tensor, target_tensor).
                 If training is False, it returns (input_tensor, target_tensor, input_norm_params).
        :rtype: tuple[torch.Tensor, torch.Tensor, NormalizationParams|None]
        """
        # Get the source-target pair at the given index.
        input_filepath, target_filepath = self.st_pairs[idx]

        # Load the AGN file and AGN-free file
        input_data = load_fits_data(input_filepath)
        target_data = load_fits_data(target_filepath)

        # Preprocess the input data
        input_tensor, input_norm_params = self._process_data(input_data, transform=True)
        target_tensor, target_norm_params = self._process_data(target_data, transform=True)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "input_norm_params": input_norm_params,
            "target_norm_params": target_norm_params
        }
    
    def _process_data(self, data: np.ndarray, transform: bool = True) -> tuple[np.ndarray, NormalizationParams]:
        """
        Process the input data and return it as a tensor.

        :param data: The input data to process.
        :type data: np.ndarray
        :param transform: Whether to apply the transformation to the data.
        :type transform: bool
        :return: A tuple containing the processed data as a tensor and the normalization parameters.
        :rtype: tuple[np.ndarray, NormalizationParams]
        """
        # Initialize normalization parameters
        data_norm_param = NormalizationParams()

        # Convert to 2D arrays if the AGN free image is 3D
        if len(data.shape) == 3:
            data = data[0]

        # Convert the data to native-endian format before creating a tensor
        data = data.astype(np.float32, copy=False)

        data = center_crop(data, 128, 128)

        if transform and self.transform is not None:
            data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            # Normalize the images
            data, data_norm_param = self.transform(data)
            return data, data_norm_param

        # Convert the data to torch tensors
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0) # (1, H, W)
        return data, data_norm_param
    