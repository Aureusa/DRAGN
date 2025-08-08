"""
NOTE FOR USERS:

This module provides dataset classes for AGN/galaxy image pairs, designed to work **exclusively**
with the file naming conventions described in the data database (`_telescopes_db.py`) of this package.

**Key Points:**
- All datasets must inherit from the `_BaseDataset` abstract class. If you wish to
implement a custom dataset, your class should inherit from `_BaseDataset` and implement
the required methods.
- The dataset classes are tailored for FITS files with AGN fraction encoded in the
filename using the pattern: `_f(.?)` (see `AGN FRACTION PATTERN` in the data database).
- The methods `filter_by_f_agn` and `filter_by_f_agn_list` in `_BaseDataset` rely on this
filename pattern to filter data by AGN fraction. If your files do not follow this pattern,
these methods will not work as intended.
- The source (AGN) and target (AGN-free) images must be paired according to the conventions
and patterns described in the data database. These pairs are expected to be the same as the output
from the `ForgeData` class's `forge_training_data` method.

**Important:**
- Only use these dataset classes with data that strictly follows the expected
FITS filename patterns.
- If you wish to use your own data, you must adapt your filenames to match these
patterns or modify the code accordingly.
- For more information about the data structure, refer to the documentation or
contact the maintainers.
"""
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import re
import torch
from torch.utils.data import Dataset

from core.registry import DATASET_REGISTRY
from data_pipeline.transforms import NormalizationParams, _BaseTransform
from data_pipeline._telescopes_db import TELESCOPES_DB
from data_pipeline.utils import load_fits_data, center_crop
from utils.printing import print_box
from utils.validation import validate_numpy_array, validate_list, validate_type
from utils.warnings import DRAGNWarning
import random


class _BaseDataset(Dataset, ABC):
    """
    Base class for datasets that load AGN and AGN-free images.
    This class is an abstract base class that provides common functionality
    for loading and processing AGN and AGN-free images from FITS files.
    It is designed to be inherited by specific dataset classes that implement
    the `__getitem__` method to return the source-target pairs.
    
    The source is the AGN image and the target is the AGN-free image.
    The source and target lists should be of the same length,
    where each index corresponds to a source-target pair.
    
    This is effectively a wraper around the `torch.utils.data.Dataset` class,
    providing additional functionality for filtering and processing the dataset.
    """
    def __init__(
        self,
        source: list[str],
        target: list[str],
        transform: _BaseTransform|None = None,
        training: bool = True
    ):
        """
        Initialize the BaseDataset class.
        This class serves as a base class for loading the galaxy
        AGN and AGN-free images from the given file groups.
        The source is the AGN image and the target is the AGN-free image.
        For consistency, the source and target lists should be of the same length,
        where each index corresponds to a source-target pair.

        :param source: The list of AGN image file paths.
        :type source: list[str]
        :param target: The list of AGN-free image file paths.
        :type target: list[str]
        :param transform: The transformation to apply to the images.
        :type transform: _BaseTransform|None
        :param training: Whether the dataset is for training or not.
        :type training: bool
        """
        validate_list(source, str)
        validate_list(target, str)
        validate_type(transform, _BaseTransform, allow_none=True)
        
        self.st_pairs = list(zip(source, target))
        self.transform = transform
        self.training = training

    def get_all_f_agn(self) -> dict[int, int]:
        """
        Get a dictionary of AGN fractions and their counts in the dataset.
        The AGN fraction is extracted from the source file names
        using the pattern `_f(\d+)`, where `\d+` is one or more digits.
        The keys of the dictionary are the AGN fractions (as integers),
        and the values are the counts of source-target pairs with that AGN fraction.
        
        :return: A dictionary where keys are AGN fractions and values are counts.
        :rtype: dict[int, int]
        """
        f_agn_dict = {}
        for pair in self.st_pairs:
            match = re.search(r"_f(\d+)", pair[0])
            if match:
                f_agn = int(match.group(1))
                if f_agn not in f_agn_dict:
                    f_agn_dict[f_agn] = 1
                else:
                    f_agn_dict[f_agn] += 1
        return f_agn_dict

    # DEPRICATED: Needs to be removed in the future
    def filter_by_f_agn(self, f_agn: int) -> list[int]:
        """
        Get the indices of the source-target pairs that have the given AGN fraction.

        :param f_agn: The AGN fraction to filter by.
        Example: 
            10 for 0.10
            30 for 0.30
            .
            .
            .
        :type f_agn: int
        :return: The indices of the source-target pairs that have the given AGN fraction.
        :rtype: list[int]
        """
        validate_type(f_agn, int)
        
        pattern = re.compile(rf"_f{f_agn}")
        new_st_pairs = [
            pair for pair in self.st_pairs if pattern.search(pair[0])
        ]
        self.st_pairs = new_st_pairs
        print_box(f"Filtered dataset to {len(self.st_pairs)} pairs with AGN fraction f_agn = 0.{f_agn}.")
    
    def filter_by_f_agn_list(self, f_agn_list: list[int]|int, n: int = float("inf")) -> None: # New name: filter_by_f_agn 
        """
        Filter the dataset by a list of AGN fractions.

        :param f_agn_list: The list of int or int of AGN fractions
        to filter by.
        :type f_agn_list: list[int]
        """
        # Newer Version of the method
        if isinstance(f_agn_list, int):
            f_agn_list = [f_agn_list]
        validate_list(f_agn_list, int)

        # Not Depricated
        new_st_pairs = []
        '''
        # DEPRICATED:
        f_agn_list = [10, 30, 44, 65, 70, 90]
        '''

        # Newer Version of the method
        for f_agn in f_agn_list:
            pattern = re.compile(rf"_f{f_agn}")
            count = 0
            for pair in self.st_pairs:
                if pattern.search(pair[0]):
                    new_st_pairs.append(pair)
                    count += 1
                    if len(new_st_pairs) == 136: # DEPRICATED: Needs to be removed in the future
                        self.st_pairs = new_st_pairs # DEPRICATED: Needs to be removed in the future
                        return # DEPRICATED: Needs to be removed in the future
                    if n == count:
                        break

        '''
        # DEPRICATED:
        for f, f_agn in enumerate(f_agn_list):
            count = 0
            pattern = re.compile(rf"_f{f_agn}")
            for pair in self.st_pairs:
                if pattern.search(pair[0]):
                    count += 1

                    if f == 0 and count == 2:
                        new_st_pairs.append(pair)
                        break

                    if f == 1 and count == 5:
                        new_st_pairs.append(pair)
                        break

                    if f == 2 and count == 5:
                        new_st_pairs.append(pair)
                        break

                    if f == 3 and count == 1:
                        new_st_pairs.append(pair)
                        break

                    if f == 4 and count == 2:
                        new_st_pairs.append(pair)
                        break

                    if f == 5 and count == 2:
                        new_st_pairs.append(pair)
                        break
        '''
        # Not Depricated
        self.st_pairs = new_st_pairs
        print_box(f"Filtered dataset to {len(self.st_pairs)} pairs with AGN fractions: {f_agn_list}.")

    def get_n_rand_gal(self, n: int) -> list[tuple[str, str]]:
        """
        Get n random source-target pairs from the dataset.

        :param n: The number of source-target pairs to select.
        :type n: int
        :return: A list of n randomly selected source-target pairs.
        :rtype: list[tuple[str, str]]
        """
        validate_type(n, int)

        # Add a seed for reproducibility
        random.seed(42)
        
        if n > len(self.st_pairs):
            n = len(self.st_pairs)
        self.st_pairs = random.sample(self.st_pairs, n)

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        :return: The number of source-target pairs in the dataset.
        :rtype: int
        """
        return len(self.st_pairs)
    
    @abstractmethod
    def __getitem__(self, idx: int) -> tuple:
        """
        Get the source-target pair at the given index.
        The source is the AGN image and the target is the AGN-free image.
        """
        pass
    

class GalaxyDataset(_BaseDataset):
    def __init__(
        self,
        source: list[str],
        target: list[str],
        transform: _BaseTransform|None = None,
        training: bool = True,
        condition_on_f_agn: bool = False
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
        super().__init__(source, target, transform, training)
        self._condition_on_f_agn = condition_on_f_agn

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

        # If condition_on_f_agn is True, condition the input tensor on the AGN fraction
        if self._condition_on_f_agn:
            input_tensor = self._condition_input_tensor_on_f_agn(input_tensor, input_filepath)
        
        if self.training:
            target_tensor, _ = self._process_data(target_data, transform=True)
            return input_tensor, target_tensor
        else:
            target_tensor, _ = self._process_data(target_data, transform=False)
            return input_tensor, target_tensor, input_norm_params
        
    def _condition_input_tensor_on_f_agn(self, input_tensor: torch.Tensor, input_filepath: str) -> torch.Tensor:
        """
        Condition the input tensor based on the AGN fraction.
        This method adds another channel dimension full with the
        f_agn fraction of the input image.

        :param input_tensor: The input tensor to condition.
        :type input_tensor: torch.Tensor
        :param f_agn: The AGN fraction to condition on.
        :type f_agn: float
        :return: The conditioned input tensor.
        :rtype: torch.Tensor
        """
        # Get the AGN fraction from the input file name
        match = re.search(
            TELESCOPES_DB["AGN FRACTION PATTERN"], # r"_f(.*?)\\.fits"
            input_filepath
        )

        # Get the AGN fraction from the match
        if match:
            f_agn = int(match.group(1)) / 100
        else:
            f_agn = 0.44  # Default value if not found
        
        # Get the shape of the input tensor
        _, height, width = input_tensor.shape
        
        # Add the AGN fraction as a new channel
        f_agn_tensor = torch.full((1, height, width), f_agn, dtype=torch.float32) # (B, 1, H, W) full with f_agn value
        input_tensor = torch.cat((input_tensor, f_agn_tensor), dim=0)
        return input_tensor

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
        data_norm_param = None

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
    

class GalaxyDatasetPSFCond(_BaseDataset):
    def __init__(
        self,
        source: list[str],
        target: list[str],
        transform: _BaseTransform|None = None,
        training: bool = True,
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
        super().__init__(source, target, transform, training)

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
        
        if self.training:
            target_tensor, _ = self._process_data(target_data, transform=True)

            input_tensor = self._condition_input_tensor_on_psf(input_tensor, target_tensor)
            return input_tensor, target_tensor
        else:
            target_tensor, _ = self._process_data(target_data, transform=False)

            input_tensor = self._condition_input_tensor_on_psf_real(input_tensor)
            DRAGNWarning().warn("Using empirical PSF loaded from the file system. If you are using mock dataset, change this in the GalaxyDatasetPSFCond().__getitem__() definition.")
            return input_tensor, target_tensor, input_norm_params
        
    def _condition_input_tensor_on_psf(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        psf_tensor = input_tensor - target_tensor

        # Normalize the PSF tensor
        eps = 1e-8
        max_val = torch.max(torch.abs(psf_tensor))
        psf_tensor_norm = psf_tensor / (max_val + eps)

        input_tensor = torch.cat((input_tensor, psf_tensor_norm), dim=0)
        return input_tensor

    def _condition_input_tensor_on_psf_real(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # 2D PSF data (np.ndarray)
        psf = load_fits_data("/home4/s4683099/Deep-AGN-Clean/testing_folder/jwst_data/psf.fits")
        psf = psf.astype(np.float32, copy=False)
        
        # Convert the PSF to a tensor
        psf_tensor = torch.tensor(psf, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        # Normalize the PSF tensor
        eps = 1e-8
        max_val = torch.max(torch.abs(psf_tensor))
        psf_tensor_norm = psf_tensor / (max_val + eps)

        input_tensor = torch.cat((input_tensor, psf_tensor_norm), dim=0)

        return input_tensor
    
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
        data_norm_param = None

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
            data = data.squeeze(0)
            return data, data_norm_param

        # Convert the data to torch tensors
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0) # (1, H, W)
        return data, data_norm_param
    
    
class MockRealGalaxyDataset(_BaseDataset):
    """
    Mock dataset for testing purposes.
    This dataset is used to test the functionality of the data pipeline
    without requiring actual data files.
    """
    def __init__(
            self,
            real_images: list[str],
            source: list[str],
            target: list[str],
            transform: _BaseTransform|None = None,
            training: bool = True
        ):
        validate_list(source, str)
        validate_list(target, str)
        validate_list(real_images, str)
        validate_type(transform, _BaseTransform, allow_none=True)
        
        self.real_images = real_images
        self.st_pairs = list(zip(source, target))
        self.transform = transform
        self.training = training
        

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
        if len(self.st_pairs) != len(self.real_images):
            new_st = self.st_pairs[:len(self.real_images)]
            self.st_pairs = new_st
        
        # Get the source-target pair at the given index.
        source, target = self.st_pairs[idx]

        # Load the AGN file and AGN-free file
        source_data = load_fits_data(source)
        target_data = load_fits_data(target)

        # Load the real image
        real_image_path = self.real_images[idx]
        real_image_data = load_fits_data(real_image_path)

        # Preprocess the data
        source_tensor, _ = self._process_data(source_data, transform=True)
        target_tensor, _ = self._process_data(target_data, transform=True)

        # Compute the psf
        psf_tensor = source_tensor - target_tensor
        
        # Preprocess the real image
        real_image_tensor, _ = self._process_data(real_image_data, transform=True)

        # Add the psf to the real image tensor
        input_tensor = real_image_tensor + psf_tensor
        input_norm_params = None

        if self.training:
            target_tensor, _ = self._process_data(target_data, transform=True)
            return input_tensor, real_image_tensor
        else:
            target_tensor, _ = self._process_data(target_data, transform=False)
            return input_tensor, real_image_tensor, input_norm_params
        
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
        data_norm_param = None

        # Convert to 2D arrays if the AGN free image is 3D
        if len(data.shape) == 3:
            data = data[0]

        # Convert the data to native-endian format before creating a tensor
        data = data.astype(np.float32, copy=False)

        data = center_crop(data, 128, 128)

        if transform and self.transform is not None:
            data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

            # Normalize the images
            data, data_norm_param = self.transform(data)
            data = data.squeeze(0)
            return data, data_norm_param

        # Convert the data to torch tensors
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return data, data_norm_param
