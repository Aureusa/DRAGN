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
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from data_pipeline.transforms import NormalizationParams, _BaseTransform
from data_pipeline.utils import load_fits_data, center_crop
from utils import print_box
from utils_utils.validation import validate_numpy_array, validate_list, validate_type


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

    def filter_first_n(self, n: int) -> list[tuple[str, str]]:
        """
        Filter the first n source-target pairs from the dataset.

        :param n: The number of source-target to filter.
        :type n: int
        """
        validate_type(n, int)
        
        self.st_pairs = self.st_pairs[:n]

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
        training: bool = True
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

    def _fits_below_threshold(self, pair, threshold):
        input_data = load_fits_data(pair[0], max_val=True)
        return input_data < threshold

    def treshhold_by_pixel_value2(self, threshold: float = 1500, max_workers=1) -> None:
        """
        Threshold the dataset by pixel value using multiple cores.
        """
        validate_type(threshold, (int, float))

        # Use ThreadPoolExecutor for parallel I/O
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(lambda pair: self._fits_below_threshold(pair, threshold), self.st_pairs),
                total=len(self.st_pairs)
            ))

        new_st_pairs = [pair for pair, keep in zip(self.st_pairs, results) if keep]

        print_box(f"Filtered dataset to {len(new_st_pairs)} pairs with pixel value threshold < {threshold}.")
        self.st_pairs = new_st_pairs

    def treshhold_by_pixel_value(
        self, threshold: float = 1500
        ) -> None:
        """
        Threshold the dataset by pixel value.
        This method filters the dataset to only include source-target pairs
        where the source image has a maximum pixel value smaller than the given threshold.

        :param threshold: The pixel value threshold to filter by.
        :type threshold: float
        """
        validate_type(threshold, (int, float))
        
        new_st_pairs = []
        for pair in tqdm(self.st_pairs):
            input_data = load_fits_data(pair[0], max_val=True)
            if input_data < threshold:
                new_st_pairs.append(pair)

        print_box(f"Filtered dataset to {len(new_st_pairs)} pairs with pixel value threshold < {threshold}.")
        
        self.st_pairs = new_st_pairs

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
            return input_tensor, target_tensor
        else:
            target_tensor, _ = self._process_data(target_data, transform=False)
            return input_tensor, target_tensor, input_norm_params

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

        real_image_tensor, _ = self._process_data(real_image_data, transform=True)

        # Compute the psf
        psf_tensor = source_tensor - target_tensor

        return real_image_tensor, psf_tensor

        # Offset psf_tensor halfway to the top left corner
        H, W = psf_tensor.shape[-2:]
        shift_y = -H // 4
        shift_x = -W // 4
        psf_tensor = torch.roll(psf_tensor, shifts=(shift_y, shift_x), dims=(-2, -1))
        
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
