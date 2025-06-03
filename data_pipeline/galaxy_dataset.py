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
from astropy.io import fits

from data_pipeline.transforms import NormalizationParams, _BaseTransform
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
    
    def filter_by_f_agn_list(self, f_agn_list: list[int]) -> None:
        """
        Filter the dataset by a list of AGN fractions.

        :param f_agn_list: The list of AGN fractions to filter by.
        :type f_agn_list: list[int]
        """
        validate_list(f_agn_list, int)
        
        # DEPRICATED: This method is deprecated and will be removed in future versions.
        f_agn_list = [10, 30, 44, 65, 70, 90]
        new_st_pairs = []
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

    def _load_fits(self, filepath: str) -> np.ndarray:
        """
        Load a FITS file and return the data as a numpy array.

        :param filepath: The path to the FITS file.
        :type filepath: str
        :raises RuntimeError: If there is an error loading the FITS file.
        :return: The data from the FITS file.
        :rtype: np.ndarray
        """
        try:
            with fits.open(filepath) as hdul:
                data = hdul[0].data
            
            validate_numpy_array(data, ndim=(2,3))
            return data
        except Exception as e:
            raise RuntimeError(f"Error loading FITS file {filepath}: {e}")
    

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
        input_data = self._load_fits(input_filepath)
        target_data = self._load_fits(target_filepath)

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

        if transform and self.transform is not None:
            # Normalize the images
            data, data_norm_param = self.transform(data)

        # Convert the data to torch tensors
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return data, data_norm_param
    