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
from torch.utils.data import Dataset

from data.transforms import _BaseTransform
from utils.validation import validate_list, validate_type


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
        **kwargs
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
    