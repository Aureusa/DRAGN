"""
NOTE FOR USERS:

This module provides the `FitsLoader` class, a custom DataLoader
for AGN/galaxy FITS datasets.
It is designed to work with datasets that inherit from the
`_BaseDataset` class in `galaxy_dataset.py`.

**Key Points:**
- `FitsLoader` extends PyTorch's `DataLoader` and is tailored for datasets
following the FITS file naming conventions described in the data database (`_telescopes_db.py`).
- The loader expects datasets to provide tuples of (image, target, stats) and uses
a custom collate function when the dataset is not in training mode.
- The collate function is designed to handle both tensor data and additional metadata
(used for the inverse transformation of the input data) in each batch.
- This loader is not guaranteed to work with arbitrary datasets or file formats.

**Important:**
- Only use `FitsLoader` with datasets that inherit from `_BaseDataset`.
- If you wish to use your own data or dataset class, ensure it is compatible with the
expected input/output structure or modify the loader accordingly.
- For more information about the data structure, refer to the documentation or
contact the maintainers.
"""
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader

from data_pipeline.galaxy_dataset import _BaseDataset
from data_pipeline.transforms import _BaseTransform
from utils_utils.validation import validate_type


class _BaseLoader(DataLoader):
    """
    Base class for custom DataLoaders in the Deep-AGN-Clean project.
    
    This class is intended to be extended by specific DataLoader implementations
    that handle different types of datasets, such as FITS files or other formats.
    It provides a common interface and structure for custom DataLoaders.
    """
    def __init__(self, dataset, **kwargs):
        """
        Initialize the base DataLoader with the given dataset.
        
        :param dataset: The dataset to load.
        :type dataset: _BaseDataset
        :param args: Additional positional arguments for DataLoader.
        :param kwargs: Additional keyword arguments for DataLoader.
        """
        super().__init__(dataset=dataset, **kwargs)
        validate_type(dataset, _BaseDataset)

    def __str__(self) -> str:
        """
        Return a string representation of the BaseLoader.
        
        :return: String representation of the BaseLoader.
        :rtype: str
        """
        info = "Loader Information:\n"
        info = f"Batch size: {self.batch_size}\n"
        # Check if using RandomSampler (shuffle=True) or SequentialSampler (shuffle=False)
        from torch.utils.data import RandomSampler
        info += f"Shuffle: {isinstance(self.sampler, RandomSampler)}\n"
        info += f"Number of workers: {self.num_workers}\n"
        info += f"Prefetch factor: {self.prefetch_factor}\n"
        info += f"Dataset transform: {self.dataset.transform}\n"
        info += f"Training mode: {self.dataset.training}\n"
        return info

    def get_kwargs(self) -> str:
        """
        Get the keyword arguments used to initialize the DataLoader.
        This method returns a dictionary containing the parameters
        used to create the DataLoader instance, such as batch size,
        shuffle status, number of workers, and prefetch factor.
        """
        batch_size = self.batch_size
        from torch.utils.data import RandomSampler
        shuffle = isinstance(self.sampler, RandomSampler)
        num_workers = self.num_workers
        prefetch_factor = self.prefetch_factor
    
        kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor
        }
        return kwargs



class FitsLoader(_BaseLoader):
    """
    Custom DataLoader that uses a custom collate function to handle
    different data types in the dataset.

    It is effectively a wrapper around PyTorch's DataLoader
    and is designed to work with datasets that inherit from
    `_BaseDataset` in `galaxy_dataset.py`.
    """
    def __init__(
            self,
            dataset: _BaseDataset,
            batch_size: int,
            shuffle: bool = False,
            num_workers: int = 0,
            prefetch_factor: int|None = None,
            **kwargs
        ):
        """
        Initialize the FitsLoader.

        :param dataset: The dataset to load.
        :type dataset: _BaseDataset
        :param batch_size: Number of samples per batch.
        :type batch_size: int
        :param shuffle: Whether to shuffle the dataset.
        :type shuffle: bool
        :param num_workers: Number of subprocesses to use for data loading.
        :type num_workers: int
        :param prefetch_factor: Number of batches to prefetch.
        :type prefetch_factor: int|None
        :param args: Additional positional arguments, 
        check torch.utils.data.DataLoader documentation.
        :param kwargs: Additional keyword arguments, 
        check torch.utils.data.DataLoader documentation.
        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            **kwargs
        )
        if dataset.training is False:
            self.collate_fn = self._custom_collate

    def _custom_collate(self, batch: list[tuple]) -> tuple:
        """
        Custom collate function to handle batches of data.
        This function is used when the dataset is not in training mode.
        It expects each item in the batch to be a tuple of the form
        (image, target, stats), where:
        - `image` is a tensor representing the image data.
        - `target` is a tensor representing the target data.
        - `stats` is an instance of a custom class containing metadata
          for the inverse transformation of the input data.
        If `stats` is None, it will be ignored.
        
        :param batch: A list of tuples, where each tuple contains
                      (image, target, stats).
        :type batch: list[tuple]
        :return: A tuple containing:
                 - Stacked images tensor.
                 - Stacked targets tensor.
                 - List of stats instances (or None if stats is not provided).
        :rtype: tuple[torch.Tensor, torch.Tensor, list|None]
        """
        # Unzip the batch: each is a tuple (img, stats)
        images, targets, stats = zip(*batch)

        # Use default_collate to stack tensors
        images = default_collate(images)
        targets = default_collate(targets)
        stats = self._collate_normalization_params(stats)

        return images, targets, stats
        
    def _collate_normalization_params(self, stats_list):
        if any(s is None for s in stats_list):
            return None

        # Get all parameter keys from the first instance
        keys = stats_list[0].params.keys()
        stacked_params = {}

        # Get the type of the first stats instance
        stats_type = type(stats_list[0])

        for key in keys:
            # Extract and squeeze the leading batch dim if present
            tensors = [s[key].squeeze(0) if s[key].shape[0] == 1 else s[key] for s in stats_list]
            # Stack along batch dimension
            stacked_params[key] = torch.stack(tensors, dim=0)  # (B, C, H, W)

        return stats_type(**stacked_params)
        