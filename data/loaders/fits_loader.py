import torch

from .base_loader import _BaseLoader
from core.registry import LOADERS_REGISTRY
from ..datasets import _BaseDataset


@LOADERS_REGISTRY.register("fits_loader")
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
        self.collate_fn = self._custom_collate_fn

    def _custom_collate_fn(self, batch: list[tuple]) -> tuple:
        return {
            "input": torch.stack([item["input"] for item in batch]),
            "target": torch.stack([item["target"] for item in batch]),
            "input_norm_params": [item["input_norm_params"] for item in batch],
            "target_norm_params": [item["target_norm_params"] for item in batch]
        }
