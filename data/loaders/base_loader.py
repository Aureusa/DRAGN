from torch.utils.data import DataLoader

from ..datasets.base_dataset import _BaseDataset
from utils.validation import validate_type


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
