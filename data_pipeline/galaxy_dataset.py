import re

import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.io import fits

from data_pipeline.getter import TELESCOPES_DB


class GalaxyDataset(Dataset):
    def __init__(
        self,
        source: list[str],
        target: list[str]
    ):
        """
        Initialize the GalaxyDataset class.
        This class is responsible for loading the galaxy AGN and AGN-free images
        from the given file groups. It creates source-target pairs for the dataset.
        The source is the AGN image and the target is the AGN-free image.
        It also creates the target difference image (AGN only) by
        subtracting the AGN-free image from the AGN image.

        :param source: The list of AGN image file paths.
        :type source: list[str]
        :param target: The list of AGN-free image file paths.
        :type target: list[str]
        """
        self.st_pairs = list(zip(source, target))

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        :return: The number of source-target pairs in the dataset.
        :rtype: int
        """
        return len(self.st_pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the source-target pair at the given index.
        The source is the AGN image and the target is the AGN-free image.
        It also creates the target difference image (AGN only) by
        subtracting the AGN-free image from the AGN image.

        :param idx: The index of the source-target pair.
        :type idx: int
        :raises RuntimeError: If there is an error loading the files.
        :return: A tuple containing the source (AGN image), target (AGN-free image),
                 and target difference image (AGN only).
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        galaxy_agn_filepath, galaxy_agn_free_filepath = self.st_pairs[idx]

        try:
            # Load the AGN file
            with fits.open(galaxy_agn_filepath) as hdul:
                galaxy_agn_data = hdul[0].data

            # Load the AGN-free file
            with fits.open(galaxy_agn_free_filepath) as hdul:
                galaxy_agn_free_data = hdul[0].data

            # Convert to 2D arrays if the AGN free image is 3D
            if len(galaxy_agn_free_data.shape) == 3:
                galaxy_agn_free_data = galaxy_agn_free_data[0]

            # Convert the data to native-endian format before creating a tensor
            galaxy_agn_data = galaxy_agn_data.astype(np.float32, copy=False)
            galaxy_agn_free_data = galaxy_agn_free_data.astype(np.float32, copy=False)

            # Convert the data to torch tensors
            galaxy_agn_tensor = torch.tensor(galaxy_agn_data, dtype=torch.float32).unsqueeze(0)
            galaxy_agn_free_tensor = torch.tensor(galaxy_agn_free_data, dtype=torch.float32).unsqueeze(0)
            target_diff_image = galaxy_agn_tensor - galaxy_agn_free_tensor

            target_diff_image = target_diff_image.squeeze(0).unsqueeze(0)

            return galaxy_agn_tensor, galaxy_agn_free_tensor, target_diff_image
        
        except Exception as e:
            # Handle corrupted files
            raise RuntimeError(f"Error loading files: {galaxy_agn_filepath}, {galaxy_agn_free_filepath}. {e}")
    