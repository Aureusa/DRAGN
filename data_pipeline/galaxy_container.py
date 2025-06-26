from data_pipeline.utils import load_fits_data, center_crop
import numpy as np
import torch
from utils import print_box
import re
import random


class GalaxyContainer:
    def __init__(
            self,
            filepaths: list[str],
        ):
        self.filepaths = filepaths

    def __len__(self) -> int:
        return len(self.filepaths)
    
    def filter_by_f_agn_list(self, f_agn_list: list[int]|int, n: int = float("inf")) -> None: # New name: filter_by_f_agn 
        """
        Filter the dataset by a list of AGN fractions.

        :param f_agn_list: The list of int or int of AGN fractions
        to filter by.
        :type f_agn_list: list[int]
        """
        # Not Depricated
        filtered_filepaths = []
        '''
        # DEPRICATED:
        f_agn_list = [10, 30, 44, 65, 70, 90]
        '''

        # Newer Version of the method
        for f_agn in f_agn_list:
            pattern = re.compile(rf"_f{f_agn}")
            count = 0
            for file in self.filepaths:
                if pattern.search(file):
                    filtered_filepaths.append(file)
                    count += 1
                    # if len(filtered_filepaths) == 136: # DEPRICATED: Needs to be removed in the future
                    #     self.filepaths = filtered_filepaths # DEPRICATED: Needs to be removed in the future
                    #     print_box(f"Filtered dataset to {len(self.filepaths)} pairs with AGN fractions: {f_agn_list}.")
                    #     return # DEPRICATED: Needs to be removed in the future
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
        self.filepaths = filtered_filepaths
        print_box(f"Filtered dataset to {len(self.filepaths)} pairs with AGN fractions: {f_agn_list}.")

    def __getitem__(self, idx: int) -> tuple:
        idx = random.randint(0, len(self.filepaths) - 1)
        
        # Load the image
        filepath = self.filepaths[idx]
        image = load_fits_data(filepath)

        # Preprocess the data
        image_tensor = self._process_data(image)
        
        return image_tensor
        
    def _process_data(self, data: np.ndarray):
        # Convert to 2D arrays if the AGN free image is 3D
        if len(data.shape) == 3:
            data = data[0]

        # Convert the data to native-endian format before creating a tensor
        data = data.astype(np.float32, copy=False)

        # Crop the data to 128x128 pixels
        data = center_crop(data, 128, 128)

        # Convert the data to torch tensors
        #data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return data
    