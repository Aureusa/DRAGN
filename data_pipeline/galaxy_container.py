from data_pipeline.utils import load_fits_data, center_crop
import numpy as np
import torch
from utils import print_box
from utils.validation import validate_list
import re
import random


class GalaxyContainer:
    def __init__(
            self,
            filepaths: list[str],
            condition_on_f_agn: bool = False,
            condition_on_psf: bool = False
        ):
        validate_list(filepaths, str, obj_name="filepaths")
        self.filepaths = filepaths
        self._condition_on_f_agn = condition_on_f_agn
        self._condition_on_psf = condition_on_psf

    def __len__(self) -> int:
        return len(self.filepaths)

    def visualize(self, idx: int, filename: str) -> None:
        """
        Visualize the image at the given index.
        This method is used for debugging purposes.
        """
        import matplotlib.pyplot as plt
        from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch

        image = self[idx]
        image = image.squeeze().numpy()

        # Normalize the image
        norm = ImageNormalize(data=image, stretch=AsinhStretch(), interval=PercentileInterval(99.5))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, cmap='gray', norm=norm)
        ax.set_title(f"Image at index {idx}")
        ax.axis('off')
        plt.savefig(f"{filename}_{idx}.png")
        plt.close()

        print_box(f"Visualized image at index {idx} and saved to {filename}_{idx}.png")

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

    def __getitem__(self, idx: int) -> torch.Tensor:        
        # Load the image
        filepath = self.filepaths[idx]
        image = load_fits_data(filepath)

        # Preprocess the data
        image_tensor = self._process_data(image)

        # Implement condition on f_agn here
        # ...
        f_agn = 0.3

        if self._condition_on_f_agn:
            image_tensor = self._condition_input_tensor_on_f_agn(image_tensor, f_agn=f_agn)

        if self._condition_on_psf:
            image_tensor = self._condition_input_tensor_on_psf(image_tensor)
        
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
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return data

    def _condition_input_tensor_on_f_agn(self, input_tensor: torch.Tensor, f_agn: float) -> torch.Tensor:
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
        # Get the shape of the input tensor
        _, height, width = input_tensor.shape
        
        # Add the AGN fraction as a new channel
        f_agn_tensor = torch.full((1, height, width), f_agn, dtype=torch.float32) # (B, 1, H, W) full with f_agn value
        input_tensor = torch.cat((input_tensor, f_agn_tensor), dim=0)
        return input_tensor
    
    def _condition_input_tensor_on_psf(self, input_tensor: torch.Tensor) -> torch.Tensor:
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
    