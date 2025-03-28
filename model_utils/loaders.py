import torch
import numpy as np
from abc import ABC, abstractmethod

# TODO:
# 1. Define the dataset class
# 2. Adjust the loader factory class accordingly
class LoaderFactory:
    def __init__(self, loader_name: str):
        self.loaders = {
            'AGNDataSet': AGNDataSet,
        }

        if loader_name not in self.loaders:
            raise ValueError(f"Loader '{loader_name}' is not recognized.")
        
        # Set the loader based on the provided name
        self.loader = self.loaders[loader_name]

    def get_loader(self, inputs, targets, batch_size, num_workers, shuffle):
        loader = self.loader(inputs, targets)
        
        data_loader = torch.utils.data.DataLoader(
            loader,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

        return data_loader
    

class AbstractDataSet(ABC):
    @abstractmethod
    def __init__(self, inputs, targets) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class AGNDataSet(AbstractDataSet):
    def __init__(self, inputs: list[np.ndarray], targets: list[np.ndarray]) -> None:
        """
        Given a list of inputs and targets, this class will convert them to PyTorch tensors.

        :param inputs: input images
        :type inputs: list[np.ndarray]
        :param targets: target images
        :type targets: list[np.ndarray]
        """
        inputs = np.array(inputs)
        targets = np.array(targets)
        self.inputs, self.targets = self._convert_to_tensor(inputs, targets)
        print("Inputs shape: ", self.inputs.shape)
        print("Targets shape: ", self.targets.shape)

    def _convert_to_tensor(self, inputs: np.ndarray, targets: np.ndarray) -> tuple[torch.Tensor,torch.Tensor]:
        """
        Convert the inputs and targets to PyTorch tensors.

        :param inputs: input images
        :type inputs: np.ndarray
        :param targets: target images
        :type targets: np.ndarray
        :return: converted inputs and targets
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        # Convert the inputs and targets to PyTorch tensors
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        return inputs, targets

    def __len__(self) -> int:
        """
        Return the number of input images.

        :return: the number of input images
        :rtype: int
        """
        return len(self.inputs)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the input and target image.

        :param index: the index of the input and target image
        :type index: int
        :return: the input and target image
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        return self.inputs[index], self.targets[index]
    