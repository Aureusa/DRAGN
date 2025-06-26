from abc import ABC, abstractmethod
import torch
import os

from utils import print_box
from utils_utils.device import get_device


class BaseModel(ABC):
    """
    Base class for all models.
    This class defines the basic structure and methods
    that all models should implement.
    """
    @abstractmethod
    def train_model(self):
        """
        Train the model.
        """
        pass

    @abstractmethod
    def save_model(self, filename: str,  dir_: str):
        """
        Save the model to a file.
        
        :param name: Name of the file to save the model to.
        :type name: str
        """
        path = os.path.join(dir_, f"{filename}.pth")

        torch.save(self.state_dict(), path)

        info = f"Model `{filename}.pth` saved successfully!\n"
        info += f"Path to model: {dir_}"
        print_box(info)

    @abstractmethod
    def load_model(self, filename: str,  dir_: str):
        """
        Load the model from a file.
        
        :param filename: Name of the file to load the model from.
        :type name: str
        :param dir_: Directory to load the model from.
        :type dir_: str
        """
        self.load_state_dict(torch.load(os.path.join(dir_, f"{filename}.pth"), map_location=get_device()))

        info = f"Model `{filename}` loaded successfully!"
        print_box(info)
