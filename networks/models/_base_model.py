from abc import abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn
import os

from core.component import Component
from utils import print_box
from utils.device import get_device


class BaseModel(nn.Module, Component):
    """
    Base for PyTorch models with config support
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x):
        """
        PyTorch forward pass - this is what should be abstract
        """
        pass
    
    @classmethod  
    @abstractmethod
    def from_config(cls, config: Dict[str, Any], **kwargs):
        """
        Config construction - this should be abstract
        """
        pass
    
    # Utility methods (concrete, not abstract)
    def save_checkpoint(self, filename: str, dir_: str):
        """
        Save the model to a file.
        
        :param name: Name of the file to save the model to.
        :type name: str
        """
        if filename.endswith(".pth"):
            filename = filename[:-4]

        path = os.path.join(dir_, f"{filename}.pth")

        torch.save(self.state_dict(), path)

        info = f"Model `{filename}.pth` saved successfully!\n"
        info += f"Path to model: {dir_}"
        print_box(info)

    def load_checkpoint(self, filename: str, dir_: str):
        """
        Load the model from a file.
        
        :param filename: Name of the file to load the model from.
        :type name: str
        :param dir_: Directory to load the model from.
        :type dir_: str
        """
        if filename.endswith(".pth"):
            filename = filename[:-4]
            
        self.load_state_dict(torch.load(os.path.join(dir_, f"{filename}.pth"), map_location=get_device()))

        info = f"Model `{filename}` loaded successfully!"
        print_box(info)
