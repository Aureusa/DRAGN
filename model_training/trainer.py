import os
import inspect

from data_pipeline import _BaseLoader
from networks.models import AVALAIBLE_MODELS
from model_training.loss_functions import get_loss_function
from utils import (
    print_box
)
from loggers_utils import log_execution
from utils_utils.validation import validate_type


class Trainer:
    def __init__(
            self,
            model_type: str,
            model_filename: str,
            data_folder: str,
            train_loader: _BaseLoader,
            val_loader: _BaseLoader,
            **kwargs
        ) -> None:
        """
        Initialize the Trainer class.
        This class is responsible for training the model with the specified parameters.
        
        :param model_type: The type of model to train. All avaliable models should be in the
        AVALAIBLE_MODELS dict in network.models.
        :type model_type: str
        :param model_filename: The filename to save the trained model.
        :type model_filename: str
        :param data_folder: The folder where the data is stored.
        :type data_folder: str
        :param train_loader: The DataLoader for the training data.
        :type train_loader: _BaseLoader
        :param val_loader: The DataLoader for the validation data.
        :type val_loader: _BaseLoader
        :param kwargs: Additional keyword arguments to pass to the model.
        :type kwargs: dict
        """
        validate_type(model_type, str, "model_type")
        validate_type(model_filename, str, "model_filename")
        validate_type(data_folder, str, "data_folder")
        validate_type(train_loader, _BaseLoader, "train_loader")
        validate_type(val_loader, _BaseLoader, "val_loader")

        if model_type not in AVALAIBLE_MODELS:
            raise ValueError(f"Model type {model_type} is not available. Choose from {list(AVALAIBLE_MODELS.keys())}.")
        
        self.model = AVALAIBLE_MODELS[model_type](**kwargs)

        self._model_type = model_type
        self._model_filename = model_filename
        self.data_folder = data_folder

        self._train_loader = train_loader
        self._val_loader = val_loader

    @property
    def data_folder(self) -> str:
        """
        Get the data folder.
        
        :return: The data folder.
        :rtype: str
        """
        return self._data_folder
    
    @data_folder.setter
    def data_folder(self, value: str) -> None:
        """
        Set the data folder.
        
        :param value: The new data folder.
        :type value: str
        """
        self._data_folder = os.path.abspath(value)
        if not os.path.exists(self._data_folder):
            os.makedirs(self._data_folder)

    def fine_tune_model(
            self,
            loss_name: str,
            lr: float = 0.001,
            num_epochs: int = 50,
            **kwargs
        ):
        """
        Fine-tune the model with the specified parameters.

        :param loss_name: The name of the loss function to use for training.
        Should be one of the available loss functions in model_training.loss_functions.
        You can check the available loss functions by running:
                ```
                from model_training import AVALIABLE_LOSS_FUNCTIONS

                print(AVALIABLE_LOSS_FUNCTIONS)
                ```
        :type loss_name: str
        :param lr: The learning rate for the optimizer.
        :type lr: float
        :param num_epochs: The number of epochs to train the model.
        :type num_epochs: int
        :param kwargs: Additional keyword arguments to pass to the model.
        Usually you need to pass `filename` and specify which model to load,
        but that depends on the model you are using as some of them have
        different way of loading the model.
        :type kwargs: dict
        """
        try:
            if kwargs.get("dir_", None) is not None:
                dir_ = kwargs.pop("dir_")
            else:
                dir_ = self._data_folder
            self.model.load_model(dir_ = dir_, **kwargs)
        except Exception as e:
            if e.__class__.__name__ == "TypeError":
                info = f"{kwargs} were not the right signature for loading the model!"
                info += f"You should use: {inspect.signature(self.model.load_model)}\n"
                info += "Keep in mind that the default `dir_` is the data_folder used"
                info += " to instantiate the Trainer class."
                info += f"\nError: {e}"
                raise RuntimeError(info)
            elif e.__class__.__name__ == "FileNotFoundError":
                info = (
                    "Model file not found in the specified directory:" +
                    f"{kwargs.get('dir_', self._data_folder)}"
                )
                info += f"\nError: {e}"
                raise RuntimeError(info)
            else:
                # If the error is not a TypeError or FileNotFoundError, re-raise it
                raise RuntimeError(f"Error: {e}")
        
        self.train_model(
            loss_name=loss_name,
            lr=lr,
            num_epochs=num_epochs,
        )

    @log_execution("Training started...", "Model trained successfully!")
    def train_model(
            self,
            loss_name: str,
            lr: float = 0.001,
            num_epochs: int = 50,
        ):
        """
        Train the model with the specified loss function and parameters.

        :param loss_name: The name of the loss function to use for training.
        Should be one of the available loss functions in model_training.loss_functions.
        You can check the available loss functions by running:
                ```
                from model_training import AVALIABLE_LOSS_FUNCTIONS

                print(AVALIABLE_LOSS_FUNCTIONS)
                ```
        :type loss_name: str
        :param lr: The learning rate for the optimizer.
        :type lr: float
        :param num_epochs: The number of epochs to train the model.
        :type num_epochs: int
        """
        validate_type(loss_name, str, "loss_name")

        # Get the loss function
        loss_function = get_loss_function(loss_name)

        # Information
        info = f"Training `{self._model_type}` model with `{loss_name}` loss function."
        info += f"\nModel name: {self._model_filename}"
        info += f"\nData folder: {self._data_folder}"
        info += f"\nBatch size: {self._train_loader.batch_size}"
        info += f"\nNumber of workers: {self._train_loader.num_workers}"
        info += f"\nTraining data size: {len(self._train_loader.dataset)}"
        info += f"\nValidation data size: {len(self._val_loader.dataset)}"
        print_box(info)

        self.model.train_model(
            train_loader=self._train_loader,
            val_loader=self._val_loader,
            lr=lr,
            loss_function=loss_function,
            num_epochs=num_epochs,
            model_filename=self._model_filename,
            data_path=self._data_folder,
        )
