import os
import inspect
import torch
from tqdm import tqdm

from data_pipeline import _BaseLoader
from networks.models import AVALAIBLE_MODELS, BaseModel
from model_training.loss_functions import get_loss_function
from utils import (
    print_box,
    get_device
)
from loggers_utils import log_execution
from utils.validation import validate_type
from loggers_utils import TrainingLogger
from model_testing import Tester # DEPRICATED
from model_testing.plotter import Plotter # DEPRICATED


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
        
        self.model: BaseModel = AVALAIBLE_MODELS[model_type](**kwargs)

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

    @property
    def model_filename(self) -> str:
        """
        Get the model filename.
        
        :return: The model filename.
        :rtype: str
        """
        return self._model_filename
    
    @model_filename.setter
    def model_filename(self, value: str) -> None:
        """
        Set the model filename.
        
        :param value: The new model filename.
        :type value: str
        """
        validate_type(value, str, obj_name="model_filename")
        self._model_filename = value

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
            real_container = None, # DEPRICATED
            adverserial_logger: bool = False,
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

        # Initialize loggers
        logger = TrainingLogger(
            save_dir=self.data_folder,
            adverserial_logger=adverserial_logger,  # Set to False as this is not an adversarial training (DEPRICATED)
        )

        # Initialize the best validation loss
        best_val_loss = float('inf')

        # Load the best validation loss if in logger
        if logger.get_best_val_loss() < best_val_loss:
            best_val_loss = logger.get_best_val_loss()

        # Define the device
        device = get_device()
        self.model.to(device)

        print_box(f"Training on {device}!")

        # Define the optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Load optimzer if in logger
        optimizer_state = logger.get_optimizer_state()
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            print_box("Optimizer state loaded successfully!")

        # Iterate over epochs
        for epoch in tqdm(
            range(num_epochs),
            desc='Epochs left...'
        ):
            current_epoch = logger.get_current_epoch()

            train_loss = self.model.train_model(
                self._train_loader,
                loss_function,
                optimizer,
                device,
            )

            val_loss = self.model.validate_model(
                self._val_loader,
                loss_function,
                device,
            )

            if logger.check_best_val_loss(val_loss):
                best_val_loss = val_loss
                self.model.save_model(f"{self.model_filename}_best_model", self.data_folder)
                info = f"Best model saved with validation loss: {best_val_loss:.4f}"
                print_box(info)

            self.model.save_model(f"{self.model_filename}_epoch_{current_epoch}", self.data_folder)
            info = f"Checkpoint model saved!"
            print_box(info)

            logger.log_epoch(
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                optimizer=optimizer
            )

            # DEPRICATED
            self._test_on_real_data(
                current_epoch=current_epoch,
                real_container=real_container,
                device=device,
            )

    # This method is DEPRICATED and will be removed in future versions.
    def _test_on_real_data(
            self,
            current_epoch: int,
            real_container,
            device,
        ):
        # Get batched data from the real_container
        data = []
        for i in range(len(real_container)):
            real_data = real_container[i]
            data.append(real_data)
        
        # Stack the data
        data = torch.stack(data, dim=0) # (B, C, H, W)
        data_np = data.cpu().numpy()

        # Move the data to the device
        data = data.to(device)

        self.model.eval()
        with torch.no_grad():
            # Get the predictions
            predictions = self.model(data)
            predictions = predictions.unsqueeze(0)

        # Convert predictions to numpy
        predictions = predictions.cpu().numpy()

        # Create the folder for real data predictions if it doesn't exist
        real_data_folder = os.path.join(self.data_folder, "real_data_predictions")
        if not os.path.exists(real_data_folder):
            os.makedirs(real_data_folder)

        data_folder = os.path.join(real_data_folder, f"epoch_{current_epoch}")
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        Plotter().grid_plot(
            sources=[data_np],
            targets=None,
            outputs=predictions,
            titles=[self._model_type],
            filename=f"epoch_{current_epoch}_real_data",
            data_folder=data_folder,
            f_agn=None,
            save=True
        )

        # Clean up memory
        del data, predictions, data_np
        torch.cuda.empty_cache()

    # DEPRICATED: This method is deprecated and will be removed in future versions.    

    # @log_execution("Training started...", "Model trained successfully!")
    # def train_model(
    #         self,
    #         loss_name: str,
    #         lr: float = 0.001,
    #         num_epochs: int = 50,
    #     ):
    #     """
    #     Train the model with the specified loss function and parameters.

    #     :param loss_name: The name of the loss function to use for training.
    #     Should be one of the available loss functions in model_training.loss_functions.
    #     You can check the available loss functions by running:
    #             ```
    #             from model_training import AVALIABLE_LOSS_FUNCTIONS

    #             print(AVALIABLE_LOSS_FUNCTIONS)
    #             ```
    #     :type loss_name: str
    #     :param lr: The learning rate for the optimizer.
    #     :type lr: float
    #     :param num_epochs: The number of epochs to train the model.
    #     :type num_epochs: int
    #     """
    #     validate_type(loss_name, str, "loss_name")

    #     # Get the loss function
    #     loss_function = get_loss_function(loss_name)

    #     # Information
    #     info = f"Training `{self._model_type}` model with `{loss_name}` loss function."
    #     info += f"\nModel name: {self._model_filename}"
    #     info += f"\nData folder: {self._data_folder}"
    #     info += f"\nBatch size: {self._train_loader.batch_size}"
    #     info += f"\nNumber of workers: {self._train_loader.num_workers}"
    #     info += f"\nTraining data size: {len(self._train_loader.dataset)}"
    #     info += f"\nValidation data size: {len(self._val_loader.dataset)}"
    #     print_box(info)

    #     self.model.train_model(
    #         train_loader=self._train_loader,
    #         val_loader=self._val_loader,
    #         lr=lr,
    #         loss_function=loss_function,
    #         num_epochs=num_epochs,
    #         model_filename=self._model_filename,
    #         data_path=self._data_folder,
    #     )
