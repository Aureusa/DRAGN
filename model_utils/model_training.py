from model import AVALAIBLE_MODELS


from data_pipeline import GalaxyDataset, FilepathGetter, create_source_target_pairs, test_train_val_split
from model.attention_unet import AttentionUNET
from model.cond_GAN import cGAN
from model_utils.loss_functions import get_loss_function
from torch.utils.data import DataLoader
import pickle
import torch
import os
from utils import print_box
import matplotlib.pyplot as plt
import re
from matplotlib.colors import Normalize


class ModelTrainer:
    def __init__(
            self,
            model_type: str,
            telescope: str,
            data_folder: str,
            loss: str
        ) -> None:
        """
        Initialize the ModelTrainer class.
        
        :param model_type: The type of model to train.
        :type model_type: str
        :param telescope: The telescope to get the data from.
        :type telescope: str
        :param data_folder: The folder to save data such as
        train, val, test sets, train and val loss.
        :type data_folder: str
        :param loss: The loss function to be used for the creation
        of the model name upon saving
        Example:
            `mse`
            `constrained_mse`
            `...`
        :type loss: str
        """
        if model_type not in AVALAIBLE_MODELS:
            raise ValueError(f"Model type {model_type} is not available. Choose from {list(AVALAIBLE_MODELS.keys())}.")
        else:
            if telescope == "Euclid" and re.search("GAN", model_type) is not None:
                self.model = AVALAIBLE_MODELS[model_type](discriminator_in_shape=(2, 40, 40))
            else:
                self.model = AVALAIBLE_MODELS[model_type]()

        self._model_type = model_type
        self._telescope = telescope
        self._data_folder = data_folder
        self._loss = loss

    def load_data(
            self,
            folder: str|None = None,
        ) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
        """
        Load the data from the given folder.
        
        :param folder: The folder to load the data from.
        :type folder: str
        :return: The training, validation and test data.
        :rtype: tuple[list[str], list[str], list[str], list[str], list[str], list[str]]
        """
        if folder is None:
            if self._telescope == "JWST":
                folder = "jwst_full_data"
            elif self._telescope == "Euclid":
                folder = "euclid_full_data"
            else:
                raise ValueError(f"Something went wrong while retrieving the data from the basepath for {self._telescope}.")
        
        with open(os.path.join("data", folder, f"train_data.pkl"), "rb") as f:
            X_train, y_train = pickle.load(f)

        with open(os.path.join("data", folder, f"val_data.pkl"), "rb") as f:
            X_val, y_val = pickle.load(f)

        with open(os.path.join("data", folder, f"test_data.pkl"), "rb") as f:
            X_test, y_test = pickle.load(f)

        print_box(f"Successfully loaded data from `{os.path.join('data', folder)}`!")

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_data(
            self,
            redshift: list[str]|None = None,
            redshift_treshhold: float | None = None
        ) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
        """
        Get the data from the telescope used to initialise this object.
        
        :param redshift: The redshift to filter the data. Optionally.
        :type redshift: list[str]|None
        :param redshift_treshhold: The redshift treshhold to filter the data.
        It retrieves the data with redshift < redshift_treshhold. Optionally.
        :type redshift_treshhold: float|None
        :return: The training, validation and test data.
        :rtype: tuple[list[str], list[str], list[str], list[str], list[str], list[str]]
        """
        data_getter = FilepathGetter(self._telescope, redshift, redshift_treshhold)

        # Initialize the DataGetter class
        files, _ = data_getter.get_data()

        source, target = create_source_target_pairs(files)
        
        info = "Sanity check 1"
        info += f"\nSource: {source[0]}"
        info += f"\nTarget: {target[0]}"
        print_box(info)

        info = "Sanity check 2"
        info = f"\nSource: {source[23452]}"
        info += f"\nTarget: {target[23452]}"
        print_box(info)

        info = "Sanity check 3"
        info += f"\nSource: {source[23152]}"
        info += f"\nTarget: {target[23152]}"
        print_box(info)

        X_train, X_val, X_test, y_train, y_val, y_test = test_train_val_split(
            source, target, test_size=0.2, val_size=0.1
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
    

    def fine_tune_model(
            self,
            X_train: list[str],
            X_val: list[str],
            X_test: list[str],
            y_train: list[str],
            y_val: list[str],
            y_test: list[str],
            loss_name: str,
            batch_size: int = 32,
            num_workers: int = 8,
            prefetch_factor: int = 2,
            lr: float = 0.001,
            num_epochs: int = 50,
            checkpoints: list[int] = [25],
            wandb_project_name: str = "Deep-AGN-Clean",
            wandb_entity: str = "s4683099",
            save_datasets: bool = False,
            **kwargs
        ):
        """
        Fine-tune the model with the given data.
        
        :param X_train: The training data.
        :type X_train: list[str]
        :param X_val: The validation data.
        :type X_val: list[str]
        :param X_test: The test data.
        :type X_test: list[str]
        :param y_train: The training labels.
        :type y_train: list[str]
        :param y_val: The validation labels.    
        :type y_val: list[str]
        :param y_test: The test labels.
        :type y_test: list[str]
        :param loss_name: The loss function to be used. It has to be
        one of the available loss functions in the model. If you are not sure
        what to use, check the list of avaliable loss functions in model_utils module.
        Example:
            from model_utils import AVALIABLE_LOSS_FUNCTIONS

            print(AVALAIBLE_LOSS_FUNCTIONS)
        :type loss_name: str
        :param batch_size: The batch size to be used.
        :type batch_size: int
        :param num_workers: The number of workers to be used.
        :type num_workers: int
        :param lr: The learning rate to be used.
        :type lr: float
        :param num_epochs: The number of epochs to be used.
        :type num_epochs: int
        :param checkpoints: The checkpoints to be used.
        :type checkpoints: list[int]
        :param wandb_project_name: The wandb project name to be used.
        :type wandb_project_name: str
        :param wandb_entity: The wandb entity to be used.
        :type wandb_entity: str
        :param save_datasets: Whether to save the datasets or not.
        :type save_datasets: bool
        :param kwargs: The keyword arguments to be passed to the model's loading method.
        :type kwargs: dict"""
        self.model.load_model(**kwargs)
        self.train_model(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            loss_name,
            batch_size,
            num_workers,
            prefetch_factor,
            lr,
            num_epochs,
            checkpoints,
            wandb_project_name,
            wandb_entity,
            save_datasets
        )

    def train_model(
            self,
            X_train: list[str],
            X_val: list[str],
            X_test: list[str],
            y_train: list[str],
            y_val: list[str],
            y_test: list[str],
            loss_name: str,
            batch_size: int = 32,
            num_workers: int = 8,
            prefetch_factor: int = 2,
            lr: float = 0.001,
            num_epochs: int = 50,
            checkpoints: list[int] = [25],
            wandb_project_name: str = "Deep-AGN-Clean",
            wandb_entity: str = "s4683099",
            save_datasets: bool = False
        ):
        """
        Train the model with the given data.
        
        :param X_train: The training data.
        :type X_train: list[str]
        :param X_val: The validation data.
        :type X_val: list[str]
        :param X_test: The test data.
        :type X_test: list[str]
        :param y_train: The training labels.
        :type y_train: list[str]
        :param y_val: The validation labels.
        :type y_val: list[str]
        :param y_test: The test labels.
        :type y_test: list[str]
        :param loss_name: The loss function to be used. It has to be
        one of the available loss functions in the model. If you are not sure
        what to use, check the list of avaliable loss functions in model_utils module.
        Example:
            from model_utils import AVALIABLE_LOSS_FUNCTIONS

            print(AVALAIBLE_LOSS_FUNCTIONS)
        :type loss_name: str
        :param batch_size: The batch size to be used.
        :type batch_size: int
        :param num_workers: The number of workers to be used.
        :type num_workers: int
        :param lr: The learning rate to be used.
        :type lr: float
        :param num_epochs: The number of epochs to be used.
        :type num_epochs: int
        :param checkpoints: The checkpoints to be used.
        :type checkpoints: list[int]
        :param wandb_project_name: The wandb project name to be used.
        :type wandb_project_name: str
        :param wandb_entity: The wandb entity to be used.
        :type wandb_entity: str
        """
        # Create the folder to store the data
        data_path = os.path.join("data", self._data_folder)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        if save_datasets:
            self._save_datasets(X_train, y_train, X_val, y_val, X_test, y_test, data_path)

        # Define the model name
        model_name = f"{self._model_type}_{self._telescope}_{self._loss}"
        model_name = model_name.lower()

        # Create the dataset
        train_dataset = GalaxyDataset(X_train, y_train)
        val_dataset = GalaxyDataset(X_val, y_val)

        # Create DataLoaders
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=prefetch_factor
        )
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=prefetch_factor
        )

        # Get the loss function
        loss_function = get_loss_function(loss_name)

        # Information
        info = f"Training `{self._model_type}` model with `{loss_name}` loss function."
        info += f"\nModel name: {model_name}"
        info += f"\nTelescope: {self._telescope}"
        info += f"\nData folder: {self._data_folder}"
        info += f"\nBatch size: {batch_size}"
        info += f"\nNumber of workers: {num_workers}"
        info += f"\nTraining data size: {len(X_train)}"
        info += f"\nValidation data size: {len(X_val)}"
        info += f"\nTest data size: {len(X_test)}"
        print_box(info)

        self.model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            lr=lr,
            loss_function=loss_function,
            num_epochs=num_epochs,
            checkpoints=checkpoints,
            wandb_project_name=wandb_project_name,
            wandb_entity=wandb_entity
        )

        print_box("Training finished!")

        # Save the model
        self.model.save_model(model_name)
        
        # Save the training and validation loss
        self.model.save_train_val_loss(data_path)

    def _save_datasets(
            self,
            X_train: list[str],
            y_train: list[str],
            X_val: list[str],
            y_val: list[str],
            X_test: list[str],
            y_test: list[str],
            data_path: str
        ):
        """
        Save the data to a pickle file.
        
        :param X_train: The training data.
        :type X_train: list[str]
        :param y_train: The training labels.
        :type y_train: list[str]
        :param X_val: The validation data.
        :type X_val: list[str]
        :param y_val: The validation labels.
        :type y_val: list[str]
        :param X_test: The test data.
        :type X_test: list[str]
        :param y_test: The test labels.
        :type y_test: list[str]
        """
        with open(os.path.join(data_path, f"train_data.pkl"), "wb") as train_file:
            pickle.dump((X_train, y_train), train_file)

        with open(os.path.join(data_path, f"val_data.pkl"), "wb") as val_file:
            pickle.dump((X_val, y_val), val_file)

        with open(os.path.join(data_path, f"test_data.pkl"), "wb") as test_file:
            pickle.dump((X_test, y_test), test_file)
        info = f"Data saved successfully in `{data_path}`!"
        print_box(info)
        return info
    