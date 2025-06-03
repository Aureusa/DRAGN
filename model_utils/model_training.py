from torch.utils.data import DataLoader
import os
import re
import random


from data_pipeline import (
    FilepathGetter,
    create_source_target_pairs,
    test_train_val_split,
    _BaseDataset,
    _BaseTransform
)
from model import AVALAIBLE_MODELS
from model_utils.loss_functions import get_loss_function
from loggers_utils import log_execution
from utils import (
    load_pkl_file,
    print_box
)


class ModelTrainer:
    def __init__(
            self,
            model_type: str,
            model_name: str,
            telescope: str,
            data_folder: str,
            **kwargs
        ) -> None:
        """
        Initialize the ModelTrainer class.
        
        :param model_type: The type of model to train.
        :type model_type: str
        :param model_name: The name of the model to train.
        :type model_name: str
        :param telescope: The telescope to get the data from.
        :type telescope: str
        :param data_folder: The folder to save data such as
        model weights, train and val loss.
        :type data_folder: str
        :param kwargs: Additional arguments to pass to the model.
        :type kwargs: dict
        """
        if model_type not in AVALAIBLE_MODELS:
            raise ValueError(f"Model type {model_type} is not available. Choose from {list(AVALAIBLE_MODELS.keys())}.")
        else:
            if telescope == "Euclid" and re.search("GAN", model_type) is not None:
                self.model = AVALAIBLE_MODELS[model_type](discriminator_in_shape=(2, 40, 40), **kwargs)
            else:
                self.model = AVALAIBLE_MODELS[model_type](**kwargs)

        self._model_type = model_type
        self._model_name = model_name
        self._telescope = telescope
        self._data_folder = os.path.join("data", data_folder)

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
        self._data_folder = value
        if not os.path.exists(self._data_folder):
            os.makedirs(self._data_folder)
    
    @log_execution("Loading data...", "Data loaded successfully!")
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
        if folder is None: # Retrieve the data from the default path
            if self._telescope == "JWST":
                folder = "jwst_full_data"
            elif self._telescope == "Euclid":
                folder = "euclid_full_data"
            else:
                raise ValueError(f"Something went wrong while retrieving the data from the basepath for {self._telescope}.")
        
        X_train, y_train = load_pkl_file(os.path.join("data", folder, f"train_data.pkl"))
        X_val, y_val = load_pkl_file(os.path.join("data", folder, f"val_data.pkl"))
        X_test, y_test = load_pkl_file(os.path.join("data", folder, f"test_data.pkl"))

        print_box(f"Loaded data from `{os.path.join('data', folder)}`.")

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @log_execution("Forging data...", "Data forged successfully!")
    def get_data(
            self,
            redshift: list[str]|None = None,
            redshift_treshhold: float | None = None,
            sanity_check: bool = False,
        ) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
        """
        Get the data from the telescope used to initialise this object.
        
        :param redshift: The redshift to filter the data. Optionally.
        :type redshift: list[str]|None
        :param redshift_treshhold: The redshift treshhold to filter the data.
        It retrieves the data with redshift < redshift_treshhold. Optionally.
        :type redshift_treshhold: float|None
        :param sanity_check: If True, it will print 3 random elements of the data.
        :type sanity_check: bool
        :return: The training, validation and test data.
        :rtype: tuple[list[str], list[str], list[str], list[str], list[str], list[str]]
        """
        data_getter = FilepathGetter(self._telescope, redshift, redshift_treshhold)

        # Initialize the DataGetter class
        files, _ = data_getter.get_data()

        source, target = create_source_target_pairs(files)
        
        if sanity_check:
            for i in range(3):
                indx = random.randint(0, len(source) - 1)
                info = f"Sanity check {i+1}"
                info += f"\nSource: {source[indx]}"
                info += f"\nTarget: {target[indx]}"
                print_box(info)

        X_train, X_val, X_test, y_train, y_val, y_test = test_train_val_split(
            source, target, test_size=0.2, val_size=0.1
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @log_execution("Forging loaders...", "Loaders forged successfully!")
    def forge_loaders(
            self,
            X_train: list[str],
            X_val: list[str],
            y_train: list[str],
            y_val: list[str],
            dataset: _BaseDataset,
            loader: DataLoader,
            batch_size: int,
            tramsform: _BaseTransform|None = None,
            shuffle: bool = True,
            num_workers: int = 0,
            prefetch_factor: int|None = None,
            **kwargs
        ) -> tuple[DataLoader, DataLoader]:
        train_dataset = dataset(
            X_train,
            y_train,
            transform=tramsform,
            training=True
        )
        val_dataset = dataset(
            X_val,
            y_val,
            transform=tramsform,
            training=True
        )

        train_loader = loader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            **kwargs
        )

        val_loader = loader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            **kwargs
        )
        return train_loader, val_loader

    def fine_tune_model(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            loss_name: str,
            lr: float = 0.001,
            num_epochs: int = 50,
            **kwargs
        ):
        self.model.load_model(dir_ = self._data_folder, **kwargs)
        self.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            loss_name=loss_name,
            lr=lr,
            num_epochs=num_epochs,
        )

    @log_execution("Transfering weights between models...", "Transfering weights completed!")
    def transfer_learning(self, **kwargs):
        other_model = AVALAIBLE_MODELS[self._model_type]()
        other_model.load_model(**kwargs)

        other_dict = other_model.get_state_dict()

        self.model.load_from_state_dict(other_dict)

    @log_execution("Training initialised", "Model trained successfully!")
    def train_model(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            loss_name: str,
            lr: float = 0.001,
            num_epochs: int = 50,
        ):
        if not os.path.exists(self._data_folder):
            os.makedirs(self._data_folder)

        # Get the loss function
        loss_function = get_loss_function(loss_name)

        # Information
        info = f"Training `{self._model_type}` model with `{loss_name}` loss function."
        info += f"\nModel name: {self._model_name}"
        info += f"\nTelescope: {self._telescope}"
        info += f"\nData folder: {self._data_folder}"
        info += f"\nBatch size: {train_loader.batch_size}"
        info += f"\nNumber of workers: {train_loader.num_workers}"
        info += f"\nTraining data size: {len(train_loader.dataset)}"
        info += f"\nValidation data size: {len(val_loader.dataset)}"
        print_box(info)

        self.model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            lr=lr,
            loss_function=loss_function,
            num_epochs=num_epochs,
            model_name=self._model_name,
            data_path=self._data_folder,
        )

        print_box("Training finished!")

        # Save the model
        self.model.save_model(self._model_name, self._data_folder)
        
        # Save the training and validation loss
        self.model.save_train_val_loss(self._data_folder)
