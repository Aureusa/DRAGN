import torch
import torch.nn as nn
from monai.networks.nets import BasicUNet

import os
from tqdm import tqdm
from copy import deepcopy
import pickle


from networks.models._base_model import BaseModel
from loggers_utils import TrainingLogger
from utils import print_box
from utils_utils.device import get_device


# TODO: Impliment the train_model, save_model, and save_train_val_loss methods
class UNet(BasicUNet, BaseModel):
    def __init__(
            self,
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            features=(32, 32, 64, 128, 256, 32),
            act=('LeakyReLU', {'inplace': True, 'negative_slope': 0.1}),
            norm=('instance', {'affine': True}),
            bias=True,
            dropout=0.1,
            upsample='deconv'
        ) -> None:
        """
        Initialize the UNet model.

        :param spatial_dims: Number of spatial dimensions.
        :type spatial_dims: int
        :param in_channels: Number of input channels.
        :type in_channels: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        :param features: Number of features in each layer.
        :type features: tuple
        :param act: Activation function.
        :type act: tuple
        :param norm: Normalization layer.
        :type norm: tuple
        :param bias: Whether to use bias in the convolutional layers.
        :type bias: bool
        :param dropout: Dropout rate.
        :type dropout: float
        :param upsample: Upsampling method.
        :type upsample: str
        """
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            upsample=upsample
        )

    def forward(self, x):
        """
        Forward pass through the UNet model.
        
        :param x: Input tensor.
        :return: Output tensor.
        """
        return super().forward(x)
    
    def train_model(
            self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            lr: float,
            loss_function: callable,
            num_epochs: int,
            model_filename: str = "Placeholder",
            data_path: str = "data",
        ) -> None:
        """
        Train the model.

        :param train_loader: DataLoader for training data.
        :train_loader type: torch.utils.data.DataLoader
        :param val_loader: DataLoader for validation data.
        :val_loader type: torch.utils.data.DataLoader
        :param lr: Learning rate.
        :lr type: float
        :param loss_function: Loss function to use.
        :loss_function type: callable
        :param num_epochs: Number of epochs to train for.
        :num_epochs type: int
        :param checkpoints: List of epochs to save checkpoints.
        :checkpoints type: list[int]
        :param wandb_project_name: WandB project name.
        :wandb_project_name type: str
        :param wandb_entity: WandB entity name.
        :wandb_entity type: str
        """
        # Initialize loggers
        logger = TrainingLogger(data_path)

        # Initialize the best validation loss
        best_val_loss = float('inf')

        # Load the best validation loss if in logger
        if logger.get_best_val_loss() < best_val_loss:
            best_val_loss = logger.get_best_val_loss()

        # Define the device
        device = get_device()
        self.to(device)

        print_box(f"Training on {device}!")

        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

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
            self.train()
            epoch_loss = 0
            for inputs, targets in train_loader:
                # Move the data to the device
                inputs = inputs.to(device)
                targets = targets.to(device)
                psf = inputs - targets

                # Zero the gradients
                optimizer.zero_grad()

                # Generate predictions
                outputs = self.forward(inputs)

                # Calculate the loss
                loss = loss_function(inputs, outputs, targets, psf)

                # Perform backpropagation
                loss.backward()
                optimizer.step()

                # Update the loss
                epoch_loss += loss.item()

            # Evaluate the model
            self.eval()
            val_loss = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Move the data to the device
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    psf = inputs - targets

                    # Generate predictions
                    outputs = self.forward(inputs)

                    # Calculate the loss
                    loss = loss_function(inputs, outputs, targets, psf)

                    # Update the loss
                    val_loss += loss.item()

            if logger.check_best_val_loss(val_loss / len(val_loader)):
                best_val_loss = val_loss / len(val_loader)
                self.save_model(f"{model_filename}_best_model", data_path)
                info = f"Best model saved with validation loss: {best_val_loss:.4f}"
                print_box(info)

            self.save_model(f"{model_filename}_epoch", data_path)
            info = f"Checkpoint model saved!"
            print_box(info)

            logger.log_epoch(
                train_loss=epoch_loss / len(train_loader),
                val_loss=val_loss / len(val_loader),
                best_val_loss=best_val_loss,
                optimizer=optimizer
            )

    def save_model(self, filename: str,  dir_: str):
        """
        Save the model to a file.
        
        :param filename: Name of the file to save the model to.
        :type name: str
        :param dir_: Directory to save the model to.
        :type dir_: str
        """
        super().save_model(filename, dir_)

    def load_model(self, filename: str,  dir_: str):
        """
        Load the model from a file.
        
        :param filename: Name of the file to load the model from.
        :type name: str
        :param dir_: Directory to load the model from.
        :type dir_: str
        """
        super().load_model(filename, dir_)
