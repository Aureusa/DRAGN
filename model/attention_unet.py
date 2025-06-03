import os
from tqdm import tqdm
from copy import deepcopy
import pickle
import wandb
import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet

from model._base_model import BaseModel
from loggers_utils import TrainingLogger
from utils import print_box
from utils_utils.device import get_device


class AttentionUNET(AttentionUnet, BaseModel):
    def __init__(
            self,
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 256, 512),
            strides=(2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
            dropout=0.1,
            *args,
            **kwargs
        ) -> None:
        """
        Initialize the AttentionUNET model.
        
        :param spatial_dims: Number of spatial dimensions.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param channels: Number of channels in each layer.
        :param strides: Strides for each layer.
        :param kernel_size: Kernel size for each layer.
        :param up_kernel_size: Kernel size for upsampling.
        :param dropout: Dropout rate.
        """
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            dropout=dropout,
            *args,
            **kwargs
        )

    def forward(self, x):
        """
        Forward pass of the model.

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
            model_name: str = "Placeholder",
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
        run = wandb.init(project=model_name, name=model_name)
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
            desc='Training...'
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

                # Log losses to WandB
                wandb.log({
                    "train_loss": loss.item() / len(inputs),
                })

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

                    # Log losses to WandB
                    wandb.log({
                        "val_loss": loss.item() / len(inputs),
                    })


            if logger.check_best_val_loss(val_loss / len(val_loader)):
                best_val_loss = val_loss / len(val_loader)
                self.save_model(f"{model_name}_best_model", data_path)
                info = f"Best model saved at epoch {epoch} with validation loss: {best_val_loss:.4f}"
                print_box(info)

            self.save_model(f"{model_name}_epoch", data_path)
            info = f"Checkpoint model saved at epoch {epoch}!"
            print_box(info)

            logger.log_epoch(
                train_loss=epoch_loss / len(train_loader),
                val_loss=val_loss / len(val_loader),
                best_val_loss=best_val_loss,
                optimizer=optimizer
            )
            
        # Finish WandB run
        wandb.finish()

    def save_model(self, name: str, data_path: str):
        """
        Save the model to a file.
        
        :param name: Name of the file to save the model to.
        :type name: str
        """
        path = os.path.join(data_path, f"{name}.pth")
        torch.save(self.state_dict(), path)

        info = f"Model `{name}.pth` saved successfully!"
        info += f"Path to model: {data_path}"
        print_box(info)

    def load_model(self, filename: str,  dir_ = "Default"):
        """
        Load the model from a file.
        
        :param filename: Name of the file to load the model from.
        :type name: str
        :param dir_: Directory to load the model from.
        :type dir_: str
        """
        if dir_ == "Default":
            dir_ = os.path.join(os.getcwd(), "data", "saved_models")

        self.load_state_dict(torch.load(os.path.join(dir_, f"{filename}.pth"), map_location=torch.device('cpu')))

        info = f"Model `{filename}` loaded successfully!"
        print_box(info)
