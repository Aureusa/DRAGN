import os
from tqdm import tqdm
from copy import deepcopy
import pickle
import wandb
import torch
from monai.networks.nets import AttentionUnet

from model._base_model import BaseModel
from utils import print_box


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

        # Define the training and validation loss history
        self._train_loss = []
        self._val_loss = []

    @property
    def train_loss(self):
        """
        Get the training loss history.
        
        :return: A deep copy of the training loss history.
        """
        return deepcopy(self._train_loss)
    
    @property
    def val_loss(self):
        """
        Get the validation loss history.
        
        :return: A deep copy of the validation loss history.
        """
        return deepcopy(self._val_loss)

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
            checkpoints: list[int] = [25],
            wandb_project_name: str = "Deep-AGN-Clean",
            wandb_entity: str = "myverynicemodel"
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
        run = wandb.init(project=wandb_project_name, name=wandb_entity)
        # Define the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        print_box(f"Training on {device}!")

        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Iterate over epochs
        for epoch in tqdm(
            range(num_epochs),
            desc='Training...'
        ):
            self.train()
            epoch_loss = 0
            for inputs, targets, psf in train_loader:
                # Move the data to the device
                inputs = inputs.to(device)
                targets = targets.to(device)
                psf = psf.to(device)

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
                for inputs, targets, psf in val_loader:
                    # Move the data to the device
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    psf = psf.to(device)

                    # Generate predictions
                    outputs = self.forward(inputs)

                    # Calculate the loss
                    loss = loss_function(inputs, outputs, targets, psf)

                    # Update the loss
                    val_loss += loss.item()

            # Save the training and validation loss
            self._train_loss.append(epoch_loss / len(train_loader))
            self._val_loss.append(val_loss / len(val_loader))

            # Log losses to WandB
            wandb.log({
                "train_loss": epoch_loss / len(train_loader),
                "val_loss": val_loss / len(val_loader)
            })

            if epoch in checkpoints:
                self.save_model(f"{wandb_entity}_epoch_{epoch}")

            print(f"Epoch Loss: {epoch_loss / len(train_loader):.4f}, "
                  f"Validation Loss: {val_loss / len(val_loader):.4f}")
            
        # Finish WandB run
        wandb.finish()

    def save_model(self, name: str):
        """
        Save the model to a file.
        
        :param name: Name of the file to save the model to.
        :type name: str
        """
        torch.save(self.state_dict(), f"{name}.pth")

        info = f"Model `{name}.pth` saved successfully!"
        info += f"Path to model: {os.getcwd()}"
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

        self.load_state_dict(torch.load(os.path.join(dir_, filename), map_location=torch.device('cpu')))

        info = f"Model `{filename}` loaded successfully!"
        print_box(info)

    def save_train_val_loss(self, data_dir: str):
        """
        Save the training and validation loss.

        :param data_dir: Directory to save the loss files.
        :type data_dir: str
        """
        train_loss_path = os.path.join(data_dir, f"train_loss.pkl")
        with open(train_loss_path, "wb") as file:
            pickle.dump(self.train_loss, file)

        info = f"Training Loss saved successfully in `{train_loss_path}`!"

        val_loss_path = os.path.join(data_dir, f"val_loss.pkl")
        with open(val_loss_path, "wb") as file:
            pickle.dump(self.val_loss, file)

        info += f"\nValidation Loss saved successfully in `{val_loss_path}`!"
        print_box(info)
