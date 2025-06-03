import os
from copy import deepcopy
from tqdm import tqdm
import wandb
import pickle
import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet, Discriminator

from model._base_model import BaseModel
from utils import print_box

class cGAN(torch.nn.Module, BaseModel):
    def __init__(
            self,
            discriminator_in_shape=(2, 128, 128),
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
        Initialize the cGAN model.
        
        :param discriminator_in_shape: Shape of the input to the discriminator.
        :discriminator_in_shape type: tuple
        :param spatial_dims: Number of spatial dimensions.
        :spatial_dims type: int
        :param in_channels: Number of input channels.
        :in_channels type: int
        :param out_channels: Number of output channels.
        :out_channels type: int
        :param channels: Number of channels in each layer.
        :channels type: tuple
        :param strides: Strides for each layer.
        :strides type: tuple
        :param kernel_size: Kernel size for each layer.
        :kernel_size type: int
        :param up_kernel_size: Kernel size for upsampling.
        :up_kernel_size type: int
        :param dropout: Dropout rate.
        :dropout type: float
        """
        super().__init__()

        # Define the generator
        self.G = AttentionUnet(
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
        
        # Define the discriminator
        self.D = Discriminator(
            in_shape = discriminator_in_shape,
            channels=channels,
            strides=strides,
        )
        self.loss_D = nn.BCEWithLogitsLoss()

        # Objective function weight
        self.of_loss_weight = 100

        # Define the training and validation loss history
        self._val_loss = []
        self._train_loss_discriminator = []
        self._train_loss_generator = []

        self.real_label = 1.0
        self.fake_label = 0.0

    @property
    def train_loss_generator(self):
        """
        Get the training loss history for the generator.
        
        :return: A deep copy of the training loss history for the generator."""
        return deepcopy(self._train_loss_generator)
    
    @property
    def train_loss_discriminator(self):
        """
        Get the training loss history for the discriminator.

        :return: A deep copy of the training loss history for the discriminator.
        """
        return deepcopy(self._train_loss_discriminator)
    
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
        return self.G(x)

    def forward_G(self, x):
        """
        Forward pass of the generator.
        
        :param x: Input tensor.
        :return: Output tensor.
        """
        return self.G(x)
    
    def forward_D(self, input_img, target_img):
        """
        Forward pass of the discriminator.
        
        :param input_img: Input image tensor.
        :param target_img: Target image tensor.
        :return: Output tensor.
        """
        x = torch.cat([input_img, target_img], dim=1)
        return self.D(x)
    
    def train_model(
            self,
            train_loader,
            val_loader,
            lr,
            loss_function,
            num_epochs,
            model_name: str = "Placeholder",
            data_path: str = "data"
        ):
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
        run = wandb.init(project=model_name, name=model_name)
        # Define the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        print_box(f"Training on {device}!")

        # Define the optimizer
        optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lr)
        optimizer_D = torch.optim.Adam(self.D.parameters(), lr=lr)

        # Iterate over epochs
        for epoch in tqdm(
            range(num_epochs),
            desc='Training...'
        ):
            self.D.train()
            self.G.train()

            epoch_loss_G = 0 
            epoch_loss_D = 0 
            for inputs, targets in train_loader:
                # Move the data to the device
                inputs = inputs.to(device)
                targets = targets.to(device)
                psf = inputs - targets

                # ------- Train the Discriminator -------
                # Zero the gradients
                optimizer_D.zero_grad()

                # Generate predictions
                fake_output = self.forward_G(inputs)

                real_pred = self.forward_D(inputs, targets)
                fake_pred = self.forward_D(inputs, fake_output.detach())

                # Calculate the loss
                d_loss_real = self.loss_D(real_pred, torch.full_like(real_pred, self.real_label))
                d_loss_fake = self.loss_D(fake_pred, torch.full_like(fake_pred, self.fake_label))
                d_loss = (d_loss_real + d_loss_fake) * 0.5


                # Perform backpropagation
                d_loss.backward()
                optimizer_D.step()

                # ------- Train the Generator -------
                # Zero the gradients
                optimizer_G.zero_grad()

                # Generate predictions
                fake_pred = self.forward_D(inputs, fake_output)

                # Calculate the loss
                g_adv_loss = self.loss_D(
                    fake_pred,
                    torch.full_like(fake_pred, self.real_label)
                ) # Adversarial loss
                g_of_loss = (
                    loss_function(
                        inputs,
                        fake_output,
                        targets,
                        psf
                    ) * self.of_loss_weight
                ) # Objective function loss
                g_loss = (
                    g_adv_loss + g_of_loss
                ) # Combine the losses

                # Perform backpropagation
                g_loss.backward()
                optimizer_G.step()

                epoch_loss_G += g_loss.item()
                epoch_loss_D += d_loss.item()

            # Evaluate the model
            self.G.eval()
            val_loss = 0

            l1_loss = nn.L1Loss()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Move the data to the device
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    psf = inputs - targets

                    # Generate predictions
                    outputs = self.forward(inputs)

                    # Calculate the loss
                    loss = l1_loss(outputs, targets)

                    # Update the loss
                    val_loss += loss.item()

            # Save the training and validation loss
            self._train_loss_generator.append(epoch_loss_D / len(train_loader))
            self._train_loss_discriminator.append(epoch_loss_D / len(train_loader))
            self._val_loss.append(val_loss / len(val_loader))

            # Log losses to WandB
            wandb.log({
                "train_loss_Gen": epoch_loss_G / len(train_loader),
                "train_loss_Disc": epoch_loss_D / len(train_loader),
                "val_loss": val_loss / len(val_loader)
            })

            # Save the model
            self.save_model(f"{model_name}_epoch", data_path)
            info = f"Checkpoint model saved at epoch {epoch}!"
            print_box(info)

            # Info about the training
            info = f"Epoch Loss Generator: {epoch_loss_G / len(train_loader):.4f}"
            info += f"\nEpoch Loss Discriminator: {epoch_loss_D / len(train_loader):.4f}"
            info += f"\nValidation Loss: {val_loss / len(val_loader):.4f}"
            print_box(info)
            
        # Finish WandB run
        wandb.finish()

    def save_model(self, name: str, data_path: str):
        """
        Save the model to a file.
        
        :param name: Name of the file to save the model to.
        :type name: str
        """
        d_path = os.path.join(data_path, f"discriminator_{name}.pth")
        g_path = os.path.join(data_path, f"generator_{name}.pth")

        torch.save(self.D.state_dict(), d_path)
        info = f"Discriminator model `discriminator_{name}.pth` saved successfully!"

        torch.save(self.G.state_dict(), g_path)
        info += f"\nGenerator model with generator_{name}.pth saved successfully!"

        info += f"Path to model: {data_path}"
        print_box(info)

    def load_model(self, discriminator_name: str, generator_name: str,  dir_ = "Default"):
        """
        Load the model from a file.
        
        :param name: Name of the file to load the model from.
        :type name: str
        """
        if dir_ == "Default":
            dir_ = os.path.join(os.getcwd(), "data", "saved_models")
        info = f"Path to model: {dir_}"

        self.D.load_state_dict(torch.load(os.path.join(dir_, f"{discriminator_name}.pth"), map_location=torch.device('cpu')))
        info += f"\nDiscriminator model `{discriminator_name}.pth` loaded successfully!"

        self.G.load_state_dict(torch.load(os.path.join(dir_, f"{generator_name}.pth"), map_location=torch.device('cpu')))
        info += f"\nGenerator model `{generator_name}.pth` loaded successfully!"
        print_box(info)

    def save_train_val_loss(self, data_dir: str):
        """
        Save the training and validation loss.

        :param data_dir: Directory to save the loss files.
        :type data_dir: str
        """
        train_loss_path = os.path.join(data_dir, f"train_loss_g.pkl")
        with open(train_loss_path, "wb") as file:
            pickle.dump(self.train_loss_generator, file)

        info = f"Training Loss (Generator) saved successfully in `{train_loss_path}`!"

        train_loss_path = os.path.join(data_dir, f"train_loss_d.pkl")
        with open(train_loss_path, "wb") as file:
            pickle.dump(self.train_loss_discriminator, file)

        info += f"\nTraining Loss (Discriminator) saved successfully in `{train_loss_path}`!"

        val_loss_path = os.path.join(data_dir, f"val_loss.pkl")
        with open(val_loss_path, "wb") as file:
            pickle.dump(self.val_loss, file)

        info += f"\nValidation Loss saved successfully in `{val_loss_path}`!"
        print_box(info)
