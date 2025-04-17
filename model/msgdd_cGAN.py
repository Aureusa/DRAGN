import os
from copy import deepcopy
from tqdm import tqdm
import wandb
import pickle
import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet, Discriminator

from model.cond_GAN import cGAN
from utils import print_box


class MSGDD_cGAN(cGAN):
    """
    Multi-Scale Gradients Dual Discriminator Conditional Generative Adversarial Network (MSGDD-cGAN) model.
    This model is a subclass of the cGAN model and includes two discriminators.
    The first discriminator is conditioned on the input image, while the second discriminator
    is conditioned on the real point spread function (PSF) image.
    """
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
        super().__init__(
            discriminator_in_shape=discriminator_in_shape,
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
        # Define the discriminators
        self.D2 = Discriminator(
            in_shape=discriminator_in_shape,
            channels=channels,
            strides=strides,
        )

        # Define the train loss history
        self._train_loss_discriminator2 = []

    @property
    def train_loss_discriminator2(self):
        """
        Get the training loss history for the second discriminator.

        :return: A deep copy of the training loss history for the discriminator.
        """
        return deepcopy(self._train_loss_discriminator2)
    
    def forward_D2(self, real_psf, fake_psf):
        """
        Forward pass of the second discriminator.
        
        :param input_img: Input image tensor.
        :param target_img: Target image tensor.
        :return: Output tensor.
        """
        x = torch.cat([real_psf, fake_psf], dim=1)
        return self.D2(x)
    
    def train_model(
            self,
            train_loader,
            val_loader,
            lr,
            loss_function,
            num_epochs,
            checkpoints: list[int] = [25],
            wandb_project_name: str = "Deep-AGN-Clean",
            wandb_entity: str = "myverynicemodel"
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
        run = wandb.init(project=wandb_project_name, name=wandb_entity)
        # Define the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        print_box(f"Training on {device}!")

        # Define the optimizer
        optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lr)
        optimizer_D = torch.optim.Adam(self.D.parameters(), lr=lr)
        optimizer_D2 = torch.optim.Adam(self.D2.parameters(), lr=lr)

        # Iterate over epochs
        for epoch in tqdm(
            range(num_epochs),
            desc='Training...'
        ):
            self.D.train()
            self.D2.train()
            self.G.train()

            epoch_loss_G = 0 
            epoch_loss_D = 0 
            epoch_loss_D2 = 0 
            for inputs, targets, psf in train_loader:
                # Move the data to the device
                inputs = inputs.to(device)
                targets = targets.to(device)
                psf = psf.to(device)

                # ------- Train the first Discriminator (conditioned on the input image) -------
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

                # ------- Train the second Discriminator (conditioned on the real psf image) -------
                # Zero the gradients
                optimizer_D2.zero_grad()

                # Generate predictions
                fake_output_psf = inputs - self.forward_G(inputs)

                real_pred_psf = self.forward_D(psf, psf)
                fake_pred_psf = self.forward_D(psf, fake_output_psf.detach())

                # Calculate the loss
                d2_loss_real = self.loss_D(real_pred_psf, torch.full_like(real_pred_psf, self.real_label))
                d2_loss_fake = self.loss_D(fake_pred_psf, torch.full_like(fake_pred_psf, self.fake_label))
                d2_loss = (d2_loss_real + d2_loss_fake) * 0.5


                # Perform backpropagation
                d2_loss.backward()
                optimizer_D2.step()

                # ------- Train the Generator -------
                # Zero the gradients
                optimizer_G.zero_grad()

                # Generate predictions
                fake_pred = self.forward_D(inputs, fake_output)
                fake_pred_psf = self.forward_D2(psf, fake_output_psf)

                # Calculate the loss
                g_adv_loss1 = self.loss_D(
                    fake_pred,
                    torch.full_like(fake_pred, self.real_label)
                ) # Adversarial loss for the first discriminator
                g_adv_loss2 = self.loss_D(
                    fake_pred_psf,
                    torch.full_like(fake_pred_psf, self.real_label)
                ) # Adversarial loss for the second discriminator
                g_adv_loss = (
                    g_adv_loss1 + g_adv_loss2
                ) * 0.5 # Combine the adversarial losses
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
                epoch_loss_D2 += d2_loss.item()

            # Evaluate the model
            self.G.eval()
            val_loss = 0

            l1_loss = nn.L1Loss()
            with torch.no_grad():
                for inputs, targets, psf in val_loader:
                    # Move the data to the device
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    psf = psf.to(device)

                    # Generate predictions
                    outputs = self.forward(inputs)

                    # Calculate the loss
                    loss = l1_loss(outputs, targets)

                    # Update the loss
                    val_loss += loss.item()

            # Save the training and validation loss
            self._train_loss_generator.append(epoch_loss_D / len(train_loader))
            self._train_loss_discriminator.append(epoch_loss_D / len(train_loader))
            self._train_loss_discriminator2.append(epoch_loss_D2 / len(train_loader))
            self._val_loss.append(val_loss / len(val_loader))

            # Log losses to WandB
            wandb.log({
                "train_loss_Gen": epoch_loss_G / len(train_loader),
                "train_loss_Disc": epoch_loss_D / len(train_loader),
                "val_loss": val_loss / len(val_loader)
            })

            if epoch in checkpoints:
                self.save_model(f"{wandb_entity}_epoch_{epoch}")

            print(f"Epoch Loss Generator: {epoch_loss_G / len(train_loader):.4f},"
                  f"Epoch Loss Discriminator: {epoch_loss_D / len(train_loader):.4f}"
                  f"Validation Loss: {val_loss / len(val_loader):.4f}")
            
        # Finish WandB run
        wandb.finish()

    def save_model(self, name: str):
        """
        Save the model to a file.
        
        :param name: Name of the file to save the model to.
        :type name: str
        """
        torch.save(self.D.state_dict(), f"discriminator1_{name}.pth")
        info = f"First discriminator model `discriminator1_{name}.pth` saved successfully!"

        torch.save(self.D2.state_dict(), f"discriminator2_{name}.pth")
        info = f"Second discriminator model `discriminator2_{name}.pth` saved successfully!"

        torch.save(self.G.state_dict(), f"generator_{name}.pth")
        info += f"\nGenerator model with generator_{name}.pth saved successfully!"

        info += f"Path to models: {os.getcwd()}"
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

        train_loss_path = os.path.join(data_dir, f"train_loss_d1.pkl")
        with open(train_loss_path, "wb") as file:
            pickle.dump(self.train_loss_discriminator, file)

        info += f"\nTraining Loss (First Discriminator) saved successfully in `{train_loss_path}`!"

        train_loss_path = os.path.join(data_dir, f"train_loss_d2.pkl")
        with open(train_loss_path, "wb") as file:
            pickle.dump(self.train_loss_discriminator2, file)

        info += f"\nTraining Loss (Second Discriminator) saved successfully in `{train_loss_path}`!"

        val_loss_path = os.path.join(data_dir, f"val_loss.pkl")
        with open(val_loss_path, "wb") as file:
            pickle.dump(self.val_loss, file)

        info += f"\nValidation Loss saved successfully in `{val_loss_path}`!"
        print_box(info)
