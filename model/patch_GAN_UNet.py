import os
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from model._base_model import BaseModel
from model.patch_discriminator import PatchDiscriminator
from model.u_net import UNet
from model_utils.loss_functions import PerceptualLoss
from model_utils.lr_scheduler import DiscScheduler
from utils import print_box
from utils_utils.device import get_device
from loggers_utils import TrainingLogger


class PatchGANUNet(torch.nn.Module, BaseModel):
    def __init__(
            self,
            gen_features=(32, 32, 64, 128, 256, 32),
            disc_channels=(32, 64, 128),
            disc_in_channels=2,
            disc_out_channels=1,
            kernel_size=3,
            bias=True,
            gen_dropout=0.1,
            disc_dropout=0.4,
            l_rec_weight=1.0,
            l_adv_weight=0.05,
            l_p_weight=0.0,
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
        self.G = UNet(
            features=gen_features,
            bias=bias,
            dropout=gen_dropout
        )

        # Define the discriminator
        self.D = PatchDiscriminator(
            in_channels=disc_in_channels,
            out_channels=disc_out_channels,
            kernel_size=kernel_size,
            channels=disc_channels,
            dropout=disc_dropout
        )

        # Initialize the perceptual loss
        # self._perceptual_loss = PerceptualLoss()

        # Define the discriminator loss function
        self.loss_D = nn.BCEWithLogitsLoss()

        # Define loss weights
        self.l_rec_weight = l_rec_weight
        self.l_adv_weight = l_adv_weight
        self.l_p_weight = l_p_weight
        
        # Define the labels for real and fake images
        self.real_label = 0.9
        self.fake_label = 0.0

    def forward(self, x):
        """
        Forward pass of the model.
        
        :param x: Input tensor.
        :return: Output tensor.
        """
        return self.forward_G(x)

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
        x = torch.cat([input_img, target_img], dim=1) # (B, 2, H, W)
        return self.D(x)
    
    def train_model(
            self,
            train_loader,
            val_loader,
            lr,
            loss_function,
            num_epochs,
            model_name: str = "Placeholder",
            data_path: str = "data",
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
        """
        # Initialize loggers
        logger = TrainingLogger(data_path, adverserial_logger=True)

        # Initialize discriminator scheduler
        disc_scheduler = DiscScheduler(update_freq=10)

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
        optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lr, weight_decay=1e-4)
        optimizer_D = torch.optim.Adam(self.D.parameters(), lr=1e-5, weight_decay=1e-4)

         # Load optimzer if in logger
        optimizer_G_state = logger.get_optimizer_state()
        optimizer_D_state = logger.get_optimizer2_state()
        if optimizer_G_state is not None:
            optimizer_G.load_state_dict(optimizer_G_state)
            print_box("Optimizer state (Generator) loaded successfully!")
        if optimizer_D_state is not None:
            optimizer_D.load_state_dict(optimizer_D_state)
            print_box("Optimizer state (Discriminator) loaded successfully!")

        # Iterate over epochs
        for epoch in tqdm(
            range(num_epochs),
            desc='Training...'
        ):
            # Reset the disc scheduler
            disc_scheduler.reset()

            # Set the model to training mode
            self.D.train()
            self.G.train()

            # Log the ADV and REC loss for finetunning purpose
            ADV_loss = 0
            REC_loss = 0

            # Log the epoch losses
            epoch_loss_G = 0 
            epoch_loss_D = 0
            for inputs, targets in train_loader:
                # Move the data to the device
                inputs = inputs.to(device)
                targets = targets.to(device)
                psf = inputs - targets

                if disc_scheduler.step():
                    d_loss = self._step_disc(
                        inputs,
                        targets,
                        optimizer_D
                    )

                    # Log the discriminator loss
                    epoch_loss_D += d_loss.item()
                
                g_loss, g_adv_loss, g_rec_loss = self._step_gen(
                    inputs,
                    targets,
                    psf,
                    loss_function,
                    optimizer_G
                )

                # Log the generator loss
                epoch_loss_G += g_loss.item()

                # Log the ADV and REC loss for finetunning purpose
                ADV_loss += g_adv_loss.item()
                REC_loss += g_rec_loss.item()

                # Log the losses to the scheduler
                disc_scheduler.log_step(g_loss.item())

            # Evaluate the model
            self.G.eval()
            self.D.eval()
            val_loss = 0
            val_loss_D = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Move the data to the device
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    psf = inputs - targets

                    # Generate predictions
                    fake_output = self.forward_G(inputs)

                    # Generate predictions for the discriminator
                    real_pred = self.forward_D(inputs, targets)
                    fake_pred = self.forward_D(inputs, fake_output.detach())

                    # Calculate the loss
                    d_loss_real = self.loss_D(real_pred, torch.full_like(real_pred, self.real_label))
                    d_loss_fake = self.loss_D(fake_pred, torch.full_like(fake_pred, self.fake_label))
                    d_loss = (d_loss_real + d_loss_fake) * 0.5

                    # Calculate the adversarial loss
                    g_adv_loss = self.loss_D(
                        fake_pred,
                        torch.full_like(fake_pred, self.real_label)
                    )

                    # Calculate the reconstruction loss
                    g_rec_loss = loss_function(
                        inputs,
                        fake_output,
                        targets,
                        psf
                    )
                    
                    # Calculate the perceptual loss
                    # g_p_loss = self._perceptual_loss(
                    #     inputs,
                    #     fake_output,
                    #     targets,
                    #     psf
                    # )

                    # Calculate the total generator loss
                    g_loss = self.l_rec_weight * g_rec_loss + \
                        self.l_adv_weight * g_adv_loss# + \
                        #self.l_p_weight * g_p_loss

                    val_loss += g_loss.item()
                    val_loss_D += d_loss.item()

            if logger.check_best_val_loss(val_loss / len(val_loader)):
                best_val_loss = val_loss / len(val_loader)
                self.save_model(f"{model_name}_best_model", data_path)
                info = f"Best model saved at epoch {epoch} with validation loss: {best_val_loss:.4f}"
                print_box(info)

            self.save_model(f"{model_name}_epoch", data_path)
            info = f"Checkpoint model saved at epoch {epoch}!"
            print_box(info)

            logger.log_epoch(
                train_loss=epoch_loss_G / len(train_loader),
                val_loss=val_loss / len(val_loader),
                best_val_loss=best_val_loss,
                optimizer=optimizer_G,
                optimizer2=optimizer_D,
                train_loss_D=epoch_loss_D / disc_scheduler.get_num_steps(),
                val_loss_D=val_loss_D / len(val_loader),
            )

            # Print the epoch information for finetunning purpose
            info = f"Epoch {epoch + 1} completed!\n"
            info += f"ADV Loss {ADV_loss / len(train_loader)}\n"
            info += f"REC Loss {REC_loss / len(train_loader)}\n"
            info += f"Train Loss (Generator): {epoch_loss_G / len(train_loader)}\n"
            print_box(info)

    def _step_disc(self, inputs, targets, optimizer):
        # Zero the gradients
        optimizer.zero_grad()

        # Generate predictions
        fake_output = self.forward_G(inputs)

        real_pred = self.forward_D(inputs, targets)
        fake_pred = self.forward_D(inputs, fake_output.detach())

        # Switch the real and fake predictions with a probability of 0.1
        if torch.rand(1).item() < 0.1:
            real_pred, fake_pred = fake_pred, real_pred

        # Calculate the loss
        d_loss_real = self.loss_D(real_pred, torch.full_like(real_pred, self.real_label))
        d_loss_fake = self.loss_D(fake_pred, torch.full_like(fake_pred, self.fake_label))
        d_loss = (d_loss_real + d_loss_fake) * 0.5

        # Perform backpropagation
        d_loss.backward()
        optimizer.step()

        return d_loss

    def _step_gen(self, inputs, targets, psf, loss_function, optimizer):
        # Zero the gradients
        optimizer.zero_grad()

        # Generate predictions
        fake_output = self.forward_G(inputs)

        # Generate predictions
        fake_pred = self.forward_D(inputs, fake_output)

        # Calculate the adversarial loss
        g_adv_loss = self.loss_D(
            fake_pred,
            torch.full_like(fake_pred, self.real_label)
        )

        # Calculate the reconstruction loss
        g_rec_loss = loss_function(
            inputs,
            fake_output,
            targets,
            psf
        )
        
        # Calculate the perceptual loss
        # g_p_loss = self._perceptual_loss(
        #     inputs,
        #     fake_output,
        #     targets,
        #     psf
        # )

        # Calculate the total generator loss
        g_loss = self.l_rec_weight * g_rec_loss + \
            self.l_adv_weight * g_adv_loss# + \
            #self.l_p_weight * g_p_loss
        
        # Perform backpropagation
        g_loss.backward()
        optimizer.step()

        return g_loss, g_adv_loss, g_rec_loss #, g_p_loss

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

    def get_state_dict(self):
        """
        Get the state dictionary of the model.
        
        :return: State dictionary of the model.
        :rtype: dict
        """
        return {
            "generator": self.G.state_dict(),
            "discriminator": self.D.state_dict()
        }
    
    def load_from_state_dict(self, state_dict: dict):
        """
        Load the model from a state dictionary.
        
        :param state_dict: State dictionary to load the model from.
        The state dictionary should contain the keys "generator" and "discriminator".
        :type state_dict: dict
        """
        # Get the state dict of self
        self_g_dict = self.G.state_dict()
        self_d_dict = self.D.state_dict()

        # Get the state dict of the other model
        other_g_dict = state_dict["generator"]
        other_d_dict = state_dict["discriminator"]

        # Get the compatible weights
        compatible_weights_g, transferred_keys_g, skipped_keys_g = self._load_from_state_dict(
            source_state=other_g_dict,
            target_state=self_g_dict
        )
        compatible_weights_d, transferred_keys_d, skipped_keys_d = self._load_from_state_dict(
            source_state=other_d_dict,
            target_state=self_d_dict
        )

        # Load the compatible weights into the model
        self_g_dict.update(compatible_weights_g)
        self_d_dict.update(compatible_weights_d)

        # Load the state dict into the model
        self.G.load_state_dict(self_g_dict)
        self.D.load_state_dict(self_d_dict)

        num_params_g = self._count_state_dict_params(self_g_dict)
        num_params_d = self._count_state_dict_params(self_d_dict)
        num_params_other_g = self._count_state_dict_params(other_g_dict)
        num_params_other_d = self._count_state_dict_params(other_d_dict)


        info = f"Generator and Discriminator state dict loaded successfully!"
        print_box(info)
        info = f"Number of parameters in the generator: {num_params_g}\n"
        info += f"Number of parameters in the discriminator: {num_params_d}\n"
        info += f"Number of parameters in the other generator: {num_params_other_g}\n"
        info += f"Number of parameters in the other discriminator: {num_params_other_d}"
        print_box(info)
        info = f"Generator state dict:\n"
        info += f"Transferred {len(transferred_keys_g)} / {len(other_g_dict)} parameters. Transfer percentage: {len(transferred_keys_g) / len(other_g_dict) * 100:.2f}%\n"
        info += f"Skipped keys due to shape mismatch or absence: {len(skipped_keys_g)} / {len(other_g_dict)} parameters.\n"
        print_box(info)
        info = f"Discriminator state dict:\n"
        info += f"Transferred {len(transferred_keys_d)} / {len(other_d_dict)} parameters. Transfer percentage: {len(transferred_keys_d) / len(other_d_dict) * 100:.2f}%\n"
        info += f"Skipped keys due to shape mismatch or absence: {len(skipped_keys_d)} / {len(other_d_dict)} parameters."
        print_box(info)

    def _load_from_state_dict(self, source_state: dict, target_state: dict):
        # Track what can be transferred
        transferred_keys = []
        skipped_keys = []

        for key in source_state:
            if key in target_state and source_state[key].shape == target_state[key].shape:
                target_state[key] = source_state[key]
                transferred_keys.append(key)
            else:
                skipped_keys.append(key)

        return target_state, transferred_keys, skipped_keys

    def _count_state_dict_params(self, state_dict):
        return sum(v.numel() for v in state_dict.values())

    def load_model(self, discriminator_name: str, generator_name: str,  dir_ = "Default"):
        """
        Load the model from a file.
        
        :param name: Name of the file to load the model from.
        :type name: str
        """
        self.D.load_model(discriminator_name, dir_)
        self.G.load_model(generator_name, dir_)
