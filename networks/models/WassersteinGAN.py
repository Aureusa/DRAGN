from tqdm import tqdm
import torch

from networks.models._base_model import BaseModel
from networks.models.patchdiscriminator import PatchDiscriminator
from networks.models.UNet import UNet
from utils import print_box
from utils.device import get_device
from loggers_utils import TrainingLogger


class WGAN(BaseModel):
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
            use_gradient_penalty=True,
            gradient_penalty_weight=10.0,
            clip_value=0.01,
            *args,
            **kwargs
        ) -> None:
        """
        Initialize the Wasserstein GAN model.
        
        :param gen_features: Features for the generator UNet layers.
        :type gen_features: tuple
        :param disc_channels: Channels for the discriminator layers.
        :type disc_channels: tuple
        :param disc_in_channels: Number of input channels for discriminator.
        :type disc_in_channels: int
        :param disc_out_channels: Number of output channels for discriminator.
        :type disc_out_channels: int
        :param kernel_size: Kernel size for convolutions.
        :type kernel_size: int
        :param bias: Whether to use bias in convolutions.
        :type bias: bool
        :param gen_dropout: Dropout rate for generator.
        :type gen_dropout: float
        :param disc_dropout: Dropout rate for discriminator.
        :type disc_dropout: float
        :param l_rec_weight: Weight for reconstruction loss.
        :type l_rec_weight: float
        :param l_adv_weight: Weight for adversarial loss.
        :type l_adv_weight: float
        :param l_p_weight: Weight for perceptual loss.
        :type l_p_weight: float
        :param use_gradient_penalty: Whether to use gradient penalty (WGAN-GP).
        :type use_gradient_penalty: bool
        :param gradient_penalty_weight: Weight for gradient penalty term.
        :type gradient_penalty_weight: float
        :param clip_value: Weight clipping value (used when gradient_penalty=False).
        :type clip_value: float
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

        # Wasserstein GAN parameters
        self.use_gradient_penalty = use_gradient_penalty
        self.gradient_penalty_weight = gradient_penalty_weight
        self.clip_value = clip_value

        # Define loss weights
        self.l_rec_weight = l_rec_weight
        self.l_adv_weight = l_adv_weight
        self.l_p_weight = l_p_weight

    def from_config(cls, config):
        """
        Config construction - this should be abstract
        """
        raise NotImplementedError("from_config method is not implemented for AttentionUNET")

    def gradient_penalty(self, real_samples, fake_samples, input_imgs):
        """
        Calculate gradient penalty for WGAN-GP.
        
        :param real_samples: Real target images
        :param fake_samples: Generated fake images  
        :param input_imgs: Input images to concatenate with
        :return: Gradient penalty term
        """
        device = real_samples.device
        batch_size = real_samples.size(0)
        
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        
        # Get random interpolation between real and fake samples
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        # Calculate discriminator output for interpolated samples
        d_interpolates = self.forward_D(input_imgs, interpolates)
        
        # Calculate gradients of discriminator w.r.t. interpolated samples
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty

    def gradient_penalty_for_validation(self, real_samples, fake_samples, input_imgs):
        """
        Calculate gradient penalty for validation (enables gradients temporarily).
        
        :param real_samples: Real target images
        :param fake_samples: Generated fake images  
        :param input_imgs: Input images to concatenate with
        :return: Gradient penalty term
        """
        # Temporarily enable gradients for gradient penalty calculation
        with torch.enable_grad():
            device = real_samples.device
            batch_size = real_samples.size(0)
            
            # Random weight term for interpolation between real and fake samples
            alpha = torch.rand(batch_size, 1, 1, 1, device=device)
            
            # Get random interpolation between real and fake samples
            interpolates = alpha * real_samples + (1 - alpha) * fake_samples
            interpolates.requires_grad_(True)
            
            # Calculate discriminator output for interpolated samples
            d_interpolates = self.forward_D(input_imgs, interpolates)
            
            # Calculate gradients of discriminator w.r.t. interpolated samples
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(d_interpolates),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Calculate gradient penalty
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            
            return gradient_penalty

    def clip_discriminator_weights(self):
        """
        Clip discriminator weights to enforce Lipschitz constraint.
        Only used when gradient penalty is disabled.
        """
        if not self.use_gradient_penalty:
            for param in self.D.parameters():
                param.data.clamp_(-self.clip_value, self.clip_value)

    def get_wasserstein_distance(self, inputs, targets):
        """
        Calculate the Wasserstein distance estimate between real and fake distributions.
        
        :param inputs: Input images
        :param targets: Target (real) images
        :return: Wasserstein distance estimate
        """
        with torch.no_grad():
            fake_output = self.forward_G(inputs)
            real_pred = self.forward_D(inputs, targets)
            fake_pred = self.forward_D(inputs, fake_output)
            
            # Wasserstein distance is E[D(real)] - E[D(fake)]
            wasserstein_distance = torch.mean(real_pred) - torch.mean(fake_pred)
            return wasserstein_distance.item()

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
    
    def train_model(self, train_loader, loss_function, optimizer, device):
        # Unpack the optimizers
        optimizer_G, optimizer_D = optimizer

        # Set the model to training mode
        self.D.train()
        self.G.train()

        # Log the epoch losses
        epoch_loss_G = 0 
        epoch_loss_D = 0
        for inputs, targets in train_loader:
            # Move the data to the device
            inputs = inputs.to(device)
            targets = targets.to(device)
            psf = inputs - targets

            # Step the discriminator
            d_loss = self._step_disc(
                inputs,
                targets,
                optimizer_D
            )

            # Log the discriminator loss
            epoch_loss_D += d_loss.item()
        
            # Step the generator
            g_loss, g_adv_loss, g_rec_loss = self._step_gen(
                inputs,
                targets,
                psf,
                loss_function,
                optimizer_G
            )

            # Log the generator loss
            epoch_loss_G += g_loss.item()
        return tuple((epoch_loss_G / len(train_loader), epoch_loss_D / len(train_loader)))
    
    def validate_model(self, val_loader, loss_function, device):
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
                psf = inputs[:, 0:1, :, :] - targets

                # Generate predictions
                fake_output = self.forward_G(inputs)

                # Generate predictions for the discriminator
                real_pred = self.forward_D(inputs, targets)
                fake_pred = self.forward_D(inputs, fake_output.detach())

                # Wasserstein discriminator loss
                d_loss_real = -torch.mean(real_pred)
                d_loss_fake = torch.mean(fake_pred)
                d_loss = d_loss_real + d_loss_fake
                
                # Add gradient penalty for validation using the special validation method
                if self.use_gradient_penalty:
                    gp = self.gradient_penalty_for_validation(targets, fake_output.detach(), inputs)
                    d_loss += self.gradient_penalty_weight * gp

                # Wasserstein adversarial loss for generator
                g_adv_loss = -torch.mean(fake_pred)

                # Calculate the reconstruction loss
                g_rec_loss = loss_function(
                    inputs[:, 0:1, :, :],
                    fake_output,
                    targets,
                    psf
                )
                
                # Calculate the total generator loss
                g_loss = self.l_rec_weight * g_rec_loss + \
                    self.l_adv_weight * g_adv_loss

                val_loss += g_loss.item()
                val_loss_D += d_loss.item()
        return tuple((val_loss / len(val_loader), val_loss_D / len(val_loader)))

    # def train_model(
    #         self,
    #         train_loader,
    #         val_loader,
    #         lr,
    #         loss_function,
    #         num_epochs,
    #         model_name: str = "Placeholder",
    #         data_path: str = "data",
    #     ):
    #     """
    #     Train the model.

    #     :param train_loader: DataLoader for training data.
    #     :train_loader type: torch.utils.data.DataLoader
    #     :param val_loader: DataLoader for validation data.
    #     :val_loader type: torch.utils.data.DataLoader
    #     :param lr: Learning rate.
    #     :lr type: float
    #     :param loss_function: Loss function to use.
    #     :loss_function type: callable
    #     :param num_epochs: Number of epochs to train for.
    #     :num_epochs type: int
    #     :param checkpoints: List of epochs to save checkpoints.
    #     :checkpoints type: list[int]
    #     """
    #     # Initialize loggers
    #     logger = TrainingLogger(data_path, adversarial_logger=True)

    #     # Initialize the best validation loss
    #     best_val_loss = float('inf')

    #     # Load the best validation loss if in logger
    #     if logger.get_best_val_loss() < best_val_loss:
    #         best_val_loss = logger.get_best_val_loss()

    #     # Define the device
    #     device = get_device()
    #     self.to(device)

    #     print_box(f"Training on {device}!")

    #     # Define the optimizer
    #     optimizer_G = torch.optim.Adam(self.G.parameters(), lr=lr, weight_decay=1e-4)
    #     optimizer_D = torch.optim.Adam(self.D.parameters(), lr=1e-5, weight_decay=1e-4)

    #      # Load optimzer if in logger
    #     optimizer_G_state = logger.get_optimizer_state()
    #     optimizer_D_state = logger.get_optimizer2_state()
    #     if optimizer_G_state is not None:
    #         optimizer_G.load_state_dict(optimizer_G_state)
    #         print_box("Optimizer state (Generator) loaded successfully!")
    #     if optimizer_D_state is not None:
    #         optimizer_D.load_state_dict(optimizer_D_state)
    #         print_box("Optimizer state (Discriminator) loaded successfully!")

    #     # Iterate over epochs
    #     for epoch in tqdm(
    #         range(num_epochs),
    #         desc='Training...'
    #     ):
    #         # Set the model to training mode
    #         self.D.train()
    #         self.G.train()

    #         # Log the ADV and REC loss for finetunning purpose
    #         ADV_loss = 0
    #         REC_loss = 0

    #         # Log the epoch losses
    #         epoch_loss_G = 0 
    #         epoch_loss_D = 0
    #         for inputs, targets in train_loader:
    #             # Move the data to the device
    #             inputs = inputs.to(device)
    #             targets = targets.to(device)
    #             psf = inputs - targets

    #             # Step the discriminator
    #             d_loss = self._step_disc(
    #                 inputs,
    #                 targets,
    #                 optimizer_D
    #             )

    #             # Log the discriminator loss
    #             epoch_loss_D += d_loss.item()
            
    #             # Step the generator
    #             g_loss, g_adv_loss, g_rec_loss = self._step_gen(
    #                 inputs,
    #                 targets,
    #                 psf,
    #                 loss_function,
    #                 optimizer_G
    #             )

    #             # Log the generator loss
    #             epoch_loss_G += g_loss.item()

    #             # Log the ADV and REC loss for finetunning purpose
    #             ADV_loss += g_adv_loss.item()
    #             REC_loss += g_rec_loss.item()

    #         # Evaluate the model
    #         self.G.eval()
    #         self.D.eval()
    #         val_loss = 0
    #         val_loss_D = 0
    #         with torch.no_grad():
    #             for inputs, targets in val_loader:
    #                 # Move the data to the device
    #                 inputs = inputs.to(device)
    #                 targets = targets.to(device)
    #                 psf = inputs - targets

    #                 # Generate predictions
    #                 fake_output = self.forward_G(inputs)

    #                 # Generate predictions for the discriminator
    #                 real_pred = self.forward_D(inputs, targets)
    #                 fake_pred = self.forward_D(inputs, fake_output.detach())

    #                 # Wasserstein discriminator loss
    #                 d_loss_real = -torch.mean(real_pred)
    #                 d_loss_fake = torch.mean(fake_pred)
    #                 d_loss = d_loss_real + d_loss_fake
                    
    #                 # Add gradient penalty for validation
    #                 if self.use_gradient_penalty:
    #                     gp = self.gradient_penalty(targets, fake_output.detach(), inputs)
    #                     d_loss += self.gradient_penalty_weight * gp

    #                 # Wasserstein adversarial loss for generator
    #                 g_adv_loss = -torch.mean(fake_pred)

    #                 # Calculate the reconstruction loss
    #                 g_rec_loss = loss_function(
    #                     inputs,
    #                     fake_output,
    #                     targets,
    #                     psf
    #                 )
                    
    #                 # Calculate the total generator loss
    #                 g_loss = self.l_rec_weight * g_rec_loss + \
    #                     self.l_adv_weight * g_adv_loss

    #                 val_loss += g_loss.item()
    #                 val_loss_D += d_loss.item()

    #         if logger.check_best_val_loss(val_loss / len(val_loader)):
    #             best_val_loss = val_loss / len(val_loader)
    #             self.save_model(f"{model_name}_best_model", data_path)
    #             info = f"Best model saved at epoch {epoch} with validation loss: {best_val_loss:.4f}"
    #             print_box(info)

    #         self.save_model(f"{model_name}_epoch", data_path)
    #         info = f"Checkpoint model saved at epoch {epoch}!"
    #         print_box(info)

            # logger.log_epoch(
            #     train_loss=epoch_loss_G / len(train_loader),
            #     val_loss=val_loss / len(val_loader),
            #     best_val_loss=best_val_loss,
            #     optimizer=optimizer_G,
            #     optimizer2=optimizer_D,
            #     train_loss_D=epoch_loss_D / len(train_loader),
            #     val_loss_D=val_loss_D / len(val_loader),
            # )

    #         # Print the epoch information for finetunning purpose
    #         info = f"Epoch {epoch + 1} completed!\n"
    #         info += f"ADV Loss {ADV_loss / len(train_loader)}\n"
    #         info += f"REC Loss {REC_loss / len(train_loader)}\n"
    #         info += f"Train Loss (Generator): {epoch_loss_G / len(train_loader)}\n"
    #         print_box(info)

    def _step_disc(self, inputs, targets, optimizer):
        # Zero the gradients
        optimizer.zero_grad()

        # Generate predictions
        fake_output = self.forward_G(inputs)

        # Get discriminator predictions
        real_pred = self.forward_D(inputs, targets)
        fake_pred = self.forward_D(inputs, fake_output.detach())

        # Wasserstein loss: maximize D(real) - D(fake)
        # For minimization, we use: minimize -D(real) + D(fake)
        d_loss_real = -torch.mean(real_pred)
        d_loss_fake = torch.mean(fake_pred)
        d_loss = d_loss_real + d_loss_fake

        # Add gradient penalty if enabled
        if self.use_gradient_penalty:
            gp = self.gradient_penalty(targets, fake_output.detach(), inputs)
            d_loss += self.gradient_penalty_weight * gp

        # Perform backpropagation
        d_loss.backward()
        optimizer.step()

        # Clip weights if not using gradient penalty
        self.clip_discriminator_weights()

        return d_loss

    def _step_gen(self, inputs, targets, psf, loss_function, optimizer):
        # Zero the gradients
        optimizer.zero_grad()

        # Generate predictions
        fake_output = self.forward_G(inputs)

        # Generate predictions from discriminator
        fake_pred = self.forward_D(inputs, fake_output)

        # Wasserstein adversarial loss: maximize D(fake) = minimize -D(fake)
        g_adv_loss = -torch.mean(fake_pred)

        # Calculate the reconstruction loss
        g_rec_loss = loss_function(
            inputs,
            fake_output,
            targets,
            psf
        )

        # Calculate the total generator loss
        g_loss = self.l_rec_weight * g_rec_loss + \
            self.l_adv_weight * g_adv_loss# + \
            #self.l_p_weight * g_p_loss
        
        # Perform backpropagation
        g_loss.backward()
        optimizer.step()

        return g_loss, g_adv_loss, g_rec_loss #, g_p_loss

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
    
    def save_model(self, filename: str, dir_: str):
        """
        Save the model to a file.
        
        :param filename: Name of the file to save the model to.
        :type name: str
        :param dir_: Directory to save the model to.
        :type dir_: str
        """
        self.D.save_model(f"discriminator_{filename}", dir_)
        self.G.save_model(f"generator_{filename}", dir_)

    def load_model(self, d_filename: str, g_filename: str,  dir_: str):
        """
        Load the model from a file.
        
        :param d_filename: Name of the file to load the discriminator model from.
        :type d_filename: str
        :param g_filename: Name of the file to load the generator model from.
        :type g_filename: str
        :param dir_: Directory to load the model from.
        :type dir_: str
        """
        self.D.load_model(d_filename, dir_)
        self.G.load_model(g_filename, dir_)
