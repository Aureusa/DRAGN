import torch

from networks.models._base_model import BaseModel
from networks.models.patchdiscriminator import PatchDiscriminator
from networks.models.UNet import UNet


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

    def save_checkpoint(self, filename: str, dir_: str):
        """
        Save the model to a file.
        
        :param name: Name of the file to save the model to.
        :type name: str
        """
        self.D.save_checkpoint(f"discriminator_{filename}", dir_)
        self.G.save_checkpoint(f"generator_{filename}", dir_)

    def load_checkpoint(self, filename: str, dir_: str):
        """
        Load the model from a file.
        
        :param filename: Name of the file to load the model from.
        :type name: str
        :param dir_: Directory to load the model from.
        :type dir_: str
        """
        self.D.load_checkpoint(f"discriminator_{filename}", dir_)
        self.G.load_checkpoint(f"generator_{filename}", dir_)
