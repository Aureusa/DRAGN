import torch
import torch.nn as nn

from networks.models._base_model import BaseModel
from networks.models.patchdiscriminator import PatchDiscriminator
from networks.models.UNet import UNet



class PatchGAN(BaseModel):
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
        from model_training.loss_functions import PerceptualLoss
    
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
        # self.perceptual_loss = PerceptualLoss()

        # Define the discriminator loss function
        self.loss_D = nn.BCEWithLogitsLoss()

        # Define loss weights
        self.l_rec_weight = l_rec_weight
        self.l_adv_weight = l_adv_weight
        self.l_p_weight = l_p_weight
        
        # Define the labels for real and fake images
        self.real_label = 0.9
        self.fake_label = 0.0

    def from_config(cls, config):
        """
        Config construction - this should be abstract
        """
        raise NotImplementedError("from_config method is not implemented for AttentionUNET")

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
