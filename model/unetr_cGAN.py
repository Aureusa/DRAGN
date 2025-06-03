from monai.networks.nets import UNETR, ViT
import torch
import os

from model.tripple_cGAN import tripple_cGAN

class unetr_cGAN(tripple_cGAN):
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
            layers=None,
            l_rec_weight=1,
            l_adv_weight=10,
            l_p_weight=0.01,
            *args,
            **kwargs
        ) -> None:
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
            layers=layers,
            l_rec_weight=l_rec_weight,
            l_adv_weight=l_adv_weight,
            l_p_weight=l_p_weight,
            *args,
            **kwargs
        )
        self.G = UNETR(
            1,
            1,
            img_size=(128,128),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type='conv',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            dropout_rate=0.1,
            spatial_dims=2,
        )

        self.D = ViT(
            in_channels,
            patch_size=16,
            img_size=(128,128),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            proj_type='conv',
            pos_embed_type='learnable',
            classification=True,
            num_classes=1,
            dropout_rate=0.0,
            spatial_dims=2,
            post_activation='Ignore',
            qkv_bias=False,
            save_attn=False
        )

    def forward_D(self, input_img, target_img):
        """
        Forward pass of the discriminator.
        
        :param input_img: Input image tensor.
        :param target_img: Target image tensor.
        :return: Output tensor.
        """
        return self.D(target_img)[0]


class UNetR(unetr_cGAN):
    def __init__(self, discriminator_in_shape=(2, 128, 128), spatial_dims=2, in_channels=1, out_channels=1, channels=(64, 128, 256, 512), strides=(2, 2, 2), kernel_size=3, up_kernel_size=3, dropout=0.1, layers=None, l_rec_weight=1, l_adv_weight=10, l_p_weight=0.01, *args, **kwargs):
        super().__init__(discriminator_in_shape, spatial_dims, in_channels, out_channels, channels, strides, kernel_size, up_kernel_size, dropout, layers, l_rec_weight, l_adv_weight, l_p_weight, *args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """
        Load the state dictionary into the model.
        
        :param state_dict: State dictionary to load.
        :param strict: Whether to enforce strict loading.
        """
        self.G.load_state_dict(state_dict, strict)

    def forward(self, x):
        return self.G(x)