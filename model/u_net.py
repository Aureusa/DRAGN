import torch
import torch.nn as nn
from monai.networks.nets import BasicUNet

from model._base_model import BaseModel

# TODO: Impliment the train_model, save_model, and save_train_val_loss methods
class UNet(BasicUNet, BaseModel):
    def __init__(
            self,
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            features=(32, 32, 64, 128, 256, 32),
            act=('LeakyReLU', {'inplace': True, 'negative_slope': 0.1}),
            norm=('instance', {'affine': True}),
            bias=True,
            dropout=0.0,
            upsample='deconv'
        ) -> None:
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
    
    def train_model(self):
        raise NotImplementedError("train_model method is not implemented for UNet.")
    
    def save_train_val_loss(self):
        raise NotImplementedError("save_train_val_loss method is not implemented for UNet.")

    def train_model(self):
        raise NotImplementedError("train_model method is not implemented for UNet.")

    def save_model(self):
        raise NotImplementedError("save_model method is not implemented for UNet.")
    