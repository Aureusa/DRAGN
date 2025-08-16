from typing import Dict, Any
from monai.networks.nets import BasicUNet

from networks.models._base_model import BaseModel
from core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("unet")
class UNet(BasicUNet, BaseModel):
    def __init__(
            self,
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            features=(32, 32, 64, 128, 256, 32),
            act=('LeakyReLU', {'inplace': True, 'negative_slope': 0.1}),
            norm=('instance', {'affine': True}),
            bias=True,
            dropout=0.1,
            upsample='deconv'
        ) -> None:
        """
        Initialize the UNet model.

        :param spatial_dims: Number of spatial dimensions.
        :type spatial_dims: int
        :param in_channels: Number of input channels.
        :type in_channels: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        :param features: Number of features in each layer.
        :type features: tuple
        :param act: Activation function.
        :type act: tuple
        :param norm: Normalization layer.
        :type norm: tuple
        :param bias: Whether to use bias in the convolutional layers.
        :type bias: bool
        :param dropout: Dropout rate.
        :type dropout: float
        :param upsample: Upsampling method.
        :type upsample: str
        """
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

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> "UNet":
        """
        Config construction - this should be abstract
        """
        return cls(
            spatial_dims=config.get("spatial_dims", 2),
            in_channels=config.get("in_channels", 1),
            out_channels=config.get("out_channels", 1),
            features=config.get("features", (32, 32, 64, 128, 256, 32)),
            act=config.get("act", ('LeakyReLU', {'inplace': True, 'negative_slope': 0.1})),
            norm=config.get("norm", ('instance', {'affine': True})),
            bias=config.get("bias", True),
            dropout=config.get("dropout", 0.1),
            upsample=config.get("upsample", 'deconv')
        )

    def forward(self, x):
        """
        Forward pass through the UNet model.
        
        :param x: Input tensor.
        :return: Output tensor.
        """
        return super().forward(x)
