from monai.networks.nets import AttentionUnet

from networks.models._base_model import BaseModel


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
        return super().forward(x)
