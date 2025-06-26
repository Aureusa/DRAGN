import torch.nn as nn

from networks.blocks.convblock import ConvBlock
from networks.blocks.attnblock import AttnBlock


class ConvBlockWithAttention(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dropout=0.1,
            norm=True
        ):
        super().__init__()
        self.conv = ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            downsample=False,
            dropout=dropout,
            norm=norm
        )
        self.attn = AttnBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.attn(x)
        return x