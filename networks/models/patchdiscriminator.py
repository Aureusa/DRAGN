import os
import torch
import torch.nn as nn


from networks.blocks import ConvBlock, ConvBlockWithAttention
from networks.models._base_model import BaseModel
from utils import print_box


class PatchDiscriminatorWithAttention(BaseModel):
    def __init__(self, in_channels=2, out_channels=1, kernel_size=3, channels=(64, 128, 256, 512), dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels

        # Create the first convolutional block without attention
        self.first_conv = ConvBlock(in_channels, channels[0], kernel_size=kernel_size, downsample=True, dropout=dropout)

        # Create the convolutional blocks with attention
        self.conv_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            if i >= 1:
                # Create the attention block
                att_block = ConvBlockWithAttention(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                )

                # Create the convolutional block to downsample the feature maps
                block = ConvBlock(
                    in_channels=channels[i + 1],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    downsample=True,
                    dropout=dropout
                )

                # Append the blocks to the list
                self.conv_blocks.append(att_block)
                self.conv_blocks.append(block)
            else:
                # Create the convolutional block to downsample the feature maps
                block = ConvBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    downsample=True,
                    dropout=dropout
                )

                # Append the block to the list
                self.conv_blocks.append(block)

        # Create the final convolutional layer
        self.final_conv = nn.Conv2d(channels[-1], out_channels, kernel_size=kernel_size, stride=1, padding=1)

    def from_config(cls, config):
        """
        Config construction - this should be abstract
        """
        raise NotImplementedError("from_config method is not implemented for AttentionUNET")

    def forward(self, x):
        # Pass the input through the first convolutional block
        x = self.first_conv(x)

        # Pass the input through the convolutional blocks with attention
        for block in self.conv_blocks:
            x = block(x)

        # Pass the output through the final convolutional layer
        x = self.final_conv(x)
        return x
    

class PatchDiscriminator(BaseModel):
    def __init__(self, in_channels=2, out_channels=1, kernel_size=3, channels=(16, 32, 64), dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels

        # First convolutional block without normalization
        self.blocks = nn.ModuleList()
        self.blocks.append(ConvBlock(in_channels, channels[0], kernel_size=kernel_size, downsample=True, dropout=dropout, norm=False))

        # Intermediate blocks (no attention)
        for i in range(len(channels) - 1):
            self.blocks.append(ConvBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_size,
                downsample=True,
                dropout=dropout
            ))

        # Final convolutional layer (produces 1-channel patch logits)
        self.final_conv = nn.Conv2d(channels[-1], out_channels, kernel_size=kernel_size, stride=1, padding=1)

    def from_config(cls, config):
        """
        Config construction - this should be abstract
        """
        raise NotImplementedError("from_config method is not implemented for AttentionUNET")

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x
