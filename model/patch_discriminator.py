import os
import torch
import torch.nn as nn
import torch.nn.functional as F


from model._base_model import BaseModel
from utils import print_box


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=True, dropout=0.1, norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1 if not downsample else 2,
                padding=1
                )
        ]

        if norm:
            layers.append(nn.InstanceNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttnBlock, self).__init__()
        # Create query, key, value linear transformations
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Lernable parameters starting from 0 for the attention map,
        # encourages the network to learn the attention map gradually
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        # Calculate the query and key matrices
        proj_query = self.query_conv(x).view(B, -1, H * W) # (B,C,H,W)x(H*W) -> (B, C//8, N)
        proj_key = self.key_conv(x).view(B, -1, H * W) # (B,C,H,W)x(H*W) -> (B, C//8, N)

        # Calculate the attention map QK^T
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key) # (B, N, C//8) x (B, C//8, N) -> (B, N, N)

        # Normalize the attention map
        attention = F.softmax(energy, dim=-1) # (B, N, N)

        # Calculate the value matrix
        proj_value = self.value_conv(x).view(B, -1, H * W) # (B,C,H,W)x(H*W) -> (B, C, N)

        # Calculate the weighted sum of the value matrix and the attention map
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # (B, C, N) x (B, N, N) -> (B, C, N)

        # Reshape the output to the original input shape
        out = out.view(B, C, H, W) # (B, C, N) -> (B, C, H, W)

        # Apply the learnable parameter to the output
        # and add it to the original input
        out = self.gamma * out + x # (B, C, H, W) + (B, C, H, W)
        return out
    

class ConvBlockWithAttention(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
        ):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, False)
        self.attn = AttnBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.attn(x)
        return x
    

class PatchDiscriminatorWithAttention(nn.Module, BaseModel):
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

    def forward(self, x):
        # Pass the input through the first convolutional block
        x = self.first_conv(x)

        # Pass the input through the convolutional blocks with attention
        for block in self.conv_blocks:
            x = block(x)

        # Pass the output through the final convolutional layer
        x = self.final_conv(x)
        return x
    
    def train_model(self):
        """
        Train the model.
        """
        pass
    
    def save_model(self, name: str, data_path: str):
        """
        Save the model to a file.
        
        :param name: Name of the file to save the model to.
        :type name: str
        """
        path = os.path.join(data_path, f"{name}.pth")
        torch.save(self.state_dict(), path)

        info = f"Model `{name}.pth` saved successfully!"
        info += f"Path to model: {data_path}"
        print_box(info)

    def load_model(self, filename: str,  dir_ = "Default"):
        """
        Load the model from a file.
        
        :param filename: Name of the file to load the model from.
        :type name: str
        :param dir_: Directory to load the model from.
        :type dir_: str
        """
        if dir_ == "Default":
            dir_ = os.path.join(os.getcwd(), "data", "saved_models")

        self.load_state_dict(torch.load(os.path.join(dir_, f"{filename}.pth"), map_location=torch.device('cpu')))

        info = f"Model `{filename}` loaded successfully!"
        print_box(info)


class PatchDiscriminator(nn.Module, BaseModel):
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

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x

    def train_model(self):
        pass

    def save_model(self, name: str, data_path: str):
        path = os.path.join(data_path, f"{name}.pth")
        torch.save(self.state_dict(), path)
        print_box(f"Model `{name}.pth` saved successfully!\nPath to model: {data_path}")

    def load_model(self, filename: str, dir_="Default"):
        if dir_ == "Default":
            dir_ = os.path.join(os.getcwd(), "data", "saved_models")
        self.load_state_dict(torch.load(os.path.join(dir_, f"{filename}.pth"), map_location=torch.device('cpu')))
        print_box(f"Model `{filename}` loaded successfully!")
    