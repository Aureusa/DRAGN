import torch
import torch.nn as nn
import torch.nn.functional as F


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