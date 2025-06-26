from .attention_unet import AttentionUNET
from .UNet import UNet
from .patchGAN import PatchGANUNet

AVALAIBLE_MODELS = {
    "AttentionUNET": AttentionUNET,
    "PatchGANUNet": PatchGANUNet,
    "UNet": UNet,
}
