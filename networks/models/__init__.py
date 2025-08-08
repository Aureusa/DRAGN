from ._base_model import BaseModel
from .attention_unet import AttentionUNET
from .UNet import UNet
from .patchGAN import PatchGANUNet
from .WassersteinGAN import WGAN
from .UNet_CUT import UNet_CUT

AVALAIBLE_MODELS = {
    "AttentionUNET": AttentionUNET,
    "PatchGANUNet": PatchGANUNet,
    "WGAN": WGAN,
    "UNet": UNet,
    "UNet_CUT": UNet_CUT,
}
