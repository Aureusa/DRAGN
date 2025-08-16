from ._base_model import BaseModel
from .attention_unet import AttentionUNET
from .UNet import UNet
from .patchGAN import PatchGAN
from .WassersteinGAN import WGAN
from .UNet_CUT import UNet_CUT

AVAILABLE_MODELS = {
    "AttentionUNET": AttentionUNET,
    "PatchGAN": PatchGAN,
    "WGAN": WGAN,
    "UNet": UNet,
    "UNet_CUT": UNet_CUT,
}
