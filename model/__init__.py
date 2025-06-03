from model.attention_unet import AttentionUNET
from model.cond_GAN import cGAN
from model.u_net import UNet
from model.cond_GAN_UNet import cGAN_UNet
from model.tripple_cGAN import tripple_cGAN
from model.unetr_cGAN import unetr_cGAN, UNetR
from model.patch_GAN import PatchGAN
from model.patch_GAN_UNet import PatchGANUNet


AVALAIBLE_MODELS = {
    "AttentionUNET": AttentionUNET,
    "cGAN": cGAN,
    "PatchGAN": PatchGAN,
    "PatchGANUNet": PatchGANUNet,
    "UNet": UNet,
    "cGAN_UNet": cGAN_UNet,
    "tripple_cGAN": tripple_cGAN,
    "unetr_cGAN": unetr_cGAN,
    "UNetR": UNetR,
}
