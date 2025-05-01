from model.attention_unet import AttentionUNET
from model.cond_GAN import cGAN
from model.u_net import UNet
from model.cond_GAN_UNet import cGAN_UNet
from model.tripple_cGAN import tripple_cGAN


AVALAIBLE_MODELS = {
    "AttentionUNET": AttentionUNET,
    "cGAN": cGAN,
    "UNet": UNet,
    "cGAN_UNet": cGAN_UNet,
    "tripple_cGAN": tripple_cGAN,
}
