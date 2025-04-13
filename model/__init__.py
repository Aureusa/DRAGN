from model.attention_unet import AttentionUNET
from model.cond_GAN import cGAN
from model.u_net import UNet


AVALAIBLE_MODELS = {
    "AttentionUNET": AttentionUNET,
    "cGAN": cGAN,
    "UNet": UNet,
}
