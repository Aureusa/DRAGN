from model.attention_unet import AttentionUNET
from model.cond_GAN import cGAN
from model.u_net import UNet
from model.msgdd_cGAN import MSGDD_cGAN


AVALAIBLE_MODELS = {
    "AttentionUNET": AttentionUNET,
    "cGAN": cGAN,
    "UNet": UNet,
    "MSGDD_cGAN": MSGDD_cGAN,
}
