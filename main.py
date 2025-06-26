import os

from pipelines.pipelines import real_image_cleaner_pipeline
from data_pipeline import GalaxyDataset, FitsLoader


if __name__ == "__main__":
    attunet_l1_path = os.path.join(os.getcwd(), "data", "attunet_l1")
    attunet_l1_plus_weighted_l2_path = os.path.join(os.getcwd(), "data", "attunet_l1_plus_weighted_l2")
    attunet_mse_path = os.path.join(os.getcwd(), "data", "attunet_mse")
    attunet_mse_weighted_path = os.path.join(os.getcwd(), "data", "attunet_mse_weighted_squared")

    basic_unet_l1_path = os.path.join(os.getcwd(), "data", "basic_unet_l1")
    basic_unet_l1_plus_weighted_l2_path = os.path.join(os.getcwd(), "data", "basic_unet_l1_plus_weighted_l2")
    basic_unet_mse_path = os.path.join(os.getcwd(), "data", "basic_unet_mse")
    basic_unet_mse_weighted_path = os.path.join(os.getcwd(), "data", "basic_unet_mse_weighted_squared")
    basic_unet_rescaled_best_model = os.path.join(os.getcwd(), "data", "unet_rescaled")

    patch_gan_unet_path = os.path.join(os.getcwd(), "data", "patch_gan_unet", "checkpoint_2")
    patch_gan_unet_2_path = os.path.join(os.getcwd(), "data", "patch_gan_unet_2")

    real_image_cleaner_pipeline(
        model_names=["UNet (L1 + wL2)", "cGAN", "AttUNet (L2)"],
        model_types=["UNet", "UNet", "AttentionUNET"],
        model_filenames=["basic_unet_l1_plus_weighted_l2_best_model", "generator_patch_gan_unet_best_model", "attunet_mse_best_model"],
        data_folders=[basic_unet_l1_plus_weighted_l2_path, patch_gan_unet_path, attunet_mse_path],
        datasets=[GalaxyDataset, GalaxyDataset, GalaxyDataset],
        loaders=[FitsLoader, FitsLoader, FitsLoader],
        transforms=[None, None, None],
        n=136,
    )

    # Mock Data Analysis:
    # Model: UNet (L1 + wL2), Mean FRF: -0.5826477790695435
    # Model: cGAN, Mean FRF: -0.5956644206420112
    # Model: AttUNet (L2), Mean FRF: -0.13454654596769436

    # Real Data Analysis:
    # Model: UNet (L1 + wL2), Mean FRF: 0.0622
    # Model: cGAN, Mean FRF: -0.1305
    # Model: AttUNet (L2), Mean FRF: 0.1650