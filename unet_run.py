import os

from pipelines.training import training_pipeline


if __name__ == "__main__":
    training_pipeline(
        model_type="UNet",
        model_filename="unet_model",
        data_folder=os.path.join("testing_folder", "unet"),
        loss="L1 + Weighted L2 Loss",
        batch_size=220,
        prefetch_factor=18,
        num_workers=32,
        lr=0.001,
        num_epochs=50,
    )