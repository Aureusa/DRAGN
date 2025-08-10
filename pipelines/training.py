import os
import glob

from data import GalaxyDataset, GalaxyDatasetPSFCond, FitsLoader
from data.galaxy_container import GalaxyContainer
from model_training import Trainer
from networks.models import AVAILABLE_MODELS
from utils import load_pkl_file

def training_pipeline(
        model_type,
        model_filename,
        data_folder,
        loss,
        batch_size,
        prefetch_factor,
        num_workers,
        lr,
        num_epochs,
        mockdata_filepath: str = os.path.join("testing_folder", "jwst_data"),
        condition_on_f_agn: bool = False,
        condition_on_psf: bool = False,
        training_GAN: bool = False,
        **model_kwargs
    ) -> None:
    # If condition_on_f_agn is True or condition_on_psf is True, 
    # ensure the model supports it by adding `in_channels=2` to the model_kwargs
    if condition_on_f_agn or condition_on_psf and model_type in AVAILABLE_MODELS:
        model_kwargs['in_channels'] = 2

    if condition_on_psf:
        dataset = GalaxyDatasetPSFCond
    else:
        dataset = GalaxyDataset

    # Load training and validation data
    X_train, y_train = load_pkl_file(
        os.path.join(mockdata_filepath, "train_data.pkl")
    )
    X_val, y_val = load_pkl_file(
        os.path.join(mockdata_filepath, "val_data.pkl")
    )

    # Creating the datasets
    if condition_on_psf:
        train_set = dataset(X_train, y_train, training=True)
        val_set = dataset(X_val, y_val, training=True)
    else:
        train_set = dataset(
            X_train,
            y_train,
            training=True,
            condition_on_f_agn=condition_on_f_agn
        )
        val_set = dataset(
            X_val,
            y_val,
            training=True,
            condition_on_f_agn=condition_on_f_agn
        )

    # Creating the data loaders
    train_loader = FitsLoader(
        train_set,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        shuffle=True
    )
    val_loader = FitsLoader(
        val_set,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        shuffle=False
    )

    # Check if model with similar filename signature already exists in data_folder
    existing_models = []
    if os.path.exists(data_folder):
        for filename in os.listdir(data_folder):
            if model_filename in filename and (filename.endswith('.pth') or filename.endswith('.pt')):
                existing_models.append(filename)
    
    if existing_models:
        from utils.warnings import DRAGNWarning
        DRAGNWarning().warn(
            f"Models with similar filename signature already exist in `{data_folder}`: {existing_models}. "
            "The Trainer would proceed to fine-tune the model. If you want to start fresh erase the contents of"
            f" `{data_folder}` or chose a different `data_folder` argument."
        )

    # Load real data
    real_data = glob.glob("/scratch/s4683099/real_JWST/COSMOS-Web_cutouts_Zhuang2024/*.fits", recursive=True)

    # Use only 16 samples for testing
    real_container = GalaxyContainer(
        real_data[:16],
        condition_on_f_agn=condition_on_f_agn,
        condition_on_psf=condition_on_psf
    )

    trainer = Trainer(
        model_type=model_type,
        model_filename=model_filename,
        data_folder=data_folder,
        train_loader=train_loader,
        val_loader=val_loader,
        **model_kwargs
    )

    trainer.train_model(
        loss_name=loss,
        lr=lr,
        num_epochs=num_epochs,
        real_container=real_container,
        training_GAN=training_GAN
    )
