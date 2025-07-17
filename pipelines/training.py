import os
import glob

from data_pipeline import GalaxyDataset, FitsLoader
from data_pipeline.galaxy_container import GalaxyContainer
from model_training import Trainer
from networks.models import AVALAIBLE_MODELS
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
        **model_kwargs
    ) -> None:
    # Load training and validation data
    X_train, y_train = load_pkl_file(
        os.path.join(mockdata_filepath, "train_data.pkl")
    )
    X_val, y_val = load_pkl_file(
        os.path.join(mockdata_filepath, "val_data.pkl")
    )

    # Creating the datasets
    train_set = GalaxyDataset(
        X_train,
        y_train,
        training=True
    )
    val_set = GalaxyDataset(
        X_val,
        y_val,
        training=True
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
        DRAGNWarning(
            f"Models with similar filename signature already exist in `{data_folder}`: {existing_models}. "
            "The Trainer would proceed to fine-tune the model. If you want to start fresh erase the contents of"
            f" `{data_folder}` or chose a different `data_folder` argument."
        ).warn()

    # Load real data
    real_data = glob.glob("/scratch/s4683099/real_JWST/COSMOS-Web_cutouts_Zhuang2024/*.fits", recursive=True)
    real_container = GalaxyContainer(real_data[:16]) # Use only 16 samples for testing

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
        real_container=real_container
    )
