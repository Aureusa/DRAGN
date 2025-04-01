from data_pipeline import GalaxyDataset, FilepathGetter, create_source_target_pairs, test_train_val_split
from model.attention_unet import AttentionUNET
from model_utils.loss_functions import get_loss_function
from torch.utils.data import DataLoader
import pickle
import torch


if __name__ == "__main__":
    # Initialize the DataGetter class
    data_getter = FilepathGetter()
    
    # Print the found .fits files
    files, keys = data_getter.get_data()

    source, target = create_source_target_pairs(files)

    X_train, X_val, X_test, y_train, y_val, y_test = test_train_val_split(
        source, target, test_size=0.2, val_size=0.1
    )
    
    train_dataset = GalaxyDataset(X_train, y_train)
    val_dataset = GalaxyDataset(X_val, y_val)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False)

    # Save the dataset (X_train, y_train, etc.) instead of the DataLoader
    with open("train_data.pkl", "wb") as train_file:
        pickle.dump((X_train, y_train), train_file)

    with open("val_data.pkl", "wb") as val_file:
        pickle.dump((X_val, y_val), val_file)

    model = AttentionUNET()

    loss_function = get_loss_function("PSF-Constrained MSE Loss")

    model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        lr=0.001,
        loss_function=loss_function,
        num_epochs=10
    )

    torch.save(model.state_dict(), "attention_unet_model_euclid.pth")
    print("Model saved successfully!")
    