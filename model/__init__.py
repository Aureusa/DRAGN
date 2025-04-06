from data_pipeline import GalaxyDataset, FilepathGetter, create_source_target_pairs, test_train_val_split
from model.attention_unet import AttentionUNET
from model_utils.loss_functions import get_loss_function
from torch.utils.data import DataLoader
import pickle
import torch
import os
from utils import print_box
import matplotlib.pyplot as plt
import re
from matplotlib.colors import Normalize


def train_AttentionUNET_model(telescope: str, loss: str, loss_name: str, batch_size: int = 32, num_workers: int = 8):
    # Initialize the DataGetter class
    # data_getter = FilepathGetter(telescope)
    
    # files, _ = data_getter.get_data()

    # source, target = create_source_target_pairs(files)
    
    # info = "Sanity check 1"
    # info += f"\nSource: {source[0]}"
    # info += f"\nTarget: {target[0]}"
    # print_box(info)

    # info = "Sanity check 2"
    # info = f"\nSource: {source[23452]}"
    # info += f"\nTarget: {target[23452]}"
    # print_box(info)

    # info = "Sanity check 3"
    # info += f"\nSource: {source[23152]}"
    # info += f"\nTarget: {target[23152]}"
    # print_box(info)

    # X_train, X_val, X_test, y_train, y_val, y_test = test_train_val_split(
    #     source, target, test_size=0.2, val_size=0.1
    # )

    with open(os.path.join("data", "mse_constr_euclid", "train_data_mse_constr_euclid.pkl"), "rb") as f:
        X_train, y_train = pickle.load(f)

    with open(os.path.join("data", "mse_constr_euclid", "val_data_mse_constr_euclid.pkl"), "rb") as f:
        X_val, y_val = pickle.load(f)

    with open(os.path.join("data", "mse_constr_euclid", "test_data_mse_constr_euclid.pkl"), "rb") as f:
        X_test, y_test = pickle.load(f)
    
    train_dataset = GalaxyDataset(X_train, y_train)
    val_dataset = GalaxyDataset(X_val, y_val)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Rebase to the current dir
    os.chdir(os.path.expanduser("~"))
    os.chdir(os.path.join(os.getcwd(), "Deep-AGN-Clean"))

    # Save the dataset (X_train, y_train, etc.) instead of the DataLoader
    train_data_path = os.path.join("data", loss, f"train_data_{loss}.pkl")
    train_dir = os.path.dirname(train_data_path)

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    with open(train_data_path, "wb") as train_file:
        pickle.dump((X_train, y_train), train_file)

    val_data_path = os.path.join("data", loss, f"val_data_{loss}.pkl")
    val_dir = os.path.dirname(val_data_path)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    with open(val_data_path, "wb") as val_file:
        pickle.dump((X_val, y_val), val_file)

    test_data_path = os.path.join("data", loss, f"test_data_{loss}.pkl")
    test_dir = os.path.dirname(test_data_path)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    with open(test_data_path, "wb") as test_file:
        pickle.dump((X_test, y_test), test_file)

    info = "Train-Val-Test data failpaths saved successfully!\n"
    info += "The paths are:\n"
    info += f"Training data path: {train_data_path}\n"
    info += f"Validation data path: {val_data_path}\n"
    info += f"Testing data path: {test_data_path}"
    print_box(info)

    info = f"Ready for training! Current working path is: {os.getcwd()}"
    print_box(info)

    model = AttentionUNET()

    loss_function = get_loss_function(loss_name)

    model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        lr=0.001,
        loss_function=loss_function,
        num_epochs=50,
        checkpoint_save_name=f"{telescope}_{loss}"
    )

    torch.save(model.state_dict(), f"attention_unet_model_{telescope}_{loss}.pth")
    print_box(f"Model with {loss_name} saved successfully!")

    train_loss = model.train_loss
    val_loss = model.val_loss

    train_loss_path = os.path.join("data", loss, f"train_loss_{loss}.pkl")
    with open(train_loss_path, "wb") as test_file:
        pickle.dump(train_loss, test_file)

    info = f"Training Loss saved successfully in `{train_loss_path}`!"
    print_box(info)

    val_loss_path = os.path.join("data", loss, f"val_loss_{loss}.pkl")
    with open(val_loss_path, "wb") as val_file:
        pickle.dump(val_loss, val_file)

    info = f"Validation Loss saved successfully in `{val_loss_path}`!"
    print_box(info)


def test_AttentionUNET_model(name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_box(f"Using device {device}")
    # Load the model
    model = AttentionUNET()

    model.to(device)
    model.load_state_dict(torch.load(f"{name}.pth", map_location=device))

    # Load testing data
    with open(os.path.join("data", "mse_constr_euclid", "test_data_mse_constr_euclid.pkl"), "rb") as f:
        X_test, y_test = pickle.load(f)

    print_box(f"Whole test corpus:\nX: {len(X_test)}\nY:{len(y_test)}")

    x_sn = []
    y_sn = []

    pattern = "_sn067_"
    for idx, dat in enumerate(X_test):
        if re.search(pattern, dat):
            x_sn.append(dat)
            y_sn.append(y_test[idx])

    print_box(f"Retrieved {len(x_sn)} {pattern} images, with {len(x_sn)} targets!")
    print_box(f"Example: \n{x_sn[0]}\n{y_sn[0]}")

    test_dataset = GalaxyDataset(x_sn, y_sn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    print_box("Successfully created test loader!")

    count = 0
    # Plot test image and prediction
    # Plot test image and prediction
    for inputs, targets, psf in test_loader:
        inputs = inputs.to(device)

        cleaned_image = model(inputs)

        diff_predicted = inputs - cleaned_image
        diff_predicted = diff_predicted[0][0].cpu().detach().numpy()
        cleaned_image = cleaned_image[0][0].cpu().detach().numpy()
        source = inputs[0][0].cpu().detach().numpy()
        target = targets[0][0].cpu().detach().numpy()
        psf = psf[0][0].cpu().detach().numpy()

        # Make plot
        fig, ax = plt.subplots(1, 5, figsize=(15, 5))
        im0 = ax[0].imshow(source, cmap="gray")
        ax[0].set_title("Input Image")
        fig.colorbar(im0, ax=ax[0])

        im1 = ax[1].imshow(target, cmap="gray")
        ax[1].set_title("Target Image")
        fig.colorbar(im1, ax=ax[1])

        im2 = ax[2].imshow(cleaned_image, cmap="gray")
        ax[2].set_title("Cleaned Image")
        fig.colorbar(im2, ax=ax[2])

        im3 = ax[3].imshow(diff_predicted, cmap="gray")
        ax[3].set_title("Difference Image")
        fig.colorbar(im3, ax=ax[3])

        im4 = ax[4].imshow(psf, cmap="gray")
        ax[4].set_title("PSF Image")
        fig.colorbar(im4, ax=ax[4])

        plt.savefig(f"test_image_{count}.png")

        print_box(f"Image {count} saved successfully!")

        if count == 10:
            break

        count += 1
