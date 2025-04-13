from data_pipeline import GalaxyDataset, FilepathGetter, create_source_target_pairs, test_train_val_split
from data_pipeline.getter import TELESCOPES_DB
from model.attention_unet import AttentionUNET
from model_utils.loss_functions import get_loss_function
from model_utils.metrics import get_metrics
from torch.utils.data import DataLoader
from collections import Counter
import pickle
import torch
import os
from utils import print_box
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from astropy.visualization import AsinhStretch, ImageNormalize



MODELS_FOLDER = os.path.join("data", "saved_models")


class ModelTester:
    def __init__(self, model_type: str, model_name: str, data_folder: str, *args, **kwargs):
        self._model_name = model_name

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        models = {
            "AttentionUNET": AttentionUNET,
            # Add other models here if needed
        }

        if model_type not in models:
            raise ValueError(f"Model type '{model_type}' is not supported.")
        
        self._model = models[model_type](*args, **kwargs)

        try:
            self._model.load_state_dict(torch.load(os.path.join(MODELS_FOLDER, f"{model_name}.pth"), map_location=self._device))
            
            with open(os.path.join("data", data_folder, f"test_data.pkl"), "rb") as file:
                self._test_data_X, self._test_data_Y = pickle.load(file)

            with open(os.path.join("data", data_folder, f"train_data.pkl"), "rb") as file:
                self._train_data_X, self._train_data_Y = pickle.load(file)

            # with open(os.path.join("data", data_folder, f"train_loss.pkl"), "rb") as file:
            #     self._train_loss = pickle.load(file)

            with open(os.path.join("data", data_folder, f"val_loss.pkl"), "rb") as file:
                self._val_loss = pickle.load(file)

            info = "Model Tester initialized.\n"
            info += f"Model type: {model_type}\n"
            info += f"Model name: {model_name}\n"
            info += f"Test data folder: {data_folder}\n"
            info += f"Test data length: {len(self._test_data_X)}\n"
            info += f"Device: {self._device}"
            print_box(info)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {e.filename}. Please check the file path.") from e
        
    def test_model(self, metrics: list[str]) -> None:
        """
        Test the model on the test dataset.
        """
        evaluation_dict = self._test(self._test_data_X, self._test_data_Y, metrics)

        info = "Model evaluation completed.\n"
        for metric, losses in evaluation_dict.items():
            avg_loss = sum(losses) / len(losses)

            info += f"Average {metric}: {avg_loss}\n"

        print_box(info)

        return evaluation_dict
    
    def performance_analysis(self, metrics: list[str], save: bool = True) -> None:
        agn_fraction_pattern = TELESCOPES_DB["AGN FRACTION PATTERN"]

        pattern = re.compile(agn_fraction_pattern)
        matches = (f"_f{match.group(1)}" for item in self._test_data_X for match in [pattern.search(item)] if match)
        
        # Count occurrences of each match
        match_counts = Counter(matches)

        # Sort matches by their counts in descending order
        sorted_matches = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)

        analysis = {}
        for matches in sorted_matches:
            match, count = matches
            match_pattern = re.compile(match)
            filtered_data = [item for item in self._test_data_X if match_pattern.search(item)]

            shared_pattern = re.compile(r"sn(\d+)_.*?_(\d+)")
            
            corresponding_y_data = []

            Y_data_set = set(self._test_data_Y)
            for dat in filtered_data:
                match = re.search(shared_pattern, dat)
                if match:
                    full_match = match.group(0)
                    # Find the element in Y_data_set that contains the full match
                    corresponding_element = next((y for y in Y_data_set if full_match in y), None)
                    if corresponding_element:
                        corresponding_y_data.append(corresponding_element)

            evaluation_dict = self._test(filtered_data, corresponding_y_data, metrics)

            if match not in analysis:
                analysis[match] = {}
                analysis[match]["count"] = count
                analysis[match]["evaluation"] = evaluation_dict
        
        if save:
            with open(os.path.join("data", f"{self._model_name}_performance_analysis.pkl"), "wb") as file:
                pickle.dump(analysis, file)

            info = "Performance analysis saved successfully!"
            print_box(info)

        return analysis

    def plot_loss(self) -> None:
        """
        Plot the training and validation loss.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self._train_loss, label="Training Loss")
        plt.plot(self._val_loss, label="Validation Loss")
        plt.title("Loss over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{self._model_name}_loss.png")

    def clean_n_images(self, n: int) -> None:
        # Create test dataset and dataloader
        test_dataset = GalaxyDataset(self._test_data_X, self._test_data_Y)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        self._model.to(self._device)

        self._model.eval()
        
        with torch.no_grad():
            print_box("Successfully created test loader!")

            count = 0
            # Plot test image and prediction
            for inputs, targets, psf in tqdm(test_loader, desc="Evaluating"):
                inputs = inputs.to(self._device)

                cleaned_image = self._model(inputs)

                diff_predicted = inputs - cleaned_image
                diff_predicted = diff_predicted[0][0].cpu().detach().numpy()
                cleaned_image = cleaned_image[0][0].cpu().detach().numpy()
                source = inputs[0][0].cpu().detach().numpy()
                target = targets[0][0].cpu().detach().numpy()
                psf = psf[0][0].cpu().detach().numpy()

                # Normalize the images
                norm_source = ImageNormalize(source, stretch=AsinhStretch())
                norm_target = ImageNormalize(target, stretch=AsinhStretch())
                norm_cleaned_image = ImageNormalize(cleaned_image, stretch=AsinhStretch())
                norm_diff_predicted = ImageNormalize(diff_predicted, stretch=AsinhStretch())
                norm_psf = ImageNormalize(psf, stretch=AsinhStretch())

                # Make plot
                fig, ax = plt.subplots(1, 5, figsize=(15, 5))
                im0 = ax[0].imshow(source, norm=norm_source, cmap="gray")
                ax[0].set_title("Input Image")
                fig.colorbar(im0, ax=ax[0])

                im1 = ax[1].imshow(target, norm=norm_target, cmap="gray")
                ax[1].set_title("Target Image")
                fig.colorbar(im1, ax=ax[1])

                im2 = ax[2].imshow(cleaned_image, norm=norm_cleaned_image, cmap="gray")
                ax[2].set_title("Cleaned Image")
                fig.colorbar(im2, ax=ax[2])

                im3 = ax[3].imshow(diff_predicted, norm=norm_diff_predicted, cmap="gray")
                ax[3].set_title("Difference Image")
                fig.colorbar(im3, ax=ax[3])

                im4 = ax[4].imshow(psf, norm=norm_psf, cmap="gray")
                ax[4].set_title("PSF Image")
                fig.colorbar(im4, ax=ax[4])

                plt.savefig(f"test_image_{count}.png")

                print_box(f"Image {count} saved successfully!")

                if count == n:
                    break

                count += 1

    def _test(self, x_data: list[str], y_data: list[str], metrics: list[str]) -> None:
        """
        Test the model on the test dataset.
        """
        # Create test dataset and dataloader
        test_dataset = GalaxyDataset(x_data, y_data)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        # Get the metrics
        metrics = get_metrics(metrics)

        self._model.to(self._device)

        self._model.eval()
        
        evaluation_dict = {}
        with torch.no_grad():
            print_box("Successfully created test loader!")

            for inputs, targets, psf in test_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                psf = psf.to(self._device)

                cleaned_image = self._model(inputs)

                # Calculate the loss using the specified metrics
                for metric in metrics:
                    loss = metric(inputs, cleaned_image, targets, psf)

                    # Store the loss in the evaluation dictionary
                    if str(metric) not in evaluation_dict:
                        evaluation_dict[str(metric)] = []

                    evaluation_dict[str(metric)].append(loss)

        return evaluation_dict
    