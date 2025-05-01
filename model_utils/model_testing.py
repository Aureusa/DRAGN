import os
import re
from tqdm import tqdm
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader
from astropy.visualization import AsinhStretch, ImageNormalize

from data_pipeline import GalaxyDataset
from data_pipeline.getter import TELESCOPES_DB
from model import AVALAIBLE_MODELS
from model_utils.metrics import get_metrics
from model_utils.performance_analysis import PAdict
from utils import load_pkl_file, save_pkl_file, print_box


MODELS_FOLDER = os.path.join("data", "saved_models")


class ModelTester:
    def __init__(
            self,
            model_name: str,
            model_type: str,
            model_filename: str,
            data_folder: str,
            *args,
            **kwargs
        ) -> None:
        """
        Initialize the ModelTester class.

        :param model_name: Name of the model, used for saving the performance analysis.
        :type model_name: str
        :param model_type: Type of the model.
        :type model_type: str
        :param model_filename: Filename of the model, used for loading (exclude .pth).
        :type model_filename: str
        :param data_folder: Folder containing the test and train data.
        :type data_folder: str
        :param args: Additional arguments for the model.
        :param kwargs: Additional keyword arguments for the model.
        """
        self._model_name = model_name
        self._model_type = model_type

        # Initialize the device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the model
        if model_type not in AVALAIBLE_MODELS:
            raise ValueError(f"Model type '{model_type}' is not supported.")
        else:
            self._model = AVALAIBLE_MODELS[model_type](*args, **kwargs)

        try:
            # Load the model
            self._model.load_state_dict(
                torch.load(
                    os.path.join(MODELS_FOLDER, f"{model_filename}.pth"),
                    map_location=self._device
                )
            )

            # Load the test and train data
            self._test_data_X, self._test_data_Y = load_pkl_file(
                os.path.join("data", data_folder, f"test_data.pkl")
            )
            self._train_data_X, self._train_data_Y = load_pkl_file(
                os.path.join("data", data_folder, f"train_data.pkl")
            )

            info = "Model Tester initialized.\n"
            info += f"Model type: {model_type}\n"
            info += f"Model name: {model_name}\n"
            info += f"Test data folder: {data_folder}\n"
            info += f"Test data length: {len(self._test_data_X)}\n"
            info += f"Device: {self._device}"
            print_box(info)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {e.filename}. Please check the file path.") from e

    def remove_common_images(self) -> None: # Depricated
        """
        Depricated. Removes common images between the test and train datasets.
        """
        common_images = set(self._test_data_Y) & set(self._train_data_Y)

        # Use tqdm to track progress while filtering out common images
        filtered_data = [
            (x, y) for x, y in 
                zip(self._test_data_X, self._test_data_Y)
                if y not in common_images
        ]

        # Unzip the filtered data back into self._test_data_X and self._test_data_Y
        test_data_X, test_data_Y = zip(*filtered_data)

        # Convert back to lists (zip returns tuples)
        test_data_X = list(test_data_X)
        test_data_Y = list(test_data_Y)

        info = f"Filtered {len(common_images)} common images from test data.\n"
        info += f"Test data length before filtering: {len(self._test_data_Y)}\n"
        info += f"Test data length after filtering: {len(test_data_Y)}\n"
        info += f"Test data X: {len(test_data_X)}\n"
        print_box(info)

        self._test_data_X = test_data_X
        self._test_data_Y = test_data_Y

        print_box("Common images removed successfully!")
        
    def test_model(self, metrics: list[str], verbose: bool = True) -> dict:
        """
        Tests the loaded model on the test dataset.

        :param metrics: List of metrics to evaluate the model on.
        :type metrics: list[str]
        :return: Dictionary of evaluation metrics.
        :rtype: dict
        :param verbose: Whether to print the evaluation results.
        :type verbose: bool
        :return: Dictionary of evaluation metrics. The raw dict structure is:
        {
        metric: loss,
        ...
        }
        :rtype: dict
        """
        evaluation_dict = self._test(self._test_data_X, self._test_data_Y, metrics)

        if verbose:
            info = "Model evaluation completed.\n"
            for metric, losses in evaluation_dict.items():
                info += f"{metric}: {losses}\n"

            print_box(info)

        return evaluation_dict
    
    def performance_analysis(self, metrics: list[str], save: bool = True, verbose: bool = True) -> PAdict:
        """
        Perform a performance analysis of the model on the test dataset. It
        separates the data based on the AGN fraction and evaluates the model on each
        subset.

        :param metrics: List of metrics to evaluate the model on.
        :type metrics: list[str]
        :param save: Whether to save the performance analysis to a file.
        If true dumps the analysis to a pickle file in the `data` folder.
        :type save: bool
        :param verbose: Whether to print the performance analysis.
        :type verbose: bool
        :return: Dictionary of performance analysis. Its raw dict structure is:
        {
        f = ?*: {
            "count": ?,
            "evaluation": {
                metric: loss,
                ...}
            },
        ...
        }
        :rtype: PAdict
        """
        # Get the metrics
        metrics = get_metrics(metrics)

        # Get the AGN fraction pattern from the database
        agn_fraction_pattern = TELESCOPES_DB["AGN FRACTION PATTERN"]

        # Find all matches of the AGN fraction pattern in the test data
        pattern = re.compile(agn_fraction_pattern)
        matches = (
            f"_f{match.group(1)}"
            for item in self._test_data_X
            for match in [pattern.search(item)]
            if match
        )

        # Count occurrences of each match
        match_counts = Counter(matches)

        # Sort matches by their counts in descending order
        sorted_matches = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)

        analysis = PAdict(self._model_name)
        for matches in sorted_matches:
            agn_match, count = matches # Unpack the tuple

            # Extract the AGN fraction from the match that is going to
            # be the key for the analysis dictionary
            agn_frac = re.search(r'_f(\d+)', agn_match)
            agn_frac_num = agn_frac.group(1)
            fraction_key = f"f_agn = 0.{agn_frac_num}"

            print_box(f"Performing analysis for {fraction_key}...")
            
            # Filter the X data based on the AGN fraction
            # `filtered_data` contains the filenames of the images that match the AGN fraction
            # from the test data
            match_pattern = re.compile(agn_match)
            filtered_data = [item for item in self._test_data_X if match_pattern.search(item)]

            # Define the pattern to extract the shared part of the filenames 
            # between the X data and the Y data (e.g., sn1234_..._5678)
            shared_pattern = re.compile(r"sn(\d+)_.*?_(\d+)")
            
            # This list represents the Y data that corresponds to each X data
            # stored in the `filtered_data` list. Effectively creating a 1:1 mapping
            # between the X data and the Y data essential for the loading logic in the loader.
            corresponding_y_data = []

            Y_data_set = set(self._test_data_Y)
            for dat in filtered_data: # Loop through all the X data and find its corresponding Y data
                match = re.search(shared_pattern, dat)
                if match:
                    full_match = match.group(0)

                    # Find the element in Y_data_set that contains the full match
                    corresponding_element = next(
                        (y for y in Y_data_set if full_match in y),
                        None
                    )
                    if corresponding_element:
                        corresponding_y_data.append(corresponding_element)

            # Evaluate the model on the filtered data
            evaluation_dict = self._test(filtered_data, corresponding_y_data, metrics)

            # Store the results in the analysis dictionary
            analysis.add(fraction_key, count, evaluation_dict)

            # Free up memory
            del filtered_data, corresponding_y_data, evaluation_dict
            torch.cuda.empty_cache()

        if save:
            save_pkl_file(analysis, os.path.join("data", f"{self._model_name}_performance_analysis.pkl"))
            print_box("Performance analysis saved successfully!")

        if verbose:
            print_box("Performance analysis:")
            print(analysis)
        return analysis
    
    def clean_n_images(
            self,
            n: int,
            test_set: bool = True
        ) -> tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            list[ImageNormalize]
            ]:
        """
        Clean n images using the model. It uses the test dataset by default.
        If `test_set` is set to False, it uses the training dataset.

        :param n: Number of images to clean.
        :type n: int
        :param test_set: Whether to use the test dataset or the training dataset.
        :type test_set: bool
        :return: Tuple of numpy arrays containing the source images,
        target images, cleaned images , the difference between the cleaned and
        source images (the diff images), the injected psfs, list containing the
        normalization of the images (an ImageNormalize object from astropy.visualization).
        The dimensions are (n, width, height) for the source, target, cleaned,
        diff, and psf arrays. The normalization list contains n ImageNormalize
        objects, one for each image.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[ImageNormalize]]
        """
        if test_set:
            dataset = GalaxyDataset(self._test_data_X, self._test_data_Y)
        else:
            dataset = GalaxyDataset(self._train_data_X, self._train_data_Y)
        
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        self._model.to(self._device)

        self._model.eval()
        
        source_list = []
        target_list = []
        cleaned_image_list = []
        diff_predicted_list = []
        psf_list = []
        norm_list = []
        with torch.no_grad():
            for inputs, targets, psf in tqdm(loader[:n], desc="Cleaning images", unit="image"):
                (
                    source,
                    target,
                    cleaned_image,
                    diff_predicted,
                    psf,
                    norm
                ) = self._make_autoregressive_pred(
                    self._model,
                    inputs,
                    targets,
                    psf
                )

                source_list.append(source)
                target_list.append(target)
                cleaned_image_list.append(cleaned_image)
                diff_predicted_list.append(diff_predicted)
                psf_list.append(psf)
                norm_list.append(norm)
        
        source_arr = np.array(source_list)
        target_arr = np.array(target_list)
        cleaned_image_arr = np.array(cleaned_image_list)
        diff_predicted_arr = np.array(diff_predicted_list)
        psf_arr = np.array(psf_list)
        return source_arr, target_arr, cleaned_image_arr, diff_predicted_arr, psf_arr, norm_list

    def _make_autoregressive_pred(
            self,
            model,
            inputs,
            targets,
            psf
        ) -> tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            ImageNormalize
            ]:
        """
        Make an autoregressive prediction using the model.
        
        :param model: The model to use for prediction.
        :type model: torch.nn.Module
        :param inputs: The input images.
        :type inputs: torch.Tensor
        :param targets: The target images.
        :type targets: torch.Tensor
        :param psf: The point spread function (PSF) images.
        :type psf: torch.Tensor
        :return: Tuple of numpy arrays containing the source images,
        target images, cleaned images, the difference between the cleaned and
        source images (the diff images), the injected psfs, list containing the
        normalization of the images (an ImageNormalize object from astropy.visualization).
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ImageNormalize]
        """
        inputs = inputs.to(self._device)

        cleaned_image = model(inputs)

        diff_predicted = inputs - cleaned_image
        diff_predicted = diff_predicted[0][0].cpu().detach().numpy()
        cleaned_image = cleaned_image[0][0].cpu().detach().numpy()
        source = inputs[0][0].cpu().detach().numpy()
        target = targets[0][0].cpu().detach().numpy()
        psf = psf[0][0].cpu().detach().numpy()
        
        # Normalize the images
        norm = ImageNormalize(target, stretch=AsinhStretch())
        return source, target, cleaned_image, diff_predicted, psf, norm

    def compute_real_and_predicted_psf_fluxes(self, test_set: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the real and predicted PSF fluxes for the test dataset.

        :param test_set: Whether to use the test dataset or the training dataset.
        :type test_set: bool
        :return: Tuple of numpy arrays containing the real PSF fluxes and
        predicted PSF fluxes. The dimensions are (n,). The arrays are sorted
        from the lowest to the highest PSF fluxes.
        The real PSF fluxes are the sum of the PSF images, and the predicted
        PSF fluxes are the sum of the difference between the source images
        and the cleaned images (difference image).
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        if test_set:
            dataset = GalaxyDataset(self._test_data_X, self._test_data_Y)
        else:
            dataset = GalaxyDataset(self._train_data_X, self._train_data_Y)
        loader = DataLoader(
            dataset=dataset,
            batch_size=300,
            num_workers=14,
            prefetch_factor=12,
            shuffle=True
        )

        self._model.to(self._device)

        self._model.eval()
        
        real_psf_fluxes = []
        predicted_psf_fluxes = []
        with torch.no_grad():
            for inputs, targets, psf in tqdm(loader, desc="Evaluating"):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                psf = psf.to(self._device)

                cleaned_image = self._model(inputs)

                diff_predicted = inputs - cleaned_image

                # Calculate the psf fluxes
                real_psf_flux = psf.sum().item()
                predicted_psf_flux = diff_predicted.sum().item()

                # Append the psf fluxes to the lists
                real_psf_fluxes.append(real_psf_flux)
                predicted_psf_fluxes.append(predicted_psf_flux)

                # Free up memory
                del inputs, targets, psf, cleaned_image, diff_predicted
                torch.cuda.empty_cache()

        # Convert the lists to numpy arrays
        real_psf_fluxes = np.array(real_psf_fluxes)
        predicted_psf_fluxes = np.array(predicted_psf_fluxes)

        # Sort the real_psf_fluxes and predicted_psf_fluxes from the lowest to the highest
        sorted_indices = np.argsort(real_psf_fluxes)

        real_psf_fluxes = real_psf_fluxes[sorted_indices]
        predicted_psf_fluxes = predicted_psf_fluxes[sorted_indices]

        return real_psf_fluxes, predicted_psf_fluxes

    def _test(self, x_data: list[str], y_data: list[str], metrics: list[str]) -> None:
        """
        Test the model on the given data.

        :param x_data: List of input images (filepaths).
        :type x_data: list[str]
        :param y_data: List of target images (filepaths).
        :type y_data: list[str]
        :param metrics: List of metrics to evaluate the model on. Must be
        available in the `get_metrics` function.
        :type metrics: list[str]
        :return: Dictionary of evaluation metrics. The raw dict structure is:
        {
        metric: loss,
        ...
        }
        :rtype: dict
        """
        # Create test dataset and dataloader
        dataset = GalaxyDataset(x_data, y_data)
        loader = DataLoader(dataset=dataset, batch_size=100, num_workers=12, prefetch_factor=6, shuffle=True)

        self._model.to(self._device)

        self._model.eval()
        
        evaluation_dict = {}
        with torch.no_grad():
            for inputs, targets, psf in loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                psf = psf.to(self._device)

                cleaned_image = self._model(inputs)

                # Calculate the loss using the specified metrics
                for metric in metrics:
                    metric = metric.to(self._device)
                    loss = metric(inputs, cleaned_image, targets, psf)

                    # Store the loss in the evaluation dictionary
                    if str(metric) not in evaluation_dict:
                        evaluation_dict[str(metric)] = []

                    evaluation_dict[str(metric)].append(loss.item())

        # Calculate the average loss for each metric
        for metric in evaluation_dict:
            avg_loss = sum(evaluation_dict[metric]) / len(evaluation_dict[metric])
            evaluation_dict[metric] = avg_loss

        return evaluation_dict
    