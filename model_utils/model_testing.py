import os
import re
from tqdm import tqdm
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader

from astro_pipeline import (
    Photometry,
    StructuralParamsRegistry
)
from data_pipeline import (
    GalaxyDataset,
    TELESCOPES_DB,
    _BaseDataset,
    _BaseTransform,
    FitsLoader
)
from model import AVALAIBLE_MODELS
from model_utils.metrics import get_metrics
from model_utils.performance_analysis import PAdict
from loggers_utils import log_execution
from utils import (
    load_pkl_file,
    save_pkl_file,
    print_box
)
from utils_utils.device import get_device


MODELS_FOLDER = os.path.join("data", "saved_models")


class ModelTester:
    @log_execution("Initializing model tester...", "Model tester initialized!")
    def __init__(
            self,
            model_name: str,
            model_type: str,
            model_filename: str,
            data_folder: str,
            dataset: _BaseDataset = GalaxyDataset,
            transform: _BaseTransform|None = None, 
            loader: DataLoader = FitsLoader,
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
        self._device = get_device()

        # Initialize the model
        if model_type not in AVALAIBLE_MODELS:
            raise ValueError(f"Model type '{model_type}' is not supported.")
        else:
            self._model = AVALAIBLE_MODELS[model_type](*args, **kwargs)

        try:
            # Load the model
            self._model.load_state_dict(
                torch.load(
                    os.path.join(data_folder, f"{model_filename}.pth"),
                    map_location=self._device
                )
            )

            # Load the test and train data
            self._test_data_X, self._test_data_Y = load_pkl_file(
                os.path.join("data", "jwst_full_data", f"test_data.pkl")
            )
            self._train_data_X, self._train_data_Y = load_pkl_file(
                os.path.join("data", "jwst_full_data", f"train_data.pkl")
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
        
        # Hold the dataset and loader classes
        self._dataset = dataset
        self._loader = loader
        self._transform = transform
        self._data_folder = data_folder

    @log_execution("Performing photometry... (DEPRICATED)", "Photometry completed!")
    def do_photometry(
        self,
        test_set: bool = True,
        n: int = 10,
        batch_size: int = 1,
        num_workers: int = 1,
        prefetch_factor: int|None = None,
        shuffle: bool = False,
        **loader_kwargs
    ) -> tuple:
        info = f"Test data length: {len(self._test_data_X)}"
        print_box(info)

        if test_set:
            dataset = self._dataset(self._test_data_X[:n], self._test_data_Y[:n], transform=self._transform, training=False)
        else:
            dataset = self._dataset(self._train_data_X[:n], self._train_data_Y[:n], transform=self._transform, training=False)
        
        loader = self._loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            **loader_kwargs
        )

        # Create the photometry object
        photometry = Photometry()

        self._model.to(self._device)

        self._model.eval()
        
        input_registry = StructuralParamsRegistry("input_registry")
        target_registry = StructuralParamsRegistry("target_registry")
        cleaned_image_registry = StructuralParamsRegistry("cleaned_image_registry")
        with torch.no_grad():
            for inputs, targets, _ in tqdm(loader, desc="Performing photometry", unit="image"):
                inputs = inputs.to(self._device)
                cleaned_image = self._model(inputs)

                # Get properties of the images
                input_registry.extend(
                    photometry.perform_segmentation(
                        inputs, max_workers=45, registry_name="input_registry"
                    )
                )
                target_registry.extend(
                    photometry.perform_segmentation(
                        targets, max_workers=45, registry_name="target_registry"
                    )
                )
                cleaned_image_registry.extend(
                    photometry.perform_segmentation(
                        cleaned_image, max_workers=45, registry_name="cleaned_image_registry"
                    )
                )
                
        return input_registry, target_registry, cleaned_image_registry
    
    @log_execution("Testing model...", "Model testing completed!")
    def test_model(
        self,
        metrics: list[str],
        batch_size: int = 140,
        num_workers: int = 15,
        prefetch_factor: int = 14,
        shuffle: bool = True,
        verbose: bool = True,
        **loader_kwargs
    ) -> dict:
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
        evaluation_dict = self._test(
            self._test_data_X,
            self._test_data_Y,
            metrics,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            shuffle=shuffle,
            **loader_kwargs
        )

        if verbose:
            info = "Model evaluation completed.\n"
            for metric, losses in evaluation_dict.items():
                info += f"{metric}: {losses}\n"

            print_box(info)

        return evaluation_dict
    
    @log_execution("Performing performance analysis...", "Performance analysis completed!")
    def performance_analysis(
        self,
        metrics: list[str],
        batch_size: int = 140,
        num_workers: int = 15,
        prefetch_factor: int = 14,
        shuffle: bool = True,
        save: bool = True,
        verbose: bool = True,
        **loader_kwargs
    ) -> PAdict:
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
        for matches in tqdm(sorted_matches):
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

            shared_pattern = re.compile(r"sn(\d+)_.*?_(\d+)")
            y_index = {}
            for y in self._test_data_Y:
                match = shared_pattern.search(y)
                if match:
                    y_index[match.group(0)] = y

            corresponding_y_data = []
            for dat in filtered_data:
                match = shared_pattern.search(dat)
                if match:
                    full_match = match.group(0)
                    corresponding_element = y_index.get(full_match)
                    if corresponding_element:
                        corresponding_y_data.append(corresponding_element)

            # Evaluate the model on the filtered data
            evaluation_dict = self._test(
                filtered_data,
                corresponding_y_data,
                metrics,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                shuffle=shuffle,
                **loader_kwargs
            )

            # Store the results in the analysis dictionary
            analysis.add(fraction_key, count, evaluation_dict)

            # Free up memory
            del filtered_data, corresponding_y_data, evaluation_dict
            torch.cuda.empty_cache()

        if save:
            save_pkl_file(analysis, os.path.join(self._data_folder, f"{self._model_name}_performance_analysis.pkl"))
            print_box("Performance analysis saved successfully!")

        if verbose:
            print_box("Performance analysis:")
            print(analysis)
        return analysis
    
    @log_execution("Cleaning images...", "Image cleaning completed!")
    def clean_n_images(
            self,
            n: int,
            batch_size: int = 1,
            num_workers: int = 1,
            prefetch_factor: int|None = None,
            shuffle: bool = False,
            test_set: bool = True,
            f_agn: float|None = None,
            **loader_kwargs
        ) -> tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            ]:
        """
        Clean n images using the model. It uses the test dataset by default.
        If `test_set` is set to False, it uses the training dataset.

        :param n: Number of images to clean.
        :type n: int
        :param test_set: Whether to use the test dataset or the training dataset.
        :type test_set: bool
        :return: Tuple of numpy arrays containing the source images,
        target images, output images, the difference between the cleaned and
        source images (the diff images), and the injected psfs.
        The dimensions are (n, width, height) for the source, target, cleaned,
        diff, and psf arrays.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
    
        if test_set:
            dataset = self._dataset(self._test_data_X, self._test_data_Y, transform=self._transform, training=False)
        else:
            dataset = self._dataset(self._train_data_X, self._train_data_Y, transform=self._transform, training=False)
        
        if f_agn is not None:
            if not isinstance(f_agn, list):
                dataset.filter_by_f_agn(f_agn)
                dataset.filter_first_n(n) # Limit the dataset to n images
            elif isinstance(f_agn, list):
                dataset.filter_by_f_agn_list(f_agn)

        loader = self._loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            **loader_kwargs
        )

        self._model.to(self._device)

        self._model.eval()
        
        source_list = []
        target_list = []
        cleaned_image_list = []
        psf_predicted_list = []
        psf_list = []
        frf_list = []
        with torch.no_grad():
            num = 0
            if self._transform is not None:
                for inputs, targets, norm_params in tqdm(loader, desc="Cleaning images", unit="image"):
                    inputs = inputs.to(self._device)

                    cleaned_image = self._model(inputs)

                    inputs = inputs[0][0].cpu().detach().numpy()
                    cleaned_image = cleaned_image[0][0].cpu().detach().numpy()

                    inputs = self._transform.inverse(inputs, norm_params[0])
                    cleaned_image = self._transform.inverse(cleaned_image, norm_params[0])

                    psf_predicted = inputs - cleaned_image

                    # Append the psf fluxes to the lists
                    source_list.append(inputs)
                    target_list.append(targets[0][0].cpu().detach().numpy())
                    cleaned_image_list.append(cleaned_image)
                    psf_predicted_list.append(psf_predicted)
                    psf_list.append(psf[0][0].cpu().detach().numpy())
            else:
                for inputs, targets, _ in tqdm(loader, desc="Cleaning images", unit="image"):
                    (
                        source,
                        target,
                        cleaned_image,
                        psf_predicted,
                        psf
                    ) = self._make_autoregressive_pred(
                        self._model,
                        inputs,
                        targets
                    )
                    
                    source_list.append(source)
                    target_list.append(target)
                    cleaned_image_list.append(cleaned_image)
                    psf_predicted_list.append(psf_predicted)
                    psf_list.append(psf)
        
        source_arr = np.array(source_list)
        target_arr = np.array(target_list)
        cleaned_image_arr = np.array(cleaned_image_list)
        psf_predicted_arr = np.array(psf_predicted_list)
        psf_arr = np.array(psf_list)
        return source_arr, target_arr, cleaned_image_arr, psf_predicted_arr, psf_arr

    def _make_autoregressive_pred(
            self,
            model,
            inputs,
            targets
        ) -> tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray
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
        source images (the diff images), and the injected psfs.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        inputs = inputs.to(self._device)
        targets = targets.to(self._device)
        psf = inputs - targets
        cleaned_image = model(inputs)

        print(f"FRF: {cleaned_image.sum()/targets.sum()}; FRF PSF: {(inputs - cleaned_image).sum()/(psf.sum() + 1e-8)}")

        psf_predicted = inputs - cleaned_image
        psf_predicted = psf_predicted[0][0].cpu().detach().numpy()
        cleaned_image = cleaned_image[0][0].cpu().detach().numpy()
        source = inputs[0][0].cpu().detach().numpy()
        target = targets[0][0].cpu().detach().numpy()
        psf = psf[0][0].cpu().detach().numpy()
        return source, target, cleaned_image, psf_predicted, psf

    def compute_real_and_predicted_psf_fluxes(
            self,
            test_set: bool = True,
            batch_size: int = 300,
            shuffle: bool = True,
            num_workers: int = 14,
            prefetch_factor: int|None = 12,
            **loader_kwargs
        ) -> tuple[np.ndarray, np.ndarray]:
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
            dataset = self._dataset(self._test_data_X, self._test_data_Y, transform=self._transform, training=False)
        else:
            dataset = self._dataset(self._train_data_X, self._train_data_Y, transform=self._transform, training=False)
        
        loader = self._loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            **loader_kwargs
        )

        self._model.to(self._device)

        self._model.eval()
        
        real_psf_fluxes = []
        predicted_psf_fluxes = []
        real_gal_fluxes = []
        predicted_gal_fluxes = []
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="Evaluating"):
                inputs = inputs.to(self._device) # (B, 1, H, W)
                targets = targets.to(self._device)
                psf = inputs - targets

                cleaned_image = self._model(inputs)

                psf_predicted = inputs - cleaned_image

                (
                    real_psf_flux,
                    predicted_psf_flux,
                    real_gal_flux,
                    predicted_gal_flux
                ) = self._calculate_fluxes(
                    targets=targets,
                    psf=psf,
                    cleaned_image=cleaned_image,
                    psf_predicted=psf_predicted
                )

                # Append to lists
                real_psf_fluxes.extend(real_psf_flux)
                predicted_psf_fluxes.extend(predicted_psf_flux)
                real_gal_fluxes.extend(real_gal_flux)
                predicted_gal_fluxes.extend(predicted_gal_flux)

                # Free up memory
                del inputs, targets, psf, cleaned_image, psf_predicted
                torch.cuda.empty_cache()

        (
            real_psf_fluxes,
            predicted_psf_fluxes,
            real_gal_fluxes,
            predicted_gal_fluxes
        ) = self._sorted_fluxes(
            real_psf_fluxes=real_psf_fluxes,
            predicted_psf_fluxes=predicted_psf_fluxes,
            real_gal_fluxes=real_gal_fluxes,
            predicted_gal_fluxes=predicted_gal_fluxes
        )

        return real_psf_fluxes, predicted_psf_fluxes, real_gal_fluxes, predicted_gal_fluxes

    def _calculate_fluxes(self, targets, psf, cleaned_image, psf_predicted):
        # Calculate the psf fluxes
        real_psf_flux = psf.sum(dim=(1, 2, 3)) # (B, C, H, W) -> (B,)
        predicted_psf_flux = psf_predicted.sum(dim=(1, 2, 3))

        # Calculate the galaxy fluxes
        real_gal_flux = targets.sum(dim=(1, 2, 3)) # (B, C, H, W) -> (B,)
        predicted_gal_flux = cleaned_image.sum(dim=(1, 2, 3))

        # Convert the tensors to lists
        real_psf_flux = real_psf_flux.cpu().detach().tolist() # len(real_psf_flux) = B
        predicted_psf_flux = predicted_psf_flux.cpu().detach().tolist()
        real_gal_flux = real_gal_flux.cpu().detach().tolist()
        predicted_gal_flux = predicted_gal_flux.cpu().detach().tolist()

        return real_psf_flux, predicted_psf_flux, real_gal_flux, predicted_gal_flux
    
    def _sorted_fluxes(
            self,
            real_psf_fluxes,
            predicted_psf_fluxes,
            real_gal_fluxes,
            predicted_gal_fluxes
        ):
        # Convert the lists to numpy arrays
        real_psf_fluxes = np.array(real_psf_fluxes)
        predicted_psf_fluxes = np.array(predicted_psf_fluxes)
        real_gal_fluxes = np.array(real_gal_fluxes)
        predicted_gal_fluxes = np.array(predicted_gal_fluxes)

        # Sort the from the lowest to the highest
        sorted_indices = np.argsort(real_psf_fluxes)
        sorted_indeces_gal = np.argsort(real_gal_fluxes)

        real_psf_fluxes = real_psf_fluxes[sorted_indices]
        predicted_psf_fluxes = predicted_psf_fluxes[sorted_indices]

        real_gal_fluxes = real_gal_fluxes[sorted_indeces_gal]
        predicted_gal_fluxes = predicted_gal_fluxes[sorted_indeces_gal]

        return real_psf_fluxes, predicted_psf_fluxes, real_gal_fluxes, predicted_gal_fluxes

    def _test(
            self,
            x_data: list[str],
            y_data: list[str],
            metrics: list[str],
            batch_size: int = 140,
            num_workers: int = 15,
            prefetch_factor: int = 14,
            shuffle: bool = True,
            **loader_kwargs
        ) -> None:
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
        dataset = self._dataset(x_data, y_data, transform=self._transform, training=False)
        loader = self._loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            shuffle=shuffle,
            **loader_kwargs
        )

        self._model.to(self._device)

        self._model.eval()
        
        if self._transform is None:
            evaluation_dict = self._test_no_transform(loader, metrics)
        else:
            evaluation_dict = self._test_with_transform(loader, metrics)

        metrics_set = set(metrics)
        # Calculate the average loss for each metric
        for metric in evaluation_dict:
            if metric not in metrics_set:
                continue
            avg_loss = sum(evaluation_dict[metric]) / len(evaluation_dict[metric])
            evaluation_dict[metric] = avg_loss

        return evaluation_dict
    
    def _test_no_transform(self, loader: DataLoader, metrics: list[str]) -> None:
        evaluation_dict = {}

        real_psf_fluxes = []
        predicted_psf_fluxes = []
        real_gal_fluxes = []
        predicted_gal_fluxes = []
        with torch.no_grad():
            for inputs, targets, _ in tqdm(loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                psf = inputs - targets

                cleaned_image = self._model(inputs)

                psf_predicted = inputs - cleaned_image

                (
                    real_psf_flux,
                    predicted_psf_flux,
                    real_gal_flux,
                    predicted_gal_flux
                ) = self._calculate_fluxes(
                    targets=targets,
                    psf=psf,
                    cleaned_image=cleaned_image,
                    psf_predicted=psf_predicted
                )

                # Append to lists
                real_psf_fluxes.extend(real_psf_flux)
                predicted_psf_fluxes.extend(predicted_psf_flux)
                real_gal_fluxes.extend(real_gal_flux)
                predicted_gal_fluxes.extend(predicted_gal_flux)

                # Calculate the loss using the specified metrics
                for metric in metrics:
                    metric = metric.to(self._device)
                    loss = metric(inputs, cleaned_image, targets, psf)

                    # Store the loss in the evaluation dictionary
                    if str(metric) not in evaluation_dict:
                        evaluation_dict[str(metric)] = []

                    evaluation_dict[str(metric)].append(loss.item())

        (
            real_psf_fluxes,
            predicted_psf_fluxes,
            real_gal_fluxes,
            predicted_gal_fluxes
        ) = self._sorted_fluxes(
            real_psf_fluxes=real_psf_fluxes,
            predicted_psf_fluxes=predicted_psf_fluxes,
            real_gal_fluxes=real_gal_fluxes,
            predicted_gal_fluxes=predicted_gal_fluxes
        )

        evaluation_dict["real_psf_fluxes"] = real_psf_fluxes
        evaluation_dict["predicted_psf_fluxes"] = predicted_psf_fluxes
        evaluation_dict["real_gal_fluxes"] = real_gal_fluxes
        evaluation_dict["predicted_gal_fluxes"] = predicted_gal_fluxes

        return evaluation_dict
    
    def _test_with_transform(self, loader: DataLoader, metrics: list[str]) -> None:
        evaluation_dict = {}
        with torch.no_grad():
            for inputs, targets, norm_params in loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                psf = inputs - targets

                cleaned_image = self._model(inputs)

                # Inverse the normalization
                inputs = self._transform.batched_inverse(
                    inputs.squeeze(1).cpu().detach().numpy(), norm_params
                )
                cleaned_image = self._transform.batched_inverse(
                    cleaned_image.squeeze(1).cpu().detach().numpy(), norm_params
                )

                # Convert the inputs and cleaned_image back to tensors
                inputs = torch.tensor(inputs, device=self._device).unsqueeze(1)
                cleaned_image = torch.tensor(cleaned_image, device=self._device).unsqueeze(1)

                # Calculate the loss using the specified metrics
                for metric in metrics:
                    metric = metric.to(self._device)
                    loss = metric(inputs, cleaned_image, targets, psf)

                    # Store the loss in the evaluation dictionary
                    if str(metric) not in evaluation_dict:
                        evaluation_dict[str(metric)] = []

                    evaluation_dict[str(metric)].append(loss.item())
        return evaluation_dict
    