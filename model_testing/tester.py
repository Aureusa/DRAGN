import os
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch

from astro_pipeline import (
    Photometry,
    StructuralParamsRegistry
)
from data_pipeline import (
    _BaseTransform,
    _BaseLoader,
)
from networks.models import AVALAIBLE_MODELS
from model_testing.metrics import get_metrics
from model_testing.performance_analysis import PAdict
from loggers_utils import log_execution
from utils import (
    save_pkl_file,
    print_box
)
from utils_utils.device import get_device


MODELS_FOLDER = os.path.join("data", "saved_models")


class Tester:
    @log_execution("Initializing model tester...", "Model tester initialized!")
    def __init__(
            self,
            model_type: str,
            model_filename: str,
            data_folder: str,
            test_loader: _BaseLoader,
            transform: _BaseTransform|None = None,
            **kwargs
        ) -> None:
        """
        Initialize the Tester class.

        :param model_type: The type of model to test.
        :type model_type: str
        :param model_filename: The filename of the model to test.
        The model will be loaded from `data_folder/model_filename.pth`,
        you do not need to include `.pth` in the filename.
        :type model_filename: str
        :param data_folder: The folder to save data such as
        test results, and performance analysis. This must be the folder where
        the model weights are saved!
        :type data_folder: str
        :param test_loader: DataLoader for the test dataset.
        :type test_loader: torch.utils.data.DataLoader
        :param kwargs: Additional arguments to pass to the model.
        :type kwargs: dict
        :raises ValueError: If the model type is not supported.
        :raises FileNotFoundError: If the model file is not found.
        """
        self._not_metrics = [
            "real_psf_fluxes",
            "predicted_psf_fluxes",
            "real_gal_fluxes",
            "predicted_gal_fluxes"
        ]
        # Initialize the device
        self._device = get_device()

        # Initialize the model
        if model_type not in AVALAIBLE_MODELS:
            raise ValueError(f"Model type '{model_type}' is not supported.")
        
        self._model = AVALAIBLE_MODELS[model_type](**kwargs)

        try:
            # Load the model
            self._model.load_model(model_filename, data_folder)

            info = "Model Tester initialized.\n"
            info += f"Model type: {model_type}\n"
            info += f"Model filename: {model_filename}\n"
            info += f"Data folder: {data_folder}\n"
            info += f"Device: {self._device}"
            print_box(info)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {e.filename}. Please check the file path.") from e
        
        self._model_filename = model_filename
        self._model_type = model_type
        self._test_loader = test_loader
        self._data_folder = data_folder
        self._transform = transform

    @log_execution("Performing photometry... (DEPRICATED)", "Photometry completed!")
    def do_photometry(
        self,
        max_segmentation_workers: int
    ) -> tuple:
        """
        Perform photometry on the test dataset using the model.

        :param max_segmentation_workers: Maximum number of workers to use for
        segmentation.
        :type max_segmentation_workers: int
        :return: Tuple of `StructuralParamsRegistry` objects containing the
        input images, target images, and cleaned images.
        :rtype: tuple[StructuralParamsRegistry, StructuralParamsRegistry, StructuralParamsRegistry]
        """
        print_box(str(self._test_loader))

        # Create the photometry object
        photometry = Photometry()

        self._model.to(self._device)

        self._model.eval()
        
        input_registry = StructuralParamsRegistry("input_registry")
        target_registry = StructuralParamsRegistry("target_registry")
        cleaned_image_registry = StructuralParamsRegistry("cleaned_image_registry")
        with torch.no_grad():
            for batch in tqdm(self._test_loader, desc="Performing photometry", unit="image"):
                inputs, targets, norm_params = batch
                inputs = inputs.to(self._device)

                cleaned_image = self._model(inputs)

                if self._transform is not None:
                    # Inverse the transform to get the cleaned image
                    cleaned_image = self._transform.inverse(cleaned_image, norm_params)

                # Get properties of the images
                input_registry.extend(
                    photometry.perform_segmentation(
                        inputs, max_workers=max_segmentation_workers, registry_name="input_registry"
                    )
                )
                target_registry.extend(
                    photometry.perform_segmentation(
                        targets, max_workers=max_segmentation_workers, registry_name="target_registry"
                    )
                )
                cleaned_image_registry.extend(
                    photometry.perform_segmentation(
                        cleaned_image, max_workers=max_segmentation_workers, registry_name="cleaned_image_registry"
                    )
                )
                
        return input_registry, target_registry, cleaned_image_registry
    
    @log_execution("Testing model...", "Model testing completed!")
    def test_model(
        self,
        metrics: list[str],
        verbose: bool = True,
        save: bool = True,
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
        # Get the metrics
        metrics = get_metrics(metrics)

        # Test the model
        evaluation_dict = self._test(
            self._test_loader,
            metrics,
        )

        if verbose:
            info = "Results:\n"
            for metric, scores in evaluation_dict.items():
                if metric in self._not_metrics:
                    continue
                scores = np.array(scores)
                # Remove NaN and big values
                scores = scores[scores > -100]
                scores = scores[scores < 100]
                score = scores.mean()
                info += f"{metric}: {score:.3f}\n"

            print_box(info)

        if save:
            save_pkl_file(
                evaluation_dict,
                os.path.join(self._data_folder, f"{self._model_filename}_test.pkl")
            )
            print_box("Test saved successfully!")

        return evaluation_dict
    
    @log_execution("Performing performance analysis...", "Performance analysis completed!")
    def performance_analysis(
        self,
        metrics: list[str],
        save: bool = True,
        verbose: bool = True,
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

        # Get the loader and the dataset
        loader = self._test_loader
        dataset = loader.dataset
        loader_type = type(loader)
        loader_kwargs = loader.get_kwargs()

        # Get a list of all posible AGN fractions from the dataset
        f_agn_dict = dataset.get_all_f_agn()
        info = "Fraction of AGN in the dataset:\n"
        for f_agn, count in f_agn_dict.items():
            info += f"f_agn = 0.{f_agn}: {count} images\n"
        print_box(info)

        analysis = PAdict(self._model_filename)
        for agn_match, count in tqdm(
            f_agn_dict.items(),
            desc="Performing performance analysis",
            unit="f_agn"
            ):
            agn_match_dataset = deepcopy(dataset)

            # Filter the dataset by the current AGN fraction
            agn_match_dataset.filter_by_f_agn(agn_match)
            loader = loader_type(agn_match_dataset, **loader_kwargs)
            
            # Define the key for the analysis
            fraction_key = f"f_agn = 0.{agn_match}"

            print_box(f"Performing analysis for {fraction_key}...")

            # Evaluate the model on the filtered data
            evaluation_dict = self._test(
                loader=loader,
                metrics=metrics,
            )

            # Store the results in the analysis dictionary
            analysis.add(fraction_key, count, evaluation_dict)

        if save:
            save_pkl_file(
                analysis,
                os.path.join(self._data_folder, f"performance_analysis.pkl")
            )
            print_box("Performance analysis saved successfully!")

        if verbose:
            print_box("Performance analysis:")
            print(analysis)
        return analysis
    
    @log_execution("Cleaning images...", "Image cleaning completed!")
    def clean_images(
            self,
            n: int,
            f_agn: list[int]|int|None = None,
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
        # Get the loader and the dataset
        loader_type = type(self._test_loader)
        loader_kwargs = self._test_loader.get_kwargs()

        dataset = deepcopy(self._test_loader.dataset)

        if f_agn is None:
            # Filter the dataset
            dataset.filter_first_n(n=n)
        else:
            dataset.filter_by_f_agn_list(f_agn_list=f_agn, n=n)

        loader = loader_type(dataset, **loader_kwargs)

        self._model.to(self._device)

        self._model.eval()
        
        source_list = []
        target_list = []
        cleaned_image_list = []
        psf_predicted_list = []
        psf_list = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Cleaning images", unit="image"):
                # Prepare the batch
                (
                    inputs,
                    cleaned_image,
                    targets,
                    psf,
                    psf_predicted
                ) = self._prepare_batch(
                    batch=batch
                )

                # Convert tensors to numpy arrays; shape (B, C, H, W)
                source = inputs.cpu().numpy()
                target = targets.cpu().numpy()
                cleaned_image = cleaned_image.cpu().numpy()
                psf_predicted = psf_predicted.cpu().numpy()
                psf = psf.cpu().numpy()

                # Append to lists
                source_list.append(source)
                target_list.append(target)
                cleaned_image_list.append(cleaned_image)
                psf_predicted_list.append(psf_predicted)
                psf_list.append(psf)

        # Convert lists to numpy arrays
        source_arr = np.concatenate(source_list, axis=0)  # (B, C, H, W)
        target_arr = np.concatenate(target_list, axis=0)  # (B, C, H, W)
        cleaned_image_arr = np.concatenate(cleaned_image_list, axis=0)  # (B, C, H, W)
        psf_predicted_arr = np.concatenate(psf_predicted_list, axis=0)  # (B, C, H, W)
        psf_arr = np.concatenate(psf_list, axis=0)  # (B, C, H, W)

        _, _, H, W = source_arr.shape
        FAGN = len(f_agn) if isinstance(f_agn, list) else 1

        new_shape = (n, FAGN, H, W)

        # Remove the channel dimension and replace it with an len(f_agn) dim 
        # (N, C, H, W) -> (n, len(f_agn), H, W)
        source_arr = np.squeeze(source_arr, axis=1).reshape(FAGN, n, H, W).transpose(1, 0, 2, 3)
        target_arr = np.squeeze(target_arr, axis=1).reshape(FAGN, n, H, W).transpose(1, 0, 2, 3)
        cleaned_image_arr = np.squeeze(cleaned_image_arr, axis=1).reshape(FAGN, n, H, W).transpose(1, 0, 2, 3)
        psf_predicted_arr = np.squeeze(psf_predicted_arr, axis=1).reshape(FAGN, n, H, W).transpose(1, 0, 2, 3)
        psf_arr = np.squeeze(psf_arr, axis=1).reshape(FAGN, n, H, W).transpose(1, 0, 2, 3)

        return source_arr, target_arr, cleaned_image_arr, psf_predicted_arr, psf_arr
    
    def _test(
            self,
            loader: _BaseLoader,
            metrics: list[str],
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
        self._model.to(self._device)
        
        evaluation_dict = {}

        evaluation_dict["real_psf_fluxes"] = []
        evaluation_dict["predicted_psf_fluxes"] = []
        evaluation_dict["real_gal_fluxes"] = []
        evaluation_dict["predicted_gal_fluxes"] = []

        self._model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                # Prepare the batch
                (
                    inputs,
                    cleaned_image,
                    targets,
                    psf,
                    psf_predicted
                ) = self._prepare_batch(
                    batch=batch,
                )

                # Calculate the fluxes
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
                evaluation_dict["real_psf_fluxes"].extend(real_psf_flux)
                evaluation_dict["predicted_psf_fluxes"].extend(predicted_psf_flux)
                evaluation_dict["real_gal_fluxes"].extend(real_gal_flux)
                evaluation_dict["predicted_gal_fluxes"].extend(predicted_gal_flux)

                # Calculate the loss using the specified metrics
                for metric in metrics:
                    metric = metric.to(self._device)

                    # Store the loss in the evaluation dictionary
                    if str(metric) not in evaluation_dict:
                        evaluation_dict[str(metric)] = []


                    score = metric(inputs, cleaned_image, targets, psf)
                    score = score.detach().cpu().tolist()
                    if not isinstance(score, list):
                        score = [score]

                    evaluation_dict[str(metric)].extend(score)

        return evaluation_dict

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
    
    def _prepare_batch(self, batch):
        """
        Prepares a batch for evaluation by moving tensors to device, applying transforms,
        running the model, and computing PSF and predicted PSF.

        :param inputs: Input images tensor.
        :type inputs: torch.Tensor
        :param targets: Target images tensor.
        :type targets: torch.Tensor
        :param transform: Optional transform to apply.
        :type transform: _BaseTransform|None
        :return: Tuple (inputs, targets, cleaned_image, psf, psf_predicted)
        """
        inputs, targets, norm_params = batch
        inputs = inputs.to(self._device)
        targets = targets.to(self._device)

        cleaned_image = self._model(inputs)

        if self._transform is not None:
            residual = inputs - cleaned_image
            residual = self._transform.inverse(residual, norm_params)
            inputs = self._transform.inverse(inputs, norm_params)

            cleaned_image = inputs - residual

        psf = inputs - targets

        psf_predicted = inputs - cleaned_image

        return inputs, cleaned_image, targets, psf, psf_predicted
    