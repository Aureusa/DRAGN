import os
from pathlib import Path
from typing import Any, Dict

import torch

from config.test_configs import TestExperimentConfig
from core.registry import TESTING_STRATEGY_REGISTRY
from data.transforms import _BaseTransform
from networks.models import BaseModel
from utils.plotting import Plotter

from .base_strategy import TestingStrategy
from ..metrics import Metric


@TESTING_STRATEGY_REGISTRY.register("clean_images_standard")
class CleanImagesStandardStrategy(TestingStrategy):
    """Standard testing strategy for UNet and similar models"""
    def __init__(self, config: TestExperimentConfig, **kwargs):
        # Create a directory to store the cleaned images
        self.test_dir = os.path.join(config.experiment_dir, "clean_images_standard")
        os.makedirs(self.test_dir, exist_ok=True)

        # Initialize other parameters
        self.num_images = kwargs.get('num_images', 10)
        self.f_agn = kwargs.get('f_agn', None)
        self.desired_pattern = kwargs.get('desired_pattern', None)
        self.model_name = kwargs.get('model_name', 'Model')
        self.image_filename = kwargs.get('image_filename', 'image')
        self.cleaned_already = 0

    def should_stop(self):
        return self.cleaned_already == self.num_images

    def test_step(
            self,
            model: BaseModel,
            transforms: _BaseTransform | None,
            batch: Dict[str, Any],
            metrics: Metric
        ) -> Dict[str, Any]:
        """Execute one testing step"""
        try:
            inputs = batch["input"]
            targets = batch["target"]
            inputs_norm_params = batch["input_norm_params"]
            targets_norm_params = batch["target_norm_params"]
        except KeyError as e:
            raise ValueError(f"Batch missing required key: {e}")

        batch_size = inputs.shape[0]
        cleaned_after_this_step = batch_size + self.cleaned_already

        if cleaned_after_this_step > self.num_images:
            # Take only the remaining images needed to reach num_images
            remaining_needed = self.num_images - self.cleaned_already
            inputs = inputs[:remaining_needed]
            targets = targets[:remaining_needed]
            inputs_norm_params = inputs_norm_params[:remaining_needed]
            targets_norm_params = targets_norm_params[:remaining_needed]
            batch_size = remaining_needed  # Update batch_size for later use

        with torch.no_grad():
            outputs = model(inputs)

        # Apply inverse transform if needed
        if transforms is not None:
            inputs, outputs, targets = self._apply_inverse_transforms(
                inputs, outputs, targets, transforms, inputs_norm_params, targets_norm_params
            )

        self.cleaned_already += batch_size
        return inputs, outputs, targets

    def aggregate_results(self, all_results: list[tuple]) -> Dict[str, Any]:
        """Aggregate results across all test steps"""
        # Unpack all tuples and concatenate along batch dimension
        inputs_list, outputs_list, targets_list = zip(*all_results)
        
        # Aggregate all tensors
        aggregated_inputs = torch.cat(inputs_list, dim=0) # (B, C, H, W)
        aggregated_outputs = torch.cat(outputs_list, dim=0).unsqueeze(0) # (1, B, C, H, W)
        aggregated_targets = torch.cat(targets_list, dim=0) # (B, C, H, W)

        # Convert to np.ndarray
        aggregated_inputs = aggregated_inputs.cpu().numpy()
        aggregated_outputs = aggregated_outputs.cpu().numpy()
        aggregated_targets = aggregated_targets.cpu().numpy()

        return {
            "aggregated_inputs": aggregated_inputs,
            "aggregated_outputs": aggregated_outputs,
            "aggregated_targets": aggregated_targets
        }

    def finalize_test(self, aggregated_results: Dict[str, Any], experiment_dir: str|Path, verbose: bool) -> Dict[str, Any]:
        """Final processing/reporting step"""
        plotter = Plotter(verbose=verbose)
        plotter.plot_grid(
            aggregated_results["aggregated_inputs"],
            aggregated_results["aggregated_targets"],
            aggregated_results["aggregated_outputs"],
            model_names=[self.model_name],
            filename=self.image_filename,
            data_folder=self.test_dir,
            f_agn=self.f_agn,
            desired_pattern=self.desired_pattern,
            save=True
        )