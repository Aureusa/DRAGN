from typing import Dict, Any, List
import torch
import os
import numpy as np

from config.multi_model_test_configs import MultiModelTestExperimentConfig
from core.registry import TESTING_STRATEGY_REGISTRY
from utils.plotting import Plotter

from .clean_images import CleanImagesStandardStrategy


@TESTING_STRATEGY_REGISTRY.register("clean_images_multi_model")
class CleanImagesMultiModelStrategy(CleanImagesStandardStrategy):
    """
    Testing strategy for passing images through multiple model and saving the cleaned outputs.
    It creates grid plot containing the input image (first column), target image if avaliable
    (second column) and the different models outputs (subsequent columns).
    """
    def __init__(self, config: MultiModelTestExperimentConfig, **kwargs):
        """
        Initialize the testing strategy with configuration and other parameters.

        :param configs: The configuration object for the testing strategy.
        :type configs: TestExperimentConfig
        :param kwargs: Additional keyword arguments. Modular parameters are:
            num_images - default 10
            f_agn - None
            desired_pattern - None
            model_name - 'Model'
            images_filename - 'image'
        :type kwargs: Any
        """
        super().__init__(config, **kwargs)
        self.test_dir = os.path.join(config.experiment_dir, "clean_images_multi_model")

    def should_stop(self):
        """
        Stops when the desired number of images have been cleaned for the current model
        and resets it for the next model.
        """
        if self.cleaned_already == self.num_images:
            self.cleaned_already = 0  # Reset for next round
            return True
        return False

    def aggregate_results(self, all_results: Dict[str, List[tuple]]) -> Dict[str, Any]:
        """
        Aggregate results across all test steps. Concatenates all the tensors together
        to get a unified representation. The output tensor is unsqueezed along the batch dimension
        so that it accounts for the different models.

        :param all_results: A dictionary containing the results from all models
        :type all_results: Dict[str, List[tuple]]
        :return: A dictionary containing the aggregated results
        :rtype: Dict[str, Any]
        """
        aggregated_results = {}

        outputs = []
        for model_name, results in all_results.items():
            # Unpack all tuples and concatenate along batch dimension
            inputs_list, outputs_list, targets_list = zip(*results)

            # Add the inputs and targets to the results
            if "aggregated_inputs" not in aggregated_results:
                aggregated_results["aggregated_inputs"] = torch.cat(inputs_list, dim=0).cpu().numpy() # (B, C, H, W)
            if "aggregated_targets" not in aggregated_results:
                aggregated_results["aggregated_targets"] = torch.cat(targets_list, dim=0).cpu().numpy() # (B, C, H, W)
            if "models" not in aggregated_results:
                aggregated_results["models"] = []

            # Add the name to the results
            aggregated_results["models"].append(model_name)

            # Aggregate the outputs
            outs = torch.cat(outputs_list, dim=0).unsqueeze(0).cpu().numpy() # (1, B, C, H, W)
            outputs.append(outs)

        aggregated_outputs = np.concatenate(outputs, axis=0) # (M, B, C, H, W); where M = number of models
        aggregated_results["aggregated_outputs"] = aggregated_outputs
        return aggregated_results
    