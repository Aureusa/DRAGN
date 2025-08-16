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
   
    def __init__(self, config: MultiModelTestExperimentConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.test_dir = os.path.join(config.experiment_dir, "clean_images_multi_model")

    def should_stop(self):
        if self.cleaned_already == self.num_images:
            self.cleaned_already = 0  # Reset for next round
            return True
        return False

    def aggregate_results(self, all_results: Dict[str, List[tuple]]) -> Dict[str, Any]:
        """Aggregate results across all test steps"""
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
    