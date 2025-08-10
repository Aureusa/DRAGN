from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Dict, Any

from data.transforms import _BaseTransform
from networks.models import BaseModel
from ..metrics import Metric
from utils import print_box

class TestingStrategy(ABC):
    """Base class for different testing strategies"""
    def __init__(self, *args, **kwargs):
        del kwargs, args
    
    @abstractmethod
    def test_step(
            self,
            model: BaseModel,
            transforms: _BaseTransform | None,
            batch: Dict[str, Any],
            metrics: Metric
        ) -> Dict[str, Any]:
        """Single test step - this is your core method"""

    def aggregate_results(self, all_results: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across all test steps"""
        aggregated = {}
        for result in all_results:
            for key, value in result.items():
                if key not in aggregated:
                    aggregated[key] = []
                aggregated[key].extend(value)
        return aggregated

    def finalize_test(self, aggregated_results: Dict[str, Any], verbose: bool) -> Dict[str, Any]:
        """Final processing/reporting step"""
        if verbose:
            info = "Final results:"
            for key, values in aggregated_results.items():
                info += f"\n  {key}: {np.array(values).mean()}"
            print_box(info)
        return aggregated_results

    def _apply_inverse_transforms(self, inputs, outputs, targets, transforms, input_norm_params, target_norm_params):
        """Helper method to apply inverse transforms"""
        ins, outs, trgs = [], [], []
        
        for i in range(len(outputs)):
            inv_input = transforms.inverse(inputs[i], input_norm_params[i])
            inv_output = transforms.inverse(outputs[i], input_norm_params[i])
            inv_target = transforms.inverse(targets[i], target_norm_params[i])

            ins.append(inv_input)
            outs.append(inv_output)
            trgs.append(inv_target)
        
        return (
            torch.stack(ins, dim=0),
            torch.stack(outs, dim=0), 
            torch.stack(trgs, dim=0)
        )
