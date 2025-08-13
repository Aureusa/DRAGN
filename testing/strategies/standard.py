from pathlib import Path
from typing import Any, Dict

import torch

from core.registry import TESTING_STRATEGY_REGISTRY
from data.transforms import _BaseTransform
from networks.models import BaseModel
from utils.persistence import save_pkl_file

from .base_strategy import TestingStrategy
from ..metrics import Metric


@TESTING_STRATEGY_REGISTRY.register("standard")
class StandardTestingStrategy(TestingStrategy):
    """Standard testing strategy for UNet and similar models"""
    def __init__(self, config, **kwargs):
        self.save_results = kwargs.get('save_results', True)
        del config

    def should_stop(self):
        return False

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


        with torch.no_grad():
            outputs = model(inputs)

        # Apply inverse transform if needed
        if transforms is not None:
            inputs, outputs, targets = self._apply_inverse_transforms(
                inputs, outputs, targets, transforms, inputs_norm_params, targets_norm_params
            )

        # Compute the results
        results = {}
        for metric in metrics:
            metric_name = str(metric)  # Use string representation of the metric
            if metric_name not in results:
                results[metric_name] = []
            
            metric_value = metric(inputs, outputs, targets)
            
            # Handle both scalar and multi-element metrics
            if metric_value.numel() == 1:
                # Single element tensor - convert to scalar
                results[metric_name].append(metric_value.item())
            else:
                # Multi-element tensor - extend with all values
                results[metric_name].extend(metric_value.cpu().numpy().tolist())
        return results

    def finalize_test(self, aggregated_results: Dict[str, Any], experiment_dir: str|Path, verbose: bool) -> Dict[str, Any]:
        final_results = super().finalize_test(aggregated_results, experiment_dir, verbose)
        if self.save_results:
            save_pkl_file(final_results, experiment_dir / "standard_test_results.pkl")
        return final_results
    