from abc import abstractmethod
from typing import Any, Dict

import numpy as np
import torch

from config.test_configs import TestExperimentConfig
from core.component import Component
from data.transforms import _BaseTransform
from networks.models import BaseModel
from utils import print_box
from ..metrics import Metric


class TestingStrategy(Component):
    """
    Base class for different testing strategies.
    Must implement test_step and should_stop methods.
    """
    @abstractmethod
    def __init__(self, configs: TestExperimentConfig, **kwargs) -> None:
        """
        Initialize the testing strategy with configuration and other parameters.

        :param configs: The configuration object for the testing strategy.
        :type configs: TestExperimentConfig
        :param kwargs: Additional keyword arguments.
        :type kwargs: Any
        """
        self.configs = configs
        self.experiment_dir = configs.experiment_dir
        pass
    
    @abstractmethod
    def test_step(
            self,
            model: BaseModel,
            transforms: _BaseTransform | None,
            batch: Dict[str, Any],
            metrics: Metric
        ) -> Dict[str, Any]:
        """Single test step - this is your core method"""
        pass

    @abstractmethod
    def should_stop(self) -> bool:
        """Check if the testing strategy has completed its work"""
        pass

    @classmethod
    def from_config(cls, config: TestExperimentConfig, **kwargs) -> "TestingStrategy":
        """
        Factory method to create a TestingStrategy from a config object

        :param config: The configuration object for the testing strategy.
        :type config: TestExperimentConfig
        :param kwargs: Additional keyword arguments.
        :type kwargs: Any
        """
        return cls(config, **kwargs)

    def aggregate_results(self, all_results: list[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results across all test steps

        :param all_results: A list of dictionaries containing results from each test step.
        :type all_results: list[Dict[str, Any]]
        :return: A dictionary containing the aggregated results.
        :rtype: Dict[str, Any]
        """
        aggregated = {}
        for result in all_results:
            for key, value in result.items():
                if key not in aggregated:
                    aggregated[key] = []
                aggregated[key].extend(value)
        return aggregated

    def finalize_test(self, aggregated_results: Dict[str, Any], verbose: bool) -> Dict[str, Any]:
        """
        Final processing/reporting step

        :param aggregated_results: The aggregated results from all test steps.
        :type aggregated_results: Dict[str, Any]
        :param verbose: Whether to print detailed results.
        :type verbose: bool
        :return: The finalized results after processing.
        :rtype: Dict[str, Any]
        """
        if verbose:
            info = "Final results:"
            for key, values in aggregated_results.items():
                info += f"\n  {key}: {np.array(values).mean()}"
            print_box(info)
        return aggregated_results

    def _apply_inverse_transforms(
            self,
            inputs: torch.Tensor,
            outputs: torch.Tensor,
            targets: torch.Tensor,
            transforms: _BaseTransform,
            input_norm_params: list[Dict[str, Any]],
            target_norm_params: list[Dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper method to apply inverse transforms

        :param inputs: The input tensors.
        :type inputs: torch.Tensor
        :param outputs: The output tensors.
        :type outputs: torch.Tensor
        :param targets: The target tensors.
        :type targets: torch.Tensor
        :param transforms: The transform object to apply inverse transforms.
        :type transforms: _BaseTransform
        :param input_norm_params: Normalization parameters for inputs.
        :type input_norm_params: list[Dict[str, Any]]
        :param target_norm_params: Normalization parameters for targets.
        :type target_norm_params: list[Dict[str, Any]]
        :return: The inverse transformed inputs, outputs, and targets.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
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
