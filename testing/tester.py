from typing import List

from tqdm import tqdm

from config.test_configs import TestExperimentConfig
from core.registry import (
    TESTING_STRATEGY_REGISTRY,
    METRICS_REGISTRY,
)
from .strategies.base_strategy import TestingStrategy
from .metrics import Metric
from utils.device import get_device, move_batch_to_device
from utils.building import build_model, build_loaders, build_transform
from utils import print_box


class UniversalTester:
    """Universal tester that uses configuration-driven testing strategies"""
    
    def __init__(self, config: TestExperimentConfig):
        """
        Initialize the UniversalTester
        
        :param config: Configuration object for testing
        :type config: TestExperimentConfig
        """
        self.config = config
        self.experiment_dir = self.config.experiment_dir
        self.testing_strategy_type = self.config.testing_config.testing_strategy
        self._device = get_device()
        
        # Build components from config
        self.model = build_model(config.model_settings_config).to(self._device)
        self.transform = build_transform(config.transform_config) if config.transform_config else None
        self.train_loader = build_loaders(config.test_data_config, self.transform)
        self.metrics = self._build_metrics(config.metrics_config)
        self.testing_strategy: TestingStrategy = TESTING_STRATEGY_REGISTRY.build(
            self.testing_strategy_type,
            config.testing_config.testing_strategy_params
        )

        # Load model
        self.model.load_checkpoint(config.testing_config.model_filename, self.experiment_dir)

        info = "Universal Tester initialized.\n"
        info += f"Model: {config.model_settings_config.architecture}\n"
        info += f"Model File: {config.testing_config.model_filename}\n"
        info += f"Dataset: {config.test_data_config.dataset_type}\n"
        info += f"Transforms: {config.transform_config.transforms if config.transform_config else 'None'}\n"
        info += f"Metrics: {', '.join(config.metrics_config.metrics)}\n"
        print_box(info)

    def _build_metrics(self, config) -> List[Metric]:
        """Build metrics"""
        metrics = []
        for metric_name in config.metrics:
            metric_type = METRICS_REGISTRY.get(
                metric_name,
            )
            metrics.append(metric_type(**config.params.get(metric_name, {})))
        return metrics
    
    def test(self) -> dict:
        """
        Test the model using the configured strategy
        
        :return: Dictionary of evaluation metrics
        :rtype: dict
        """
        self.model.eval()  # Set model to evaluation mode

        results = []
        for batch in tqdm(self.train_loader):
            batch = move_batch_to_device(batch, self._device)
            res = self.testing_strategy.test_step(
                model=self.model,
                transforms=self.transform,
                batch=batch,
                metrics=self.metrics
            )
            results.append(res)
        aggregated_results = self.testing_strategy.aggregate_results(results)
        final_results = self.testing_strategy.finalize_test(
            aggregated_results,
            experiment_dir=self.experiment_dir,
            verbose=self.config.testing_config.verbose
        )
        return final_results
