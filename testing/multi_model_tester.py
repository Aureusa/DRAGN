from typing import Dict, Any

from tqdm import tqdm

from data.loaders import _BaseLoader
from config.multi_model_test_configs import MultiModelTestExperimentConfig, SingleModelConfig
from config.common import DataConfig
from core.registry import TESTING_STRATEGY_REGISTRY
from utils import print_box
from utils.building import build_loaders, build_model, build_transform, build_metrics
from utils.device import get_device, move_batch_to_device

from .strategies.base_strategy import TestingStrategy


class MultiModelTester:
    """Multi-model tester that uses configuration-driven testing strategies"""
    def __init__(self, config: MultiModelTestExperimentConfig) -> None:
        """
        Initialize the MultiModelTester
        
        :param config: Configuration object for testing setup
        :type config: MultiModelTestExperimentConfig
        """
        self.config = config

        # Get the testing configs
        self.experiment_dir = self.config.experiment_dir
        self.testing_strategy_type = self.config.multi_model_testing_config.testing_strategy
        self.testing_strategy_params = self.config.multi_model_testing_config.testing_strategy_params
        self.verbose = self.config.multi_model_testing_config.verbose

        # Get the device
        self._device = get_device()

        # Build testing strategy
        self.testing_strategy: TestingStrategy = TESTING_STRATEGY_REGISTRY.build(
            self.testing_strategy_type,
            config,
            **self.testing_strategy_params
        )
        
        # Get the names of the models
        self.model_names = list(config.models_settings_config.keys())

        # Build data loaders
        self.test_loaders, self.transforms = self._build_data_loaders_and_transforms(
            config.models_settings_config, config.data_config
        )

        # Build metrics
        self.metrics = build_metrics(config.metrics_config) if config.metrics_config else []

        if self.verbose:
            # Info
            info = "Universal Tester initialized.\n"
            info += f"Models: {', '.join(self.model_names)}\n"
            info += f"Testing Strategy: {self.testing_strategy_type}\n"
            info += f"Experiment Folder: {self.experiment_dir}\n"
            info += f"Testing Data Folder: {self.config.data_config.data_path}\n"
            if self.metrics:
                info += f"Metrics: {', '.join([str(metric) for metric in self.metrics])}\n"
            print_box(info)

    def _build_data_loaders_and_transforms(
            self,
            models_settings_config: Dict[str, SingleModelConfig],
            data_config: DataConfig
        ) -> Dict[str, _BaseLoader]:
        """
        Build data loaders and transforms for each model

        :param model_settings_config: Configuration for each model.
        Should be a dict with the name of the models as keys and 
        their configurations as values
        :type model_settings_config: Dict[str, SingleModelConfig]
        :param data_config: Configuration for the data
        :type data_config: DataConfig
        :return: A dictionary of data loaders for each model
        :rtype: Dict[str, _BaseLoader]
        """
        data_loaders = {}
        transforms = {}
        for model_name, model_config in models_settings_config.items():
            transform = build_transform(model_config.transform_config) if model_config.transform_config else None
            data_loader = build_loaders(data_config, transform)
            data_loaders[model_name] = data_loader
            transforms[model_name] = transform
        return data_loaders, transforms

    def test(self) -> Any:
        """
        Test each model using the testing strategy
        
        :return: The final results
        :rtype: Any
        """
        result = {}
        for model_name in self.model_names:
            data_loader = self.test_loaders[model_name]
            transform = self.transforms[model_name]
            model_config = self.config.models_settings_config[model_name]
            model = build_model(model_config)
            model.load_checkpoint(**model_config.get_loading_args())
            model.eval()
            for batch in tqdm(data_loader):
                batch = move_batch_to_device(batch, self._device)
                res = self.testing_strategy.test_step(
                    model=model,
                    transforms=transform,
                    batch=batch,
                    metrics=self.metrics
                )
                if model_name not in result:
                    result[model_name] = []
                result[model_name].append(res)

                if self.testing_strategy.should_stop():
                    break
        aggregated_results = self.testing_strategy.aggregate_results(result)
        final_results = self.testing_strategy.finalize_test(
            aggregated_results,
            verbose=self.verbose
        )
        return final_results
        