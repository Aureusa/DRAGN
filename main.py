from config.base import create_example_config_yaml
from config.multi_model_test_configs import MultiModelTestExperimentConfig
from config.test_configs import TestExperimentConfig
from config.train_configs import ExperimentConfig

if __name__ == "__main__":
    # Create an example configuration file
    create_example_config_yaml(MultiModelTestExperimentConfig, "example_multi_model_test_config.yaml")
    create_example_config_yaml(TestExperimentConfig, "example_test_config.yaml")
    create_example_config_yaml(ExperimentConfig, "example_train_config.yaml")
