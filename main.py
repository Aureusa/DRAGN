from config.base import create_example_config_yaml
from config.multi_model_test_configs import MultiModelTestExperimentConfig

if __name__ == "__main__":
    # Create an example configuration file
    create_example_config_yaml(MultiModelTestExperimentConfig, "example_multi_model_test_config.yaml")
