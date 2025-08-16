from config import MultiModelTestExperimentConfig
from testing.multi_model_tester import MultiModelTester

def main(conf_filepath: str = "/home4/s4683099/DRAGN/example_multi_model_test_config.yaml"):
    # Load the experiment configuration
    config = MultiModelTestExperimentConfig.from_yaml(conf_filepath)

    # Initialize the tester with the loaded configuration
    tester = MultiModelTester(config)

    # Start the testing process
    tester.test()

if __name__ == "__main__":
    main()
