from config import TestExperimentConfig
from testing.tester import UniversalTester

def main(conf_filepath: str = "/home4/s4683099/DRAGN/example_test_config.yaml"):
    # Load the experiment configuration
    config = TestExperimentConfig.from_yaml(conf_filepath)

    # Initialize the tester with the loaded configuration
    tester = UniversalTester(config)

    # Start the testing process
    tester.test()

if __name__ == "__main__":
    main()
