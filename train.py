from configs import ExperimentConfig
from training.trainer import UniversalTrainer

def main(conf_filepath: str = "experiments/unet_experiment1/unet_experiment1.yaml"):
    # Load the experiment configuration
    config = ExperimentConfig.from_yaml(conf_filepath)

    # Initialize the trainer with the loaded configuration
    trainer = UniversalTrainer(config)
    
    # Start the training process
    trainer.train()

if __name__ == "__main__":
    main()
