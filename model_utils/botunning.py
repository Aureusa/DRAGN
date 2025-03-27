import torch
import monai
from monai.networks.nets import AttentionUnet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd
from monai.data import Dataset, DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch.optim import Adam

from botorch.acquisition import ExpectedImprovement
from botorch.optim.optimize import optimize_acqf


from model.attention_unet import AttentionUNET

def objective_function(bounds):
    """Train Attention U-Net and return validation Dice loss"""
    # Define hyperparameters
    # TODO: Define hyperparameters
    # bounds = torch.tensor([
    #     [0.0001, 0.0, 16],   # Min values
    #     [0.01, 0.5, 128]      # Max values
    # ])
    # dummy1, dummy2, dummy3 = X[:, 0].item(), X[:, 1].item(), X[:, 2].item()

    # Load data
    # TODO: Load data

    # Define model
    # TODO: Define model with hyperparameters
    model = AttentionUNET(...)

    # Define loss and optimizer
    # TODO: Define loss and optimizer

    # Training the model
    # TODO:
    # - Pass the data loaders, learning rate, loss func, and num epochs 
    # - Fix the return value to return the validation metric
    metric_score = model.train_model(...)

    return -metric_score  # Negate Dice score to minimize

def get_next_candidate(bounds, gp_model, Y_init):
    """Optimize the acquisition function to suggest next hyperparameters."""
    ei = ExpectedImprovement(gp_model, best_f=Y_init.max())
    
    new_X, _ = optimize_acqf(
        acq_function=ei,
        bounds=bounds,
        q=1,  # Number of points to sample per iteration
        num_restarts=5,
        raw_samples=20  # Number of random starting points
    )
    return new_X