import torch
from botorch.utils.sampling import draw_sobol_samples
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from model_utils.botunning import objective_function, get_next_candidate


# TODO: 
# 1. Define the hyperparameter space
# 2. Define the objective function
def hyperparameter_optimization(bounds: torch, num_iterations: int = 10):
    # Draw 5 random samples from the parameter space
    X_init = draw_sobol_samples(bounds=bounds, n=5, q=1).squeeze(1)

    # Evaluate the objective function on the sampled parameters
    Y_init = torch.tensor([objective_function(x.unsqueeze(0)) for x in X_init]).unsqueeze(-1)

    # Train a Gaussian Proccess model on the sampled data
    gp_model = SingleTaskGP(X_init, Y_init)

    for iteration in range(num_iterations):  # Run 10 optimization iterations
        # Define the marginal log likelihood
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

        # Optimize GP Model
        fit_gpytorch_mll(mll)

        # Get the next candidate hyperparameters
        new_X = get_next_candidate(bounds, gp_model, Y_init)

        # Evaluate the objective function on the sampled parameters
        new_Y = torch.tensor([objective_function(new_X)]).unsqueeze(-1)

        # Update dataset
        X_init = torch.cat([X_init, new_X])
        Y_init = torch.cat([Y_init, new_Y])

        # Update GP model
        gp_model.set_train_data(X_init, Y_init, strict=False)

        print(f"Iteration {iteration+1}: Best Dice Loss = {-Y_init.max().item()}")

    # Best hyperparameters
    best_params = X_init[Y_init.argmax()]
    return best_params

