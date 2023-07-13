# Author: Preston Govender

from utils import save_q_values, load_q_values, create_minigrid_environment
from grid_search import grid_search
from q_learning import q_learning
from sarsa import sarsa
import numpy as np


def main():
    # Create the MiniGrid environment
    env = create_minigrid_environment()

    # Define the hyperparameter ranges to search
    hyperparams = {
        "alpha": np.linspace(0.1, 0.9, 9),
        "gamma": [0.9],
        "epsilon": np.linspace(0.1, 0.9, 9),
    }

    episodes = (
        2000  # Number of episodes to evaluate for each hyperparameter combination
    )

    best_hyperparams, best_average_rewards, best_average_steps = grid_search(
        env, hyperparams, episodes, algorithm="sarsa"
    )
    print("Best hyperparameters:", best_hyperparams)
    print("Best average rewards:", best_average_rewards)
    print("Best average steps:", best_average_steps)


if __name__ == "__main__":
    main()
