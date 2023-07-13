# Author: Preston Govender

import numpy as np

# grid search imports
from itertools import product
import csv

# from utils import save_q_values, load_q_values
from q_learning import q_learning
from sarsa import sarsa


def grid_search(env, hyperparams, episodes, algorithm):
    """
    Perform a grid search to find the optimal hyperparameters.

    Parameters:
        env (gym.Env): The Gym environment.
        hyperparams (dict): Dictionary of hyperparameter ranges to search.
        episodes (int): The number of episodes to train the agent for each hyperparameter combination.
        algorithm (str): The algorithm to use for the grid search. Either "q_learning" or "sarsa".

    Returns:
        dict: Dictionary containing the optimal hyperparameters and the corresponding Q-values.
        list: List of average rewards per episode for the best hyperparameter combination.
        list: List of average steps per episode for the best hyperparameter combination.
    """
    best_hyperparams = {}
    best_average_rewards = []
    best_average_steps = []
    best_reward = -np.inf
    best_steps = -np.inf

    param_combinations = list(product(*hyperparams.values()))

    # create a csv file to store the results
    with open(f"{algorithm}_hyperparameter_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["alpha", "gamma", "epsilon", "average_steps", "average_rewards"]
        )

    for params in param_combinations:
        alpha, gamma, epsilon = params

        print(
            f"Testing {algorithm} hyperparameters: alpha={alpha}, gamma={gamma}, epsilon={epsilon}"
        )

        if algorithm == "q_learning":
            Q, average_rewards, average_steps, final_steps, final_reward = q_learning(
            env, episodes, alpha, gamma, epsilon)
        elif algorithm == "sarsa":
            Q, average_rewards, average_steps, final_steps, final_reward = sarsa(
            env, episodes, alpha, gamma, epsilon)

        print(f"Average reward: {average_rewards}")
        print(f"Average steps: {average_steps}")

        # store the results of the best hyperparameter combination
        with open(f"{algorithm}_hyperparameter_results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([alpha, gamma, epsilon, average_steps, average_rewards])

        if average_rewards > best_reward:
            best_reward = average_rewards
            best_steps = average_steps
            best_hyperparams = {
                "alpha": alpha,
                "gamma": gamma,
                "epsilon": epsilon,
            }
            best_average_rewards = best_reward
            best_average_steps = best_steps

    # save best hyperparameter combination result to a csv file
    with open(f"{algorithm}_hyperparameter_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "best_alpha",
                "best_gamma",
                "best_epsilon",
                "best_average_steps",
                "best_average_rewards",
            ]
        )
        writer.writerow(
            [
                best_hyperparams["alpha"],
                best_hyperparams["gamma"],
                best_hyperparams["epsilon"],
                best_average_steps,
                best_average_rewards,
            ]
        )

    return best_hyperparams, best_average_rewards, best_average_steps
