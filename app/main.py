# Author: Preston Govender

# RL imports
import numpy as np
import gymnasium as gym
import minigrid
import random
from minigrid.wrappers import *
import time

# exporting imports
import pickle
from os.path import exists

# hashing imports
import metrohash

# tensorboard imports
from torch.utils.tensorboard import SummaryWriter

# grid search imports
from itertools import product
import csv


# from environment import create_minigrid_environment
# from utils import extract_object_information


def create_minigrid_environment(grid_type="MiniGrid-Empty-8x8-v0", render_mode=None):
    """
    Create and return a MiniGrid environment.

    Parameters:
        grid_type (str): The type of grid environment to create. Defaults to 'MiniGrid-Empty-8x8-v0'.
        render_mode (str): The rendering mode. Defaults to 'none'. Set to 'human' to render the environment.

    Returns:
        gym.Env: A Gym environment object representing the MiniGrid environment.
    """
    env = gym.make(grid_type, render_mode)
    return env


def random_agent(env):
    """
    Random agent that generates random actions at each step.

    Parameters:
        env (gym.Env): The Gym environment.

    Returns:
        Tuple: A tuple containing the action, reward, done status, and info.
    """
    done = False
    # # reset the environment and get the initial observation
    obs, info = env.reset()
    print(obs)
    print(info)

    while not done:
        action = random.randint(0, env.action_space.n - 1)
        obs, reward, done, truncated, info = env.step(action)
    # print("Action a: %d was generated" % action)

    return obs, reward, done, truncated, info, action


def get_partial_observation(env):
    """
    Get the partial observation from the environment.

    Parameters:
        env (gym.Env): The Gym environment.

    Returns:
        gym.spaces.Dict: A dictionary containing the partial observation.
    """
    # Get the partial observation from the environment
    partial_obs = env.ImgObsWrapper(env)
    # reset the environment
    # env.reset()
    return partial_obs


def extract_object_information(obs):
    """
    Extracts the object index information from the image observation.

    The 'image' observation contains information about each tile around the agent.
    Each tile is encoded as a 3-dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE),
    where OBJECT_TO_IDX and COLOR_TO_IDX mapping can be found in 'minigrid/minigrid.py',
    and the STATE can be as follows:
        - door STATE -> 0: open, 1: closed, 2: locked

    Parameters:
        obs (numpy.ndarray): The image observation from the environment.

    Returns:
        numpy.ndarray: A 2D array containing the object index information extracted from the observation.
    """
    (rows, cols, x) = obs.shape
    tmp = np.reshape(obs, [rows * cols * x, 1], "F")[0 : rows * cols]
    return np.reshape(tmp, [rows, cols], "C")


# def main():
#     # Create the MiniGrid environment
#     env = create_minigrid_environment()

#     # Run the random agent
#     obs, reward, done, truncated, info, action = random_agent(env)

#     # Print the final action, reward, done, info, and observation
#     print("Final action: %d" % action)
#     print("Final reward: %d" % reward)
#     print("Final done: %s" % done)
#     print("Final info: %s" % info)
#     print("Final observation: %s" % obs)

# if __name__ == "__main__":
#     main()


def epsilon_greedy_action(Q, currentS_Key, numActions, epsilon):
    """
    Perform an epsilon-greedy action selection.

    Parameters:
        Q (dict): The value-function dictionary.
        currentS_Key (int): The hash key representing the current state in the value-function dictionary.
        numActions (int): The number of possible actions in the environment.
        epsilon (float): The exploration rate, indicating the probability of exploration (random action).

    Returns:
        int: The selected action.
    """
    if random.random() < epsilon:
        # Explore the environment by selecting a random action
        action = random.randint(0, numActions-1)
    else:
        # Exploit the environment by selecting an action that maximizes the value function at the current state
        action = np.argmax(Q[currentS_Key])

    return action


def q_learning(env, episodes, alpha, gamma, epsilon):
    """
    Train the agent using Q-Learning.

    Parameters:
        env (gym.Env): The Gym environment.
        episodes (int): The number of episodes to train the agent.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.

    Returns:
        dict: The learned Q-values.
        float: The average reward per episode.
        float: The average number of steps per episode.
    """

    # Tensorboard writer
    writer = SummaryWriter()

    Q = {}  # declare the variable to store the tabular value-function

    max_steps = env.max_steps
    numActions = 3  # env.action_space.n

    # total_reward = 0
    # total_steps = 0

    # Use a wrapper so the observation only contains the grid information
    env = ImgObsWrapper(env)

    print("Start training...")
    steps_done = 0  # Counter for total number of training steps taken

    for e in range(episodes):
        # reset the environment
        obs, _ = env.reset()


        # extract the current state from the observation
        currentS = extract_object_information(obs)

        for i in range(max_steps):
            # Choose an action using epsilon-greedy action selection
            # currentS_Hash = hash(tuple(currentS.flatten()))
            currentS_Hash = metrohash.hash64_int(currentS)

            action = epsilon_greedy_action(
                Q, currentS_Hash, numActions, epsilon
            )

            # Increment the 'steps_done' counter
            steps_done += 1

            # take the action in the environment
            next_obs, reward, done, truncated, info = env.step(action)

            # extract the next state from the observation
            nextS = extract_object_information(next_obs)
            nextS_Hash = metrohash.hash64_int(nextS)

            # Update the Q-value for the current state-action pair
            if currentS_Hash not in Q:
                Q[currentS_Hash] = np.zeros(numActions)
            if nextS_Hash not in Q:
                Q[nextS_Hash] = np.zeros(numActions)

            Q[currentS_Hash][action] += alpha * (
                reward + gamma * np.max(Q[nextS_Hash]) - Q[currentS_Hash][action]
            )

            if done:
                # if agent reached its goal successfully
                print(
                    "Finished episode successfully taking %d steps and receiving reward %f"
                    % (i, reward)
                )
                break

            if truncated:
                # agent failed to reach its goal successfully
                print(
                    "Truncated episode taking %d steps and receiving reward %f"
                    % (i, reward)
                )
                break

            # since the episode is not done, update the current state
            currentS = nextS
        # # Increment the 'steps_done' counter
        # total_steps += 1
        # total_reward += reward

        # After each episode, log the reward to TensorBoard
        writer.add_scalar("Reward/train", reward, e)
        writer.flush()
    
    # average_reward = total_reward / episodes
    # average_steps = total_steps / episodes

    print("Done training...")
    writer.close()

    return Q#, average_reward, average_steps


def save_q_values(Q, filename):
    """
    Save the Q-values to a file.

    Parameters:
        Q (dict): The learned Q-values.
        filename (str): The filename to save the Q-values.

    Returns:
        None
    """
    with open(filename, "wb") as handle:
        pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_q_values(filename):
    """
    Load the Q-values from a file.

    Parameters:
        filename (str): The filename to load the Q-values from.

    Returns:
        dict: The loaded Q-values.
    """
    if not exists(filename):
        print("Filename %s does not exist, could not load data" % filename)
        return {}

    print("Loading existing Q values")
    with open(filename, "rb") as handle:
        Q = pickle.load(handle)

    return Q


def main():
    # Create the MiniGrid environment
    env = create_minigrid_environment()


    episodes = 3000  # number of episodes to train the agent
    alpha = 0.2  # learning rate
    gamma = 0.9  # discount factor
    epsilon = 0.8  # exploration rate

    Q = q_learning(env, episodes, alpha, gamma, epsilon)
    # SARSA = sarsa(env, episodes, alpha, gamma, epsilon)

    # Save the Q-values
    save_q_values(Q, 'q_values.pkl')

    # # Load the Q-values
    # Q = load_q_values('q_values.pkl')
    # print(Q)

    # print("Average reward per episode: %f" % average_reward)
    # print("Average steps per episode: %f" % average_steps)


if __name__ == "__main__":
    main()



# def grid_search(env, hyperparams, episodes=1000):
#     """
#     Perform a grid search to find the optimal hyperparameters.

#     Parameters:
#         env (gym.Env): The Gym environment.
#         hyperparams (dict): Dictionary of hyperparameter ranges to search.
#         n (int): The number of episodes to evaluate the agent for each hyperparameter combination.

#     Returns:
#         dict: Dictionary containing the optimal hyperparameters and the corresponding Q-values.
#     """
#     best_hyperparams = {}
#     best_reward = float("-inf")
#     best_steps = float("inf")

#     param_combinations = list(product(*hyperparams.values()))

#     for params in param_combinations:
#         alpha, gamma, epsilon = params

#         print(
#             f"Testing hyperparameters: alpha={alpha}, gamma={gamma}, epsilon={epsilon}"
#         )

#         Q = q_learning(
#             env, episodes=episodes, alpha=alpha, gamma=gamma, epsilon=epsilon
#         )

#         # Evaluate the agent's performance with the current hyperparameters
#         total_reward = 0
#         total_steps = 0
#         for _ in range(episodes):
#             obs, _ = env.reset()
#             currentS = extract_object_information(obs)
#             done = False
#             steps = 0
#             while not done:
#                 currentS_Hash = metrohash.hash64_int(currentS)
#                 action = epsilon_greedy_action(
#                     Q, currentS_Hash, env.action_space.n, epsilon
#                 )
#                 obs, reward, done, _, _ = env.step(action)
#                 nextS = extract_object_information(obs)
#                 currentS = nextS
#                 total_reward += reward
#                 steps += 1
#             total_steps += steps

#         average_reward = total_reward / episodes
#         average_steps = total_steps / episodes

#         print(f"Average reward: {average_reward}")
#         print(f"Average steps: {average_steps}")

#         if average_reward > best_reward:
#             best_reward = average_reward
#             best_hyperparams = {
#                 "alpha": alpha,
#                 "gamma": gamma,
#                 "epsilon": epsilon,
#             }
#             best_steps = average_steps

#     # save each hyperparameter combination results to a csv file
#     with open("hyperparameter_results.csv", "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(
#             ["alpha", "gamma", "epsilon", "average_reward", "average_steps"]
#         )
#         writer.writerow(
#             [
#                 best_hyperparams["alpha"],
#                 best_hyperparams["gamma"],
#                 best_hyperparams["epsilon"],
#                 best_reward,
#                 best_steps,
#             ]
#         )

#     return best_hyperparams, best_reward, best_steps


# def main():
#     # Create the MiniGrid environment
#     env = create_minigrid_environment()

#     # Define the hyperparameter ranges to search
#     hyperparams = {
#         "alpha": np.linspace(0.1, 0.9, 5),
#         "gamma": np.linspace(0.1, 0.9, 5),
#         "epsilon": np.linspace(0.1, 0.9, 5),
#     }

#     episodes = 10  # Number of episodes to evaluate for each hyperparameter combination

#     best_hyperparams, best_reward, best_steps = grid_search(env, hyperparams, episodes)
#     print("Best hyperparameters:", best_hyperparams)
#     print("Best reward:", best_reward)
#     print("Best steps:", best_steps)


# if __name__ == "__main__":
#     main()
