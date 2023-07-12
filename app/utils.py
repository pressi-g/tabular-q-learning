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
        # add try catch if the action is not in the dictionary as yet. Happens when the state is actioned for the first time
        try:
            action = np.argmax(Q[currentS_Key])
        except KeyError:
            Q[currentS_Key] = np.zeros(numActions)
            action = np.argmax(Q[currentS_Key])

    return action


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