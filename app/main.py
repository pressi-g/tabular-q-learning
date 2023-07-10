import numpy as np
import gymnasium as gym
import minigrid
import random
from minigrid.wrappers import *
import time
import pickle
from os.path import exists
import metrohash
from torch.utils.tensorboard import SummaryWriter
# from environment import create_minigrid_environment
# from utils import extract_object_information


def create_minigrid_environment(grid_type='MiniGrid-Empty-8x8-v0'):
    """
    Create and return a MiniGrid environment.

    Parameters:
        grid_type (str): The type of grid environment to create. Defaults to 'MiniGrid-Empty-8x8-v0'.

    Returns:
        gym.Env: A Gym environment object representing the MiniGrid environment.
    """
    env = gym.make(grid_type)
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
    # reset the environment and get the initial observation
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
    env.reset()
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
    tmp = np.reshape(obs, [rows * cols * x, 1], 'F')[0 : rows * cols]
    return np.reshape(tmp, [rows, cols], 'C')


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
        action = random.randint(0, numActions - 1)
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
    """

    # Tensorboard writer
    writer = SummaryWriter()

    Q = {}  # declare the variable to store the tabular value-function

    max_steps = env.max_steps

    # Use a wrapper so the observation only contains the grid information
    env = ImgObsWrapper(env)

    print('Start training...')
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

            action = epsilon_greedy_action(Q, currentS_Hash, env.action_space.n, epsilon)

            # Increment the 'steps_done' counter
            steps_done += 1


            # take the action in the environment
            next_obs, reward, done, truncated, info = env.step(action)

            # extract the next state from the observation
            nextS = extract_object_information(next_obs)
            nextS_Hash = metrohash.hash64_int(nextS)
            # nextS_Hash = hash(tuple(nextS.flatten()))
            

            # Update the Q-value for the current state-action pair
            if currentS_Hash not in Q:
                Q[currentS_Hash] = np.zeros(env.action_space.n)
            if nextS_Hash not in Q:
                Q[nextS_Hash] = np.zeros(env.action_space.n)
            
            Q[currentS_Hash][action] += alpha * (reward + gamma * np.max(Q[nextS_Hash]) - Q[currentS_Hash][action])

            if done:
                # if agent reached its goal successfully
                print('Finished episode successfully taking %d steps and receiving reward %f' % (i, reward))
                break

            if truncated:
                # agent failed to reach its goal successfully
                print('Truncated episode taking %d steps and receiving reward %f' % (i, reward))
                break

            # since the episode is not done, update the current state
            currentS = nextS

        # After each episode, log the reward to TensorBoard
        writer.add_scalar("Reward/train", reward, e)
        writer.flush()

    print('Done training...')
    writer.close()

    return Q


def save_q_values(Q, filename):
    """
    Save the Q-values to a file.

    Parameters:
        Q (dict): The learned Q-values.
        filename (str): The filename to save the Q-values.

    Returns:
        None
    """
    with open(filename, 'wb') as handle:
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
        print('Filename %s does not exist, could not load data' % filename)
        return {}

    print('Loading existing Q values')
    with open(filename, 'rb') as handle:
        Q = pickle.load(handle)
    

    return Q


def main():
    # Create the MiniGrid environment
    env = create_minigrid_environment()


    episodes = 2
    alpha = 0.1  # learning rate
    gamma = 0.99  # discount factor
    epsilon = 0.9  # exploration rate

    Q = q_learning(env, episodes, alpha, gamma, epsilon)
    # SARSA = sarsa(env, episodes, alpha, gamma, epsilon)


if __name__ == "__main__":
    main()

