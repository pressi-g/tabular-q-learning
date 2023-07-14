# Author: Preston Govender

# RL imports
import numpy as np
import gymnasium as gym
import minigrid
import random
from minigrid.wrappers import *
import time
import metrohash

from utils import create_minigrid_environment, load_q_values, extract_object_information


def evaluation(algorithm):
    """
    Evaluates the performance of the agent using the Q-values
    :param algorithm: the algorithm to evaluate
    :return: the average reward, steps and completion rate

    """
    # Set the random seed
    random.seed(5)

    # Initialise the environment
    env = create_minigrid_environment(
        grid_type="MiniGrid-Empty-8x8-v0", render_mode=None
    )

    # load the q-table
    Q = load_q_values(f"q_values_{algorithm}.pkl")

    env = ImgObsWrapper(env)

    # initialise the variables
    total_reward = 0
    total_steps = 0
    total_completion = 0

    # run the evaluation
    for i in range(1000):
        # reset the environment and get the initial observation
        obs, info = env.reset()

        # epsilon-greedy policy
        done = False
        truncated = False
        while not done:
            # Extract the current state
            currentS = extract_object_information(obs)

            # Get the epsilon-greedy action based on Q-values
            currentS_Hash = metrohash.hash64_int(currentS)
            if random.random() < 0.01:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[currentS_Hash])

            # Perform the action
            obs, reward, done, truncated, _ = env.step(action)

            # Update the total reward and steps
            total_reward += reward
            total_steps += 1

        # Update the total completion
        if done:
            total_completion += 1

    # calculate the average reward, steps and completion rate
    avg_reward = total_reward / 1000
    avg_steps = total_steps / 1000
    avg_completion = total_completion / 1000 * 100

    # print the results
    print("Average reward: %f" % avg_reward)
    print("Average steps: %f" % avg_steps)
    print("Average completion rate: %f" % avg_completion)

    # return avg_reward, avg_steps, avg_completion


if __name__ == "__main__":
    print("Evaluating Q-learning")
    evaluation("q_learning")
    print("\n")
    print("Evaluating SARSA")
    evaluation("sarsa")
