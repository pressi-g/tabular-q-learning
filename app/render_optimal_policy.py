# Author: Preston Govender

import pickle
from os.path import exists
import numpy as np
import gymnasium as gym
from minigrid.wrappers import *
import time
from utils import *
import metrohash
import pygame


def render_optimal_policy(title):
    # Load the Q-values
    Q = load_q_values(f"q_values_{title}.pkl")

    # Create the environment
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="human")
    env = ImgObsWrapper(env)

    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption(title)  # Set the window title

    # Reset the environment
    obs, _ = env.reset()

    # Render the optimal policy
    done = False
    while not done:
        # Extract the current state
        currentS = extract_object_information(obs)

        # Get the optimal action based on Q-values
        currentS_Hash = metrohash.hash64_int(currentS)
        action = np.argmax(Q[currentS_Hash])

        # Perform the action
        obs, _, done, _, _ = env.step(action)

        # Render the environment
        env.render()

        # Wait a bit
        time.sleep(0.2)

    # Close the environment
    env.close()


def main():
    # Render optimal policy for Q-learning
    render_optimal_policy("q_learning")

    # Render optimal policy for SARSA
    render_optimal_policy("sarsa")


if __name__ == "__main__":
    main()
