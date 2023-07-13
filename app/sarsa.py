# RL imports
import numpy as np
from minigrid.wrappers import *
from os.path import exists

# hashing imports
import metrohash

# tensorboard imports
from torch.utils.tensorboard import SummaryWriter

# import functions
from utils import *

# set random seed
random.seed(5)


def sarsa(env, episodes, alpha, gamma, epsilon):
    """
    Train the agent using SARSA.

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
    numActions = 3 # env.action_space.n

    # Use a wrapper so the observation only contains the grid information
    env = ImgObsWrapper(env)

    print("Start SARSA training...")
    steps_done = 0  # Counter for total number of training steps taken
    average_rewards = []
    average_steps = []

    for e in range(episodes):
        # reset the environment
        obs, _ = env.reset()

        # extract the current state from the observation
        currentS = extract_object_information(obs)

        total_reward = 0
        total_steps = 0

        # decay epsilon
        epsilon = max(epsilon * 0.999, 0.05)

        # Choose an action using epsilon-greedy action selection
        currentS_Hash = metrohash.hash64_int(currentS)
        action = epsilon_greedy_action(Q, currentS_Hash, numActions, epsilon)

        for i in range(max_steps):
            # Increment the 'steps_done' counter
            steps_done += 1

            # take the action in the environment
            next_obs, reward, done, truncated, _ = env.step(action)

            # extract the next state from the observation
            nextS = extract_object_information(next_obs)
            nextS_Hash = metrohash.hash64_int(nextS)

            # Choose the next action using epsilon-greedy action selection
            next_action = epsilon_greedy_action(Q, nextS_Hash, numActions, epsilon)

            # Update the Q-value for the current state-action pair
            if currentS_Hash not in Q:
                #Q[currentS_Hash] = np.zeros(numActions)
                Q[currentS_Hash] = np.random.rand(numActions)
            if nextS_Hash not in Q:
                #Q[nextS_Hash] = np.zeros(numActions)
                Q[nextS_Hash] = np.random.rand(numActions)

            Q[currentS_Hash][action] += alpha * (
                reward + gamma * Q[nextS_Hash][next_action] - Q[currentS_Hash][action]
            )

            total_reward += reward
            total_steps += 1

            if done:
                # if agent reached its goal successfully
                if e >= episodes - 10:
                    print(
                        "Finished episode successfully taking %d steps and receiving reward %f"
                        % (i, reward)
                    )
                break

            if truncated:
                # agent failed to reach its goal successfully
                if e >= episodes - 10:
                    print(
                        "Truncated episode taking %d steps and receiving reward %f"
                        % (i, reward)
                    )
                break

            # Update the current state and action for the next step
            currentS = nextS
            currentS_Hash = nextS_Hash  # Update currentS_Hash
            action = next_action

        # Log the reward and steps per episode to TensorBoard
        writer.add_scalar("SARSA: Reward/train", total_reward, e)
        writer.add_scalar("SARSA: Steps/train", total_steps, e)
        writer.flush()

        average_rewards.append(total_reward)
        average_steps.append(total_steps)

        # Get the last 100 episodes to calculate the average reward and steps
        avg_reward = np.mean(average_rewards[-100:])
        avg_steps = np.mean(average_steps[-100:])
        final_steps = average_steps[-1]
        final_reward = average_rewards[-1]

    print("Done SARSA training...")
    writer.close()

    return Q, avg_reward, avg_steps, final_steps, final_reward
