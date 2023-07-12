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
random.seed(69)


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

    # Use a wrapper so the observation only contains the grid information
    env = ImgObsWrapper(env)

    print("Start Q-learning training...")
    steps_done = 0  # Counter for total number of training steps taken
    average_rewards = []
    average_steps = []

    for e in range(episodes):
        # reset the environment
        obs, _ = env.reset()

        # extract the current state from the observation
        currentS = []
        currentS = extract_object_information(obs)

        total_reward = 0
        total_steps = 0

        # decay epsilon
        epsilon = max(epsilon * 0.999, 0.05)

        for i in range(max_steps):
            # Choose an action using epsilon-greedy action selection
            currentS_Hash = metrohash.hash64_int(currentS)

            # Increment the 'steps_done' counter
            steps_done += 1

            action = epsilon_greedy_action(Q, currentS_Hash, numActions, epsilon)

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

            total_reward += reward
            total_steps += 1

            # if done:
            #     # if agent reached its goal successfully
            #     print(
            #         "Finished episode successfully taking %d steps and receiving reward %f"
            #         % (i, reward)
            #     )
            #     break

            # if truncated:
            #     # agent failed to reach its goal successfully
            #     print(
            #         "Truncated episode taking %d steps and receiving reward %f"
            #         % (i, reward)
            #     )
            #     break

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

            # since the episode is not done, update the current state
            currentS = nextS

        # Log the reward and steps per step to TensorBoard
        writer.add_scalar("Q-learning: Reward/train", reward, e)
        writer.add_scalar("Q-learning: Steps/train", total_steps - 1, e)
        writer.flush()

        average_rewards.append(total_reward)
        average_steps.append(total_steps - 1)

        # get the last 100 episodes to calculate the average reward and steps
        avg_reward = np.mean(average_rewards[-100:])
        avg_steps = np.mean(average_steps[-100:])
        final_steps = average_steps[-1]
        final_reward = average_rewards[-1]

    print("Done Q-learning training...")
    writer.close()

    return Q, avg_reward, avg_steps, final_steps, final_reward
