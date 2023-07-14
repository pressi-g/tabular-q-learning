# Author: Preston Govender

from utils import save_q_values, load_q_values, create_minigrid_environment
from q_learning import q_learning
from sarsa import sarsa
from render_optimal_policy import render_optimal_policy


def main():
    # Create the MiniGrid environment
    env = create_minigrid_environment()

    # Best shared hyperparameters: {'alpha': 0.35, 'gamma': 0.9, 'epsilon': 0.35} (from grid search)
    # Best Q-learning hyperparameters: {'alpha': 0.5, 'gamma': 0.9, 'epsilon': 0.3} (from grid search)
    # Best SARSA hyperparameters: {'alpha': 0.2, 'gamma': 0.9, 'epsilon': 0.4} (from grid search)
    # Random seed: 5
    episodes = 3000  # number of episodes to train the agent
    alpha = 0.35  # learning rate
    gamma = 0.9  # discount factor
    epsilon = 0.35  # exploration rate

    Q, q_average_rewards, q_average_steps, q_final_steps, q_final_reward = q_learning(
        env, episodes, alpha, gamma, epsilon
    )
    print("\n")
    print("Q-Learning")
    print("Average rewards:", q_average_rewards)
    print("Average steps:", q_average_steps)
    print("Q-values:", Q)
    print("\n")

    (
        SARSA,
        sarsa_average_rewards,
        sarsa_average_steps,
        sarsa_final_steps,
        sarsa_final_reward,
    ) = sarsa(env, episodes, alpha, gamma, epsilon)
    print("\n")
    print("SARSA")
    print("Average rewards:", sarsa_average_rewards)
    print("Average steps:", sarsa_average_steps)
    print("Q-values:", SARSA)

    # Save the Q-values
    save_q_values(Q, "q_values_q_learning.pkl")
    save_q_values(SARSA, "q_values_sarsa.pkl")


if __name__ == "__main__":
    main()
