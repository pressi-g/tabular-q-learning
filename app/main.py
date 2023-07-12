# Author: Preston Govender
from utils import save_q_values, load_q_values, create_minigrid_environment
from q_learning import q_learning
from sarsa import sarsa


# def main():
#     # Create the MiniGrid environment
#     env = create_minigrid_environment()


#     episodes = 3000  # number of episodes to train the agent
#     alpha = 0.4  # learning rate
#     gamma = 0.9  # discount factor
#     epsilon = 0.8  # exploration rate

#     Q = q_learning(env, episodes, alpha, gamma, epsilon)
#     # SARSA = sarsa(env, episodes, alpha, gamma, epsilon)

#     # Save the Q-values
#     save_q_values(Q, 'q_values.pkl')


# if __name__ == "__main__":
#     main()


# def main():
#     # Create the MiniGrid environment
#     env = create_minigrid_environment()

#     # Define the hyperparameter ranges to search
#     hyperparams = {
#         'alpha': np.linspace(0.1, 0.9, 9),
#         'gamma': [0.9],
#         'epsilon': np.linspace(0.1, 0.9, 9)
#     }

#     episodes = 3000  # Number of episodes to evaluate for each hyperparameter combination

#     best_hyperparams, best_average_rewards, best_average_steps = grid_search(env, hyperparams, episodes)
#     print("Best hyperparameters:", best_hyperparams)
#     print("Best average rewards:", best_average_rewards)
#     print("Best average steps:", best_average_steps)

# if __name__ == "__main__":
#     main()


def main():
    # Create the MiniGrid environment
    env = create_minigrid_environment()

    # Best hyperparameters: {'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.5} (from grid search)
    episodes = 3000  # number of episodes to train the agent
    alpha = 0.1  # learning rate
    gamma = 0.9  # discount factor
    epsilon = 0.5  # exploration rate
    # Best average rewards: 0.9584101562499997 (from grid search)
    # Best average steps: 10.83 (from grid search)

    Q, q_average_rewards, q_average_steps, q_final_steps, q_final_reward = q_learning(
        env, episodes, alpha, gamma, epsilon
    )
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
    # save_q_values(Q, 'q_values_q_learning.pkl')
    save_q_values(SARSA, "q_values_sarsa.pkl")

    # # load the Q-values
    # Q_q_learning = load_q_values('q_values_q_learning.pkl')
    # Q_sarsa = load_q_values('q_values_sarsa.pkl')


if __name__ == "__main__":
    main()
