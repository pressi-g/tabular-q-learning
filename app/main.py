from render_optimal_policy import render_optimal_policy
from evaluation import evaluation


def main():
    print("Evaluating Q-learning")
    evaluation("q_learning")
    print("\n")
    # Render optimal policy for Q-learning
    print("Rendering optimal policy for Q-learning")
    render_optimal_policy("q_learning")
    print("\n")

    print("Evaluating SARSA")
    evaluation("sarsa")
    print("\n")
    # Render optimal policy for SARSA
    print("Rendering optimal policy for SARSA")
    render_optimal_policy("sarsa")


if __name__ == "__main__":
    main()
