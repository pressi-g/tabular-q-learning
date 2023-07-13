# Tabular Q-Learning and SARSA on MiniGridWorld

This repository contains an implementation of Tabular Q-Learning and SARSA applied to a MiniGridWorld problem using the Farama-Foundation's [minigrid](https://github.com/Farama-Foundation/Minigrid) Gymnasium environment. The agent is restricted to 3 actions (forward, turn left, and turn right). 
The environment which the agent needs to navigate is a 2D grid, that represents an empty room, and the agents goal is to reach the green goal square, which provides a sparse reward. A small penalty is subtracted for the number of steps to reach the goal.

## Introduction

Tabular Q-Learning and SARSA are reinforcement learning techniques that allow an agent to learn an optimal policy for a given environment. In this case, we use a MiniGridWorld environment, which contains an empty grid world with sparse reward (the reward is achieved when the goal is met). The agent learns by interacting with the environment, observing the state and taking actions to maximize its cumulative reward.

Q-Learning is an off-policy learning algorithm that learns the optimal action-value function (Q-values) for each state-action pair. It uses a table (or dictionary) to store the Q-values and updates them based on the observed rewards and the maximum Q-value of the next state.

SARSA, on the other hand, is an on-policy learning algorithm that learns the action-value function by updating the Q-values based on the observed rewards and the action taken in the next state. It stands for State-Action-Reward-State-Action, indicating that it considers the next action as well.

Both Q-Learning and SARSA are iterative algorithms that improve the agent's policy over time. They are known as model-free algorithms, as they do not require prior knowledge of the environment dynamics and rely solely on interactions with the environment.

## Installation

To run this code, you need to have Python 3.8 and pip installed on your system. Here are the steps to set up the environment:

1. Create a virtual environment using `conda` or `venv`:

   ```shell
   conda create -n rl-env python=3.8
   ```

   or

   ```shell
   python3 -m venv rl-env
   ```

   You can replace `rl-env` with any name you like for your virtual environment.
   
   `conda` is recommended since this project was developed using `conda`.

2. Clone this repository:

   ```
   git clone https://github.com/pressi_g/tabular-q-learning.git
   ```

3. Change into the project directory:

   ```
   cd tabular-q-learning
   ```

4. Activate the virtual environment:

   ```shell
   conda activate rl-env
   ```

   or

   ```shell
   source rl-env/bin/activate
   ```

5. Install the required dependencies using pip:

   ```
   pip install -r requirements.txt
   ```

6. *Optional*

   Run the training and evaluation script:

   ```
   python3 main.py
   ```

7. Render the optimal policy:

   ```
   python3 render_optimal_policy.py
   ```

8. *Optional*:
   
   You can view the training and evaluation results for the grid search by running:

   ```
   tensorboard --logdir=grid-search-runs --port=8008
   ```
   You can view the training and evaluation results for the optimal policy by running:
   ```
   tensorboard --logdir=runs    
   ```
## Directory Structure

The structure of this repository is as follows:
```
└── app
|  ├── .gitignore
|  ├── grid_search.py
|  ├── joint_hyperparameter_results.csv
|  ├── main.py
|  ├── output_logs.txt
|  ├── q_learning_hyperparameter_results.csv
|  ├── q_learning.py
|  ├── q_values_q_learning.pkl
|  ├── q_values_sarsa.pkl
|  ├── render_optimal_policy.py
|  ├── run_qrid_search.py
|  ├── sarsa_hyperparameter_results.csv
|  ├── sarsa.py
|  └──  utils.py
└── README.md
└── requirements.txt
└── tests
   └── unit-tests
   |   ├── ...
   |   └── ...
   └── integration-tests
      ├── ...
      └── ...

```

Here's a brief description of the files in this repository:

The directory structure consists of the following components:

- `app`: Contains the main application code.
  - `.gitignore`: Specifies files and directories to be ignored by Git version control.
  - `grid_search.py`: Implements the grid search algorithm to find optimal hyperparameters.
  - `joint_hyperparameter_results.csv`: CSV file containing the results of the joint hyperparameter search.
  - `main.py`: Main script to run and train the agent in the MiniGridWorld environment.
  - `output_logs.txt`: Text file containing the output logs of the application.
  - `q_learning_hyperparameter_results.csv`: CSV file containing the results of Q-Learning hyperparameter search.
  - `q_learning.py`: Implements the Q-Learning algorithm.
  - `q_values_q_learning.pkl`: Pickle file to store the learned Q-values from Q-Learning.
  - `q_values_sarsa.pkl`: Pickle file to store the learned Q-values from SARSA.
  - `render_optimal_policy.py`: Renders the optimal policy learned by the agent.
  - `run_grid_search.py`: Script to run the grid search for optimal hyperparameters.
  - `sarsa_hyperparameter_results.csv`: CSV file containing the results of SARSA hyperparameter search.
  - `sarsa.py`: Implements the SARSA algorithm.
  - `utils.py`: Contains utility functions used in the application.
- `README.md`: The main documentation file providing an overview of the project.
- `requirements.txt`: Lists the required Python dependencies for the project.
- `tests`: Contains unit and integration tests for the application.
  - `unit-tests`: Directory for unit tests.
  - `integration-tests`: Directory for integration tests.

Feel free to explore the code files, modify them, and experiment with different settings to understand and improve the Tabular Q-Learning and SARSA algorithms on MiniGridWorld.


## Testing

This project includes test cases to ensure the correctness of the implemented algorithms and functionalities. The tests are written using the `pytest` framework. To run the tests, follow these steps:

1. Make sure you have installed the project dependencies as mentioned in the installation section.

2. Open a terminal or command prompt and navigate to the project directory.

3. Install `pytest` using `pip`:

   ```shell
   pip install pytest
   ```

4. Once `pytest` is installed, you can run the tests by executing the following command:

   ```shell
   pytest
   ```

   This command will discover and execute all the test cases in the project.

   Note: The test files should be named with the prefix `test_` for `pytest` to automatically detect them.

5. After running the tests, you will see the test results in the terminal or command prompt, indicating which tests passed or failed.

If you encounter any issues or failures during the test execution, please feel free to open an issue in this repository for assistance.