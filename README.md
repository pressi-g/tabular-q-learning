# Tabular Q-Learning on MiniGridWorld

This repository contains an implementation of Tabular Q-Learning applied to a MiniGridWorld problem using the OpenAI Gymnasium environment.

## Introduction

Tabular Q-Learning is a reinforcement learning technique that allows an agent to learn an optimal policy for a given environment. In this case, we use the MiniGridWorld environment from OpenAI Gymnasium, which represents a grid world with various objects and agents. The goal is to train an agent to navigate the grid world and achieve a specific objective, such as reaching a target location while avoiding obstacles.

## Installation

To run this code, you need to have Python 3.8 and pip installed on your system. Here are the steps to set up the environment:

1. Clone this repository:

   ```
   git clone https://github.com/pressi_g/tabular-q-learning.git
   ```

2. Change into the project directory:

   ```
   cd tabular-q-learning
   ```

3. Install the required dependencies using pip:

   ```
   pip install -r requirements.txt
   ```

## Directory Structure

The structure of this repository is as follows:

```
├── agent.py            # Contains the Tabular Q-Learning agent implementation
├── environment.py      # Implements the MiniGridWorld environment using Gymnasium
├── main.py             # Entry point for running the training and evaluation
├── README.md           # This file, providing an overview of the repository
├── requirements.txt    # List of dependencies required to run the code
└── utils.py            # Utility functions for visualizing the environment and results
```

Here's a brief description of the files in this repository:

- `agent.py`: This file contains the implementation of the Tabular Q-Learning agent, including the Q-table and learning algorithm.
- `environment.py`: This file implements the MiniGridWorld environment using the OpenAI Gymnasium library, providing the necessary functions to interact with the environment.
- `main.py`: The main entry point for running the training and evaluation process. It sets up the environment, creates an agent, and runs the training loop.
- `README.md`: The file you are currently reading, providing an overview of the repository and instructions for setup.
- `requirements.txt`: A file that lists the dependencies required to run the code. You can install these dependencies using `pip` as mentioned in the installation section.
- `utils.py`: Contains utility functions for visualizing the environment and displaying results, which can aid in understanding the agent's behavior.

Feel free to explore the code files, modify them, and experiment with different settings to understand and improve the Tabular Q-Learning algorithm on MiniGridWorld.


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

It's important to regularly run the tests to ensure the correctness of your implementation, especially when modifying the code or adding new features. Writing tests for different scenarios and edge cases can help uncover potential bugs and provide confidence in the functionality of your project.

Feel free to expand the test suite by adding more test cases as you develop your project further.

If you encounter any issues or failures during the test execution, please feel free to open an issue in this repository for assistance.

If you have any questions or run into issues, please open an issue in this repository, and I'll be happy to assist you.
```