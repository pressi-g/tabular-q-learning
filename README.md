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

If you have any questions or run into issues, please open an issue in this repository, and I'll be happy to assist you.
```

Feel free to customize the contents of the `README.md` file based on your specific implementation and requirements.