# import the 'random' module to generate random numbers
import random

# import the environment module
from environment import create_minigrid_environment

def random_agent(env):
    """
    Random agent that generates random actions at each step.

    Parameters:
        env (gym.Env): The Gym environment.

    Returns:
        None
    """
    done = False # set done to False to initialize the episode
    obs = env.reset() # reset the environment and get the initial observation

    while not done:
        # Generate a random action
        action = random.randint(0, env.action_space.n - 1)
        print("Action a: %d was generated" % action)

        # Take the action in the environment
        obs, reward, done, info = env.step(action)

        print("Action a: %d was generated" % action)

        # If done is True, the episode is finished
        if done:
            print("Episode finished")
            print("Reward: %d" % reward)
            print("Info: %s" % info)
            return action, reward, done, info

def main():
    # Create the MiniGrid environment
    env = create_minigrid_environment()

    # Run the random agent
    random_agent(env)

if __name__ == "__main__":
    main()
