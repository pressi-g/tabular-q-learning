import gymnasium as gym
import minigrid

def create_minigrid_environment(grid_type='MiniGrid-Empty-8x8-v0'):
    """
    Create and return a MiniGrid environment.

    Parameters:
        grid_type (str): The type of grid environment to create. Defaults to 'MiniGrid-Empty-8x8-v0'.

    Returns:
        gym.Env: A Gym environment object representing the MiniGrid environment.
    """
    env = gym.make(grid_type)
    return env

def main():
    # Create the MiniGrid environment
    env = create_minigrid_environment()

    # Reset the environment
    obs, info = env.reset()
    print(obs)
    print(info)

if __name__ == "__main__":
    main()
