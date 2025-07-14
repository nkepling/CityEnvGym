# tests/test_env.py

import pytest
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
# Assuming CityEnvironment is the main class exposed by the package
from CityEnvGym.CityEnvGym import CityEnvironment
import CityEnvGym

@pytest.fixture
def env():
    """Pytest fixture to create a CityEnv instance for testing."""
    # This correctly instantiates the environment for each test
    obstacle_map = [[False for _ in range(int(200))] for _ in range(int(200))]

    sensors = [[0.0,0.0,25.0],[-50.0,-50.0,25],[50.0,50.0,25],[-50.0,50.0,25],[50.0,-50.0,25]] # x,y ,radius

    env = gym.make("CityEnvGym/CityEnv-v0", render_mode="human",obstacle_map=obstacle_map,sensors=sensors,num_evader_steps=1000,max_episode_steps=10000)
    return env

def test_env_initialization(env):
    """
    Test if the environment is created successfully and has the correct spaces.
    """
    
    assert isinstance(env.observation_space, gym.spaces.Dict), "Observation space should be a Dict space"
    assert isinstance(env.action_space, gym.spaces.Box), "Action space should be a Box space"



def test_env_reset(env):
    """
    Test the reset method for correct output and initial values.
    """
    obs, info = env.reset()


    assert isinstance(obs, dict), "Observation should be a dictionary"
    assert "drone" in obs, "Observation should contain 'drone' key"

    assert "target" in obs, "Observation should contain 'target' key"


def test_env_step(env):
    """
    Test the step method with a valid action.
    """
    # Create a valid action within the action space bounds
    action = np.array([15.0, 15.0, 0.0], dtype=np.float32)
    
    # Step the environment
    out = env.reset()
    obs, rew, done, truncated, info = env.step(action)


    # Check if the observation is of correct type and shape
    assert isinstance(obs, dict), "Observation should be a dictionary"

def test_render(env):
    """
    Test the render method.
    """
    # Call the render method
    obs,info = env.reset()
    action = np.array([15.0, 15.0, 0.0], dtype=np.float32)
    done = False
    truncated = False
    while not (done or truncated):
        obs, rew, done, truncated, info = env.step(action)    


        env.render()

    assert True, "Render method should run without error"







if __name__ == "__main__":
    pytest.main([__file__])