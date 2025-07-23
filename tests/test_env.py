# tests/test_env.py

import pytest
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
# Assuming CityEnvironment is the main class exposed by the package
from CityEnvGym.CityEnvGym import CityEnvironment
import CityEnvGym
import time

@pytest.fixture
def env():
    """Pytest fixture to create a CityEnv instance for testing."""
    # This correctly instantiates the environment for each test
    obstacle_map = [[False for _ in range(int(100))] for _ in range(int(100))]

    # obstacle_map[5][5] = True  # Example obstacle
    # obstacle_map[7][7] = True  # Another example obstacle

    target_physics = {"mass":5,"max_speed":15.0,"max_angular_velocity":np.pi/4.0,}
    drone_physics = {"mass":5,"max_speed":15.0,"max_angular_velocity":np.pi/4.0,}

    sensors = [[0.0,0.0,25.0],[-50.0,-50.0,25],[50.0,50.0,25],[-50.0,50.0,25],[50.0,-50.0,25]] # x,y ,radius

    env = gym.make("CityEnvGym/CityEnv-v0", render_mode="human",obstacle_map=obstacle_map,sensors=sensors,num_evader_steps=50,max_episode_steps=18000, time_step=1/60.0, fov_angle=90.0, fov_distance=100.0,target_physics=target_physics, drone_physics=drone_physics)
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

    future_evader_positions = obs["future_evader_positions"]
    assert future_evader_positions.shape == (env.unwrapped.num_evader_steps, 2), f"Future evader positions should have shape (num_evader_steps, 2), got {future_evader_positions.shape}"

    assert rew is not None, "Reward should not be None"


# def test_episode_time_limit(env):
#     """
#     Test if the environment correctly handles the episode time limit.
#     """

#     action = np.array([15.0, 15.0, 0.0], dtype=np.float32)
#     times = []
    
#     for i in range(25):
#         env.reset()``
        
#         done = False
#         truncated = False
#         start_time = time.time()
#         while not (done or truncated):
#             obs, rew, done, truncated, info = env.step(action)
#         end_time = time.time()

#         elapsed_time = end_time - start_time
#         times.append(elapsed_time)

#     print(f"Elapsed time: {np.mean(times):.2f} seconds")
#     assert done or truncated, "Episode should be done when time limit is reached"


def test_observation_wrapper(env):
    """
    Test the EgoCentricObservationWrapper.
    """
    from CityEnvGym.wrappers import EgoCentricObservationWrapper

    wrapped_env = EgoCentricObservationWrapper(env)
    
    obs, info = wrapped_env.reset()
    
    assert isinstance(obs, dict), "Wrapped observation should be a dictionary"
    
    # Check if drone and target positions are transformed correctly
    drone_pos = obs['drone']
    target_pos = obs['target']


    







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