# tests/test_env.py

import pytest
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
# Assuming CityEnvironment is the main class exposed by the package
from CityEnvGym.CityEnvGym import CityEnvironment
import CityEnvGym
import time

env_params = {
    'max_time': 300.0, # max tim in seconds (5 minutes)
    'num_evader_steps': 50,  
    'render_mode': 'human',
    'time_step': 1/60.0,
    'fov_angle': 90.0,
    'fov_distance': 50.0,  
    'max_episode_steps': 18000, # 5 minutes sim time (1/60 seconds * 18000 steps = 300 seconds)

    'obstacle_map': [[False for _ in range(int(100))] for _ in range(int(100))],
    'sensors':  [[0.0,0.0,25.0],[-50.0,-50.0,25],[50.0,50.0,25],[-50.0,50.0,25],[50.0,-50.0,25]],

    'target_physics': {
        'mass': 1.0,                        # Light
        'moment_of_inertia': 0.1,           # Can turn easily
        'linear_drag_coeff': 2.0,           # High drag to fight the insane engine
        'angular_drag_coeff': 2.0,          # High angular drag
        'propulsion_gain': 1000.0,           # Overpowered engine causes overshoot
        'steering_gain': 10.0,              # Overpowered steering causes oscillation
        'max_speed': 30.0,                  # Low max speed due to instability
        'max_angular_velocity': np.pi * 4,  # Can twitch and spin very fast
    },
    
    'drone_physics': {
        'mass': 1.0,
        'moment_of_inertia': 0.1,
        'linear_drag_coeff': 0.1,
        'angular_drag_coeff': 0.1,
        'propulsion_gain': 5.0,
        'steering_gain': 2.0,
        'max_speed': 15.0,
        'max_angular_velocity': np.pi / 2,  
    }
}

@pytest.fixture
def env():
    """Pytest fixture to create a CityEnv instance for testing."""
    # This correctly instantiates the environment for each test
    # obstacle_map = [[False for _ in range(int(100))] for _ in range(int(100))]

    # obstacle_map[5][5] = True  # Example obstacle
    # obstacle_map[7][7] = True  # Another example obstacle

    target_physics = {"mass":5,"max_speed":15.0,"max_angular_velocity":np.pi/4.0,}
    drone_physics = {"mass":5,"max_speed":15.0,"max_angular_velocity":np.pi/4.0,}

    sensors = [[0.0,0.0,25.0],[-50.0,-50.0,25],[50.0,50.0,25],[-50.0,50.0,25],[50.0,-50.0,25]] # x,y ,radius

    env = gym.make("CityEnvGym/CityEnv-v0", render_mode="human",sensors=sensors,num_evader_steps=50,max_episode_steps=18000, time_step=1/60.0, fov_angle=90.0, fov_distance=100.0,target_physics=target_physics, drone_physics=drone_physics)
    return env

@pytest.fixture
def env_with_init_position():

    target_physics = {"mass":5,"max_speed":15.0,"max_angular_velocity":np.pi/4.0,}
    drone_physics = {"mass":5,"max_speed":15.0,"max_angular_velocity":np.pi/4.0,}

    sensors = [[0.0,0.0,25.0],[-50.0,-50.0,25],[50.0,50.0,25],[-50.0,50.0,25],[50.0,-50.0,25]] # x,y ,radius

    env = gym.make("CityEnvGym/CityEnv-v0", render_mode="human",sensors=sensors,num_evader_steps=50,max_episode_steps=18000, time_step=1/60.0, fov_angle=90.0, fov_distance=100.0,target_physics=target_physics, drone_physics=drone_physics,target_initial_position=np.array([0.0,0.0],dtype=np.float32))
    return env


@pytest.fixture
def env_with_big_config():
    target_physics = {"mass":5,"max_speed":15.0,"max_angular_velocity":np.pi/4.0,}
    drone_physics = {"mass":5,"max_speed":15.0,"max_angular_velocity":np.pi/4.0,}

    sensors = [[0.0,0.0,25.0],[-50.0,-50.0,25],[50.0,50.0,25],[-50.0,50.0,25],[50.0,-50.0,25]] # x,y ,radius

    env = gym.make("CityEnvGym/CityEnv-v0", **env_params)
    return env



@pytest.fixture
def env_with_rgb():

    target_physics = {"mass":5,"max_speed":15.0,"max_angular_velocity":np.pi/4.0,}
    drone_physics = {"mass":5,"max_speed":15.0,"max_angular_velocity":np.pi/4.0,}

    sensors = [[0.0,0.0,25.0],[-50.0,-50.0,25],[50.0,50.0,25],[-50.0,50.0,25],[50.0,-50.0,25]] # x,y ,radius

    env = gym.make("CityEnvGym/CityEnv-v0", render_mode="rgb_array",sensors=sensors,num_evader_steps=50,max_episode_steps=18000, time_step=1/60.0, fov_angle=90.0, fov_distance=100.0,target_physics=target_physics, drone_physics=drone_physics,target_initial_position=np.array([0.0,0.0],dtype=np.float32))
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

    assert drone_pos.shape == (5,), "Drone observation should have shape (5,)"
    assert target_pos.shape == (5,), "Target observation should have shape (5,)"





def test_render(env):
    """
    Test the render method.
    """
    # Call the render method
    obs,info = env.reset()
    action = np.array([15.0, 15.0, 0.0], dtype=np.float32)
    done = False
    truncated = False
    start = time.time()
    while not (done or truncated):
        obs, rew, done, truncated, info = env.step(action)    
        env.render()

        if (time.time() - start) > 20:
            break

    env.close()

    assert True, "Render method should run without error"

def test_set_seed(env):
    """
    Test the set_seed method for correct behavior.
    """

    obs, info = env.reset(seed=42)


def test_random_starts(env):
    
    seed = 100
    obs, info = env.reset(seed=seed)
    prev_start = obs["target"]

    seed = 24
    obs, info = env.reset(seed=seed)
    new_start = obs["target"]
    assert not np.array_equal(prev_start, new_start), "Different seeds should produce different starting positions"

    # check that start is always valid



def test_set_target_position(env_with_init_position):

    obs,info = env_with_init_position.reset()
    start = obs["target"]
    assert np.array_equal(start, np.array([0.0, 0.0,0.0,0.0,0.0], dtype=np.float32)), "Target initial position is not set correctly"

# def test_reset_no_seed(env):

#     obs,info = env.reset()
#     prev_pos = obs["target"]


#     for i in range(10):

#         obs, info = env.reset()

#         assert not np.array_equal(obs["target"], prev_pos), "Target position should be consistent across resets"
#         prev_pos = obs["target"]


def test_big_config(env_with_big_config):
    """
    Test the environment with a larger configuration.
    """
    obs,info = env_with_big_config.reset()
    action = np.array([15.0, 15.0, 0.0], dtype=np.float32)
    done = False
    truncated = False
    while not (done or truncated):
        obs, rew, done, truncated, info = env_with_big_config.step(action)    


def test_render_rgb_array(env_with_rgb):
    """
    Tests if the render method with 'rgb_array' mode returns a valid image frame.
    """
    env = env_with_rgb
    # 1. Initialize the environment specifically for rgb_array rendering

    
    try:
        env.reset()
        
        # 2. Call the render method to get the frame
        frame = env.render()
        
        # 3. Assert that the frame is a valid image array
        assert frame is not None, "Render method should return a frame, not None."
        assert isinstance(frame, np.ndarray), "The frame should be a NumPy array."
        assert frame.dtype == np.uint8, f"Frame dtype should be np.uint8, but got {frame.dtype}."
        
        assert frame.ndim == 3, "The frame should be 3-dimensional (Height, Width, Channels)."
        height, width, channels = frame.shape
        assert height > 0 and width > 0, "Frame dimensions (height and width) must be positive."
        assert channels == 3, "Frame must have 3 color channels (R, G, B)."
        
    finally:
        # 4. Ensure the environment is properly closed to free resources
        env.close()


if __name__ == "__main__":
    pytest.main([__file__])