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
def env_multi_agent():
    target_physics = {"mass":5,"max_speed":15.0,"max_angular_velocity":np.pi/4.0,}
    drone_physics = {"mass":5,"max_speed":15.0,"max_angular_velocity":np.pi/4.0,}

    target_initial_position = np.array([0.0, 0.0], dtype=np.float32)





    env = gym.make("CityEnvGym/MultiAgentCityEnv-v0", render_mode="human",num_evader_steps=50,max_episode_steps=18000, time_step=1/60.0, fov_angle=90.0, fov_distance=100.0,target_physics=target_physics, drone_physics=drone_physics, target_initial_position=target_initial_position)
    return env



def test_multi_agent(env_multi_agent):
    env = env_multi_agent

    pursuer_action = np.array([0.0, -20.0, 0.0], dtype=np.float32)
    evader_action = np.array([0.0, 20.0, 0.0], dtype=np.float32)


    joint_action = {
        "drone": pursuer_action, 
        "target": evader_action
    }

    done = False
    truncated = False   

    obs, info = env.reset()
    assert type(env.action_space) == gym.spaces.Dict, "Action space should be a Dict space"

    assert (obs["drone"] == np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)).all()
    assert (obs["target"] == np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)).all()

    for i in range(10):
        obs, rew, done, truncated, info = env.step(joint_action)

        print("drone", obs["drone"])
        print("target", obs["target"])



    
# def test_render(env_multi_agent):
#     env = env_multi_agent
    
#     pursuer_action = np.array([0.0, -1.0, 0.0], dtype=np.float32)
#     evader_action = np.array([0.0, 0.7, 0.0], dtype=np.float32)


#     joint_action = {"drone": pursuer_action, "target": evader_action}

#     done = False
#     truncated = False   

#     obs, info = env.reset()
#     assert type(env.action_space) == gym.spaces.Dict, "Action space should be a Dict space"

#     assert (obs["drone"] == np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)).all()
#     assert (obs["target"] == np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)).all()

#     for i in range(10000):
#         obs, rew, done, truncated, info = env.step(joint_action)

#         print("drone", obs["drone"])
#         print("target", obs["target"])

#         env.render()


def test_reset(env_multi_agent):
    env = env_multi_agent
    obs, info = env.reset()
    assert isinstance(obs, dict), "Observation should be a dictionary"
    assert "drone" in obs, "Observation should contain 'drone' key"
    assert "target" in obs, "Observation should contain 'target' key"


if __name__ == "__main__":
    pytest.main([__file__])