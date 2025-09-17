import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ._CityEnvGym import MultiAgentCityEnv, Drone, Target, State,Sensor
from typing import Any, SupportsFloat
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .utils import _load_map_from_image, _update_physics
from CityEnvGym import CityEnvGym


class MultiAgentCityEnvironment(CityEnvGym.CityEnvironment):
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)


        self.city_env = MultiAgentCityEnv(
            obstacle_map=self.obstacle_map,
            world_width=self.world_width,
            world_height=self.world_height,
            time_step=self.time_step,
            fov_angle=self.fov_angle,
            fov_distance=self.fov_distance,
            drone=self.drone, # Changed from 'drones'
            target=self.target,  # Changed from 'targets'
            sensors=self.sensors,
            origin = (-self.world_width / 2, -self.world_height / 2),  # Center the origin,
            seed=self.seed_value,
            target_initial_position=self.target_initial_position)
        

        self.action_space = spaces.Dict({
            "drone": spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(3,), 
            dtype=np.float32
        ),
            "target": spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(3,), 
            dtype=np.float32
        ),
        })


    def step(self, joint_action) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        drone_action, target_action = joint_action
        true_action_drone = self.true_action_low + (drone_action + 1.0) * 0.5 * (self.true_action_high - self.true_action_low)
        true_action_target = self.true_action_low + (target_action + 1.0) * 0.5 * (self.true_action_high - self.true_action_low)

        state = self.city_env.step(true_action_drone, true_action_target)
        drone_pos = state.drone.position
        drone_vel = state.drone.velocity
        drone_state = np.array([
            drone_pos.x(), 
            drone_pos.y(),
            drone_pos.yaw, 
            drone_vel[0], 
            drone_vel[1], 
        ], dtype=np.float32)

        future_pos_list = state.future_target_positions
        num_received_points = len(future_pos_list)
        padded_positions = np.zeros((self.num_evader_steps, 2), dtype=np.float32)

        if num_received_points > 0:
            received_positions = np.array(future_pos_list, dtype=np.float32)
            padded_positions[:num_received_points, :] = received_positions

        obs = {"drone": drone_state,
               "target": np.array([
                   state.target.position.x(), 
                   state.target.position.y(), 
                   state.target.position.yaw,
                   state.target.velocity[0],
                    state.target.velocity[1],
               ], dtype=np.float32),
               "future_evader_positions": padded_positions,
                "time_elapsed": np.array(state.time_elapsed, dtype=np.float32)
           }

        reward = state.reward  # Assuming the State object has a reward attribute
        done = state.time_elapsed >= self.max_time
        truncated = False  
        info = {"time_elapsed": state.time_elapsed}

        # check if there are nans/ infs in the observations
        for key, value in obs.items():
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                info[f"{key}_invalid"] = True

        # --- End Debugging Block ---
        return obs, reward, done, truncated, info
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        return super().reset(seed=seed, options=options)
    
    def render(self, window=100) -> None:
        return super().render(window)
    


