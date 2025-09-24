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

from collections import deque


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
            drone=self.drone, 
            target=self.target,
            sensors=self.sensors,
            origin = (-self.world_width / 2, -self.world_height / 2),  # Center the origin
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


      
        # --- FIX 1: Access observation_space from `self` not `super()` ---
        parent_obs_space = self.observation_space.spaces.copy() 

        # --- FIX 2: Corrected shape for past positions and removed trailing comma ---
        parent_obs_space["past_evader_positions"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_evader_steps, 5), # Shape is (history_length, state_dims)
                dtype=np.float32
            )

        self.observation_space = spaces.Dict(parent_obs_space)


        self.evader_history = deque(maxlen=self.num_evader_steps)

    def step(self, joint_action: dict) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:

        drone_action = joint_action["drone"]
        target_action = joint_action["target"]


        true_action_drone = self.true_action_low + (drone_action + 1.0) * 0.5 * (self.true_action_high - self.true_action_low)
        # true_action_target = self.target.physics.max_speed * target_action
        true_action_target = self.true_action_low + (target_action + 1.0) * 0.5 * (self.true_action_high - self.true_action_low)

        state = self.city_env.step(true_action_drone, true_action_target)
        
        drone_pos = state.drone.position
        drone_vel = state.drone.velocity
        drone_state = np.array([
            drone_pos.vector[0], 
            drone_pos.vector[1],
            drone_pos.yaw, 
            drone_vel[0], 
            drone_vel[1], 
        ], dtype=np.float32)

        target_pos = state.target.position
        target_vel = state.target.velocity
        target_state = np.array([
            target_pos.vector[0],
            target_pos.vector[1],
            target_pos.yaw,
            target_vel[0],
            target_vel[1],
        ], dtype=np.float32)

        # --- FIX 4: Correctly manage and format evader history ---
        self.evader_history.append(target_state)

        future_padded_positions = np.zeros((self.num_evader_steps, 2), dtype=np.float32)
        future_pos_list = state.future_target_positions
        if future_pos_list:
            num_received_points = len(future_pos_list)
            received_positions = np.array(future_pos_list, dtype=np.float32)
            future_padded_positions[:num_received_points, :] = received_positions
            
        past_padded_positions = np.zeros((self.num_evader_steps, 5), dtype=np.float32)
        history_array = np.array(self.evader_history)
        past_padded_positions[-len(history_array):, :] = history_array


        obs = {
            "drone": drone_state,
            "target": target_state,
            "future_evader_positions": future_padded_positions,
            "past_evader_positions": past_padded_positions,
            "time_elapsed": np.array(state.time_elapsed, dtype=np.float32),
        }

        reward = state.reward
        done = state.time_elapsed >= self.max_time
        truncated = False  
        info = {"time_elapsed": state.time_elapsed}

        for key, value in obs.items():
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                info[f"{key}_invalid"] = True
                print(f"Warning: Invalid value detected in observation key '{key}'")


        return obs, reward, done, truncated, info
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.evader_history.clear()

    
        obs,info = super().reset(seed=seed, options=options)   
        obs["past_evader_positions"] = np.zeros((self.num_evader_steps, 5), dtype=np.float32)
        return obs, info
    
    def render(self, window=100) -> None:
        return super().render(window)
    
    def _get_obs(self) -> dict:
        obs = self.city_env.get_state()
        return self._unpack_obs(obs)
    
    def _unpack_obs(self, state: Any) -> dict:
        drone_pos = state.drone.position
        drone_vel = state.drone.velocity
        drone_state = np.array([
            drone_pos.vector[0], 
            drone_pos.vector[1],
            drone_pos.yaw, 
            drone_vel[0], 
            drone_vel[1], 
        ], dtype=np.float32)

        target_pos = state.target.position
        target_vel = state.target.velocity
        target_state = np.array([
            target_pos.vector[0],
            target_pos.vector[1],
            target_pos.yaw,
            target_vel[0],
            target_vel[1],
        ], dtype=np.float32)

        self.evader_history.append(target_state)

        future_padded_positions = np.zeros((self.num_evader_steps, 2), dtype=np.float32)
        future_pos_list = state.future_target_positions
        if future_pos_list:
            num_received_points = len(future_pos_list)
            received_positions = np.array(future_pos_list, dtype=np.float32)
            future_padded_positions[:num_received_points, :] = received_positions
            
        past_padded_positions = np.zeros((self.num_evader_steps, 5), dtype=np.float32)
        history_array = np.array(self.evader_history)
        past_padded_positions[-len(history_array):, :] = history_array


        obs = {
            "drone": drone_state,
            "target": target_state,
            "future_evader_positions": future_padded_positions,
            "past_evader_positions": past_padded_positions,
            "time_elapsed": np.array(state.time_elapsed, dtype=np.float32),
        }

        return obs
        
    

if __name__ == "__main__":
    env = MultiAgentCityEnvironment(render_mode="human",num_evader_steps=50,max_time=300, time_step=1/60.0, fov_angle=90.0, fov_distance=100.0)
    obs, info = env.reset()
    done = False
    truncated = False   

    # --- FIX 5: Use a dictionary for the joint action to match the action space ---
    joint_action = {
        "drone": np.array([1.0, 0.0, 0.0], dtype=np.float32), # Move forward
        "target": np.array([0.7, 0.0, 0.0], dtype=np.float32)  # Move forward at 70% speed
    }


    while not done and not truncated:
        obs, rew, done, truncated, info = env.step(joint_action)

        print("drone", obs["drone"])
        print("target", obs["target"])

        env.render()