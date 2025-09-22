from typing import Any
from gymnasium import ObservationWrapper,Wrapper, ActionWrapper
from .utils import coordinate_transformation,rotate_vector, transform_trajectory_to_ego
import numpy as np
from gymnasium import spaces





class EgoCentricObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

 
    def observation(self, obs: dict) -> dict:
        # Extract world-frame poses and velocitiqes
        # We assume the format: [x, y, yaw, vx, vy]
        drone_state = obs['drone']

        drone_pos = drone_state[:3]  # Drone's position in world frame

        drone_yaw = drone_state[2]
        drone_vel = drone_state[3:]

        target_pos = obs['target']


        future_traj_pos = obs["future_evader_positions"]
       
        ego_drone = np.array([0, 0, 0])  # Drone's position in its own frame is always (0, 0, 0)
        ego_target_pos = coordinate_transformation(target_pos, drone_pos)
        ego_future_traj = transform_trajectory_to_ego(future_traj_pos, drone_pos)

        ego_drone_vel = rotate_vector(drone_vel, -drone_yaw)
        target_vel = rotate_vector(target_pos[3:], -drone_yaw)  # Assuming target also has vx, vy   
        
        new_obs = {

            "drone": np.concatenate([ego_drone,ego_drone_vel]).astype(np.float32),

            "target": np.concatenate([ego_target_pos, target_vel]).astype(np.float32), # Assuming target also has vx,vy

            "future_evader_positions": ego_future_traj.astype(np.float32)
        }
        
        return new_obs
    

class EgoCentricWrapperHistory(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # It's good practice to update the observation space in the wrapper
        # to reflect the new egocentric frame, but we'll omit that for brevity.

    def observation(self, obs: dict) -> dict:
        # --- 1. Get current drone state for reference ---
        drone_state = obs['drone']
        drone_pos_3d = drone_state[:3]  # World-frame [x, y, yaw]
        drone_yaw = drone_state[2]
        drone_vel_2d = drone_state[3:]  # World-frame [vx, vy]

        # --- 2. Transform the single target state ---
        target_state = obs['target']
        target_pos_3d = target_state[:3]
        target_vel_2d = target_state[3:]

        ego_target_pos = coordinate_transformation(target_pos_3d, drone_pos_3d)
        ego_target_vel = rotate_vector(target_vel_2d, -drone_yaw)

        # --- 3. Transform future and past trajectories ---
        future_traj_pos = obs["future_evader_positions"]
        evader_history = obs["past_evader_positions"]

        # Transform trajectories of (x, y) positions
        ego_future_traj = transform_trajectory_to_ego(future_traj_pos, drone_pos_3d)
        
        # ✅ CORRECTLY TRANSFORM THE HISTORY
        # Separate the history into its components
        history_positions = evader_history[:, :2]
        history_yaws = evader_history[:, 2]
        history_velocities = evader_history[:, 3:]

        # Transform the position part of the history
        ego_history_positions = transform_trajectory_to_ego(history_positions, drone_pos_3d)
        
        # Rotate each velocity vector in the history
        # Using a list comprehension for clarity
        ego_history_velocities = np.array([rotate_vector(v, -drone_yaw) for v in history_velocities])
        
        # Reconstruct the full egocentric history state
        # We keep the original yaw and combine it with the new ego positions and velocities
        ego_evader_history = np.hstack([
            ego_history_positions,
            history_yaws[:, np.newaxis], # Reshape yaw for concatenation
            ego_history_velocities
        ])

        # --- 4. Assemble the new observation dictionary ---
        new_obs = {
            "drone": np.array([0, 0, 0, *rotate_vector(drone_vel_2d, -drone_yaw)], dtype=np.float32),
            "target": np.concatenate([ego_target_pos, ego_target_vel]).astype(np.float32),
            "future_evader_positions": ego_future_traj.astype(np.float32),
            "past_evader_positions": ego_evader_history.astype(np.float32),
            "time_elapsed": obs["time_elapsed"]
        }
        
        return new_obs

    


    

class SingleAgentObservationWrapper(ActionWrapper):
    def __init__(self, env,evader_policy):
        super().__init__(env)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,), # Shape for a single agent (the drone)
            dtype=np.float32
        )

        self.evader_policy = evader_policy


    def action(self, pursuer_action):

        pursue_action = pursuer_action[:5]


        state = self.unwrapped._get_obs()

        evader_action = self.evader_policy(state)
        
        joint_action = {"drone": pursue_action, "target": evader_action}

        return joint_action
    


    
