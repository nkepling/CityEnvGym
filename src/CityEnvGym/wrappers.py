from gymnasium import ObservationWrapper
from .utils import coordinate_transformation,rotate_vector, transform_trajectory_to_ego
import numpy as np



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
