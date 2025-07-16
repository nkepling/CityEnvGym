from gymnasium import ObservationWrapper
from .utils import coordinate_transformation
import numpy as np

class EgoCentricObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):

        drone_pos = obs['drone']
        target_pos = obs['target']
        future_target_trajectory = obs["future_evader_positions"]

        ego_drone = coordinate_transformation(drone_pos, drone_pos)
        ego_centric_target = coordinate_transformation(target_pos, drone_pos)
        ego_centric_future_target = np.array([coordinate_transformation(pos, drone_pos) for pos in future_target_trajectory])


        obs['drone'] = ego_drone
        obs['target'] = ego_centric_target
        obs['future_evader_positions'] = ego_centric_future_target
        return obs  