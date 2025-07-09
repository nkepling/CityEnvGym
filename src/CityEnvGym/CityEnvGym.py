import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ._CityEnvGym import CityEnv, Drone, Target, State
from typing import Any, SupportsFloat
from PIL import Image
import os
import matplotlib.pyplot as plt
from .utils import _load_map_from_image


class CityEnvironment(gym.Env):
    """
    CityEnv is a Gymnasium environment for simulating a single drone in a city.
    It provides methods to reset the environment, step through time, and render the state.
    """

    # The __init__ signature is simplified as the environment handles one drone/target internally
    def __init__(self, world_width: float = 1000.0, world_height: float = 1000.0, time_step: float = 1/60.0, fov_angle: float = 90.0, fov_distance: float = 100.0,max_time: float = 300.0, obstacle_map: list[list[bool]] | None = None,render_mode: str = "human") -> None:
        super().__init__()
        self.render_mode = render_mode
        drone = Drone()
        target = Target()
        self.max_time = max_time
        self.world_width = world_width
        self.world_height = world_height
        self.time_step = time_step
        self.fov_angle = fov_angle
        self.fov_distance = fov_distance


        self.fig = None



        if obstacle_map is None:
            package_dir = os.path.dirname(__file__)
            map_path = os.path.join(package_dir, 'obstacles.png')
            obstacle_map = _load_map_from_image(map_path)
            if not obstacle_map:
                obstacle_map = [[False for _ in range(int(world_width))] for _ in range(int(world_height))]

        # Convert the obstacle map to a numpy array for rendering
        self.obstacle_map_for_render = np.array(obstacle_map, dtype=np.uint8)

        self.city_env = CityEnv(
            obstacle_map=obstacle_map,
            world_width=world_width,
            world_height=world_height,
            time_step=time_step,
            fov_angle=fov_angle,
            fov_distance=fov_distance,
            drone=drone, # Changed from 'drones'
            target=target  # Changed from 'targets'
        )

        # x,y,theta,vx,vy
        self.observation_space = spaces.Dict({
            "drone": spaces.Box(
                low=np.array([-500, -500, 0, -15, -15], dtype=np.float64), 
                high=np.array([500, 500, 2*np.pi, 15, 15], dtype=np.float64), 
                shape=(5,), 
                dtype=np.float64
            ),
            "target": spaces.Box(
                low=np.array([-500, -500, 0,], dtype=np.float64), 
                high=np.array([500, 500, 2*np.pi], dtype=np.float64), 
                shape=(3,), 
                dtype=np.float64
            ),
            "time_elapsed": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64)
        })

        # Action space for one drone: [target_vx, target_vy, target_yaw_rate]
        self.action_space = spaces.Box(
            low=np.array([-15.0, -15.0, -np.pi]), 
            high=np.array([15.0, 15.0, np.pi]), 
            shape=(3,), 
            dtype=np.float32
        )



    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # This method will need to be implemented next
        # For now, it will raise an error as intended by gym.Env

        if not isinstance(action, np.ndarray) or action.shape != (3,):
            raise ValueError("Action must be a numpy array of shape (3,) representing [target_vx, target_vy, target_yaw_rate].")
        
        state = self.city_env.step(action)

        drone_pos = state.drone.position
        drone_vel = state.drone.velocity
        drone_state = np.array([
            drone_pos.x(), 
            drone_pos.y(),
            drone_pos.yaw, 
            drone_vel[0], 
            drone_vel[1], 
        ], dtype=np.float64)

        obs = {"drone": drone_state,
               "target": np.array([
                   state.target.position.x(), 
                   state.target.position.y(), 
                   state.target.position.yaw,
               ], dtype=np.float64),
               "time_elapsed": np.array([state.time_elapsed], dtype=np.float64)
           }
        
        reward = state.reward  # Assuming the State object has a reward attribute
        
        if state.time_elapsed >= self.max_time:
            done = True
        else:
            done = False
        truncated = False  # Assuming no truncation logic is implemented yet

        info = {}
        return obs, reward, done, truncated, info



    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict[str, Any]]:
        """
        Resets the environment to its initial state and returns the new state.
        """
        super().reset(seed=seed)
        
        # C++ reset() returns a State object
        state = self.city_env.reset()
        
        # CORRECTED: Access the single 'state.drone' and 'state.target' directly
        obs = {
            "drone": np.array([
                state.drone.position.x(), 
                state.drone.position.y(), 
                state.drone.position.yaw,
                state.drone.velocity[0], 
                state.drone.velocity[1], 
            ], dtype=np.float64),
            "target": np.array([
                state.target.position.x(), 
                state.target.position.y(), 
                state.target.position.yaw,
            ], dtype=np.float64),
            "time_elapsed": np.array([state.time_elapsed], dtype=np.float64)
        }
        return obs, {}

    def render(self, window=100) -> None:
            """Renders the current state of the environment using matplotlib."""
            if self.render_mode != "human":
                return

            # ... (all your existing rendering logic remains the same) ...
            # Get the current state from the C++ backend
            state = self.city_env.get_state()
            drone_pos = state.drone.position
            target_pos = state.target.position

            # Note: Your C++ worldToMap function returns grid coordinates.
            # Ensure these are the coordinates you intend to plot.
            drone_grid_pos = self.city_env.world_to_map(drone_pos.vector)
            target_grid_pos = self.city_env.world_to_map(target_pos.vector)

            # Initialize the plot on the first render call
            if self.fig is None:
                plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
                self.ax.imshow(self.obstacle_map_for_render, cmap='gray_r', origin='lower', extent=[0, self.world_width, 0, self.world_height])
                self.drone_plot = self.ax.scatter([], [], s=100, marker='>', c='blue', label='Drone')
                self.target_plot = self.ax.scatter([], [], s=100, marker='x', c='red', label='Target')
                self.ax.legend()
                self.ax.set_xlim(0, self.world_width)
                self.ax.set_ylim(0, self.world_height)

            # --- UPDATES THAT RUN EVERY FRAME ---
            self.drone_plot.set_offsets([drone_grid_pos[0], drone_grid_pos[1]])
            self.target_plot.set_offsets([target_grid_pos[0], target_grid_pos[1]])

            self.ax.set_title(f"City Environment | Sim Time: {state.time_elapsed:.2f}s | Drone Pos : ({drone_pos.x():.2f}, {drone_pos.y():.2f}) | Target Pos: ({target_pos.x():.2f}, {target_pos.y():.2f})")
            plt.pause(1e-9) # A very small, non-zero pause

    def close(self):
        """Close the rendering window."""
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)