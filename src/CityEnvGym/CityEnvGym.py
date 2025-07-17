import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ._CityEnvGym import CityEnv, Drone, Target, State,Sensor
from typing import Any, SupportsFloat
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .utils import _load_map_from_image, _update_physics


class CityEnvironment(gym.Env):
    """
    CityEnv is a Gymnasium environment for simulating a single drone in a city.
    It provides methods to reset the environment, step through time, and render the state.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60} 

    # The __init__ signature is simplified as the environment handles one drone/target internally
    def __init__(self,**kwargs) -> None:
        super().__init__()

        self.max_time = kwargs.get('max_time', 300.0)
        self.num_evader_steps = kwargs.get('num_evader_steps', 100)
        self.render_mode = kwargs.get('render_mode', 'human')
        self.time_step = kwargs.get('time_step', 1/60.0)
        self.fov_angle = kwargs.get('fov_angle', 90.0)
        self.fov_distance = kwargs.get('fov_distance', 100.0)
        obstacle_map = kwargs.get('obstacle_map', None)
        self.sensors = kwargs.get('sensors', [])        
        self.target_physics = kwargs.get('target_physics', None)
        self.drone_physics = kwargs.get('drone_physics', None)

        # init drone and target

        drone = Drone()
        target = Target()
        
        if self.drone_physics is not None:
            _update_physics(drone, self.drone_physics)
        
        if self.target_physics is not None:
            _update_physics(target, self.target_physics)

        target.num_steps = self.num_evader_steps  # Set the number of steps for the target's path

        self.fig = None

        if obstacle_map is None:
            package_dir = os.path.dirname(__file__)
            map_path = os.path.join(package_dir, 'obstacles.png')
            obstacle_map = _load_map_from_image(map_path)
            self.world_width = world_width = 1000.0  # Default world width
            self.world_height = world_height = 1000.0  # Default world height
            if not obstacle_map:
                obstacle_map = [[False for _ in range(int(self.world_width))] for _ in range(int(self.world_height))]


        if self.sensors:
            self.sensors = [Sensor(*sensor) for sensor in self.sensors]

        # Convert the obstacle map to a numpy array for rendering
        self.obstacle_map_for_render = np.array(obstacle_map, dtype=np.uint8)
        self.world_width = world_width = len(obstacle_map[0])
        self.world_height = world_height = len(obstacle_map)

        self.city_env = CityEnv(
            obstacle_map=obstacle_map,
            world_width=world_width,
            world_height=world_height,
            time_step=self.time_step,
            fov_angle=self.fov_angle,
            fov_distance=self.fov_distance,
            drone=drone, # Changed from 'drones'
            target=target,  # Changed from 'targets'
            sensors=self.sensors,
            origin = (-world_width / 2, -world_height / 2),  # Center the origin
        )

        #BUG THE observation space bounds are hardcoded to the world size
        # x,y,theta,vx,vy

        # 1. Define the base bounds for a single step [x, y]
        low_bounds_single_step = np.array([-self.world_width / 2, -self.world_height / 2], dtype=np.float32)
        high_bounds_single_step = np.array([self.world_width / 2, self.world_height / 2], dtype=np.float32)

# 2. Use np.tile to repeat these bounds for each step, creating the (N, 2) shape
        low_bounds_full = np.tile(low_bounds_single_step, (self.num_evader_steps, 1))
        high_bounds_full = np.tile(high_bounds_single_step, (self.num_evader_steps, 1))

        self.observation_space = spaces.Dict({
            "drone": spaces.Box(
                low=np.array([-self.world_width/2, -self.world_height/2, -np.pi, -15, -15], dtype=np.float32), 
                high=np.array([self.world_width/2, self.world_height/2, np.pi, 15, 15], dtype=np.float32), 
                shape=(5,), 
                dtype=np.float32
            ),
           "target": spaces.Box(
            # The first two elements are relative position, the third is relative yaw
            low=np.array([-self.world_width, -self.world_height, -np.pi], dtype=np.float32),
            high=np.array([self.world_width, self.world_height, np.pi], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        ),
          "future_evader_positions": spaces.Box(
                low=low_bounds_full,
                high=high_bounds_full,

                shape=(self.num_evader_steps, 2),
                
                dtype=np.float32
            )
        })

        # Action space for one drone: [target_vx, target_vy, target_yaw_rate]
     # Define the true, physical bounds of your actions
        self.true_action_low = np.array([-15.0, -15.0, -np.pi], dtype=np.float32)
        self.true_action_high = np.array([15.0, 15.0, np.pi], dtype=np.float32)

        # The action space exposed to the SB3 agent is normalized
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(3,), 
            dtype=np.float32
        )


    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # This method will need to be implemented next
        # For now, it will raise an error as intended by gym.Env

        if not isinstance(action, np.ndarray) or action.shape != (3,):
            raise ValueError("Action must be a numpy array of shape (3,) representing [target_vx, target_vy, target_yaw_rate].")
        

        true_action = self.true_action_low + (action + 1.0) * 0.5 * (self.true_action_high - self.true_action_low)
        state = self.city_env.step(true_action)

        drone_pos = state.drone.position
        drone_vel = state.drone.velocity
        drone_state = np.array([
            drone_pos.x(), 
            drone_pos.y(),
            drone_pos.yaw, 
            drone_vel[0], 
            drone_vel[1], 
        ], dtype=np.float32)


        future_evader_positions = np.array(state.future_target_positions).reshape(2, -1).T  
        assert future_evader_positions.shape == (self.num_evader_steps, 2), f"Future evader positions should have shape (2, num_evader_steps), got {future_evader_positions.shape}"

        obs = {"drone": drone_state,
               "target": np.array([
                   state.target.position.x(), 
                   state.target.position.y(), 
                   state.target.position.yaw,
               ], dtype=np.float32),
               "future_evader_positions": future_evader_positions,
           }
        
        reward = state.reward  # Assuming the State object has a reward attribute
        
        if state.time_elapsed >= self.max_time:
            done = True
        else:
            done = False
        truncated = False  # Assuming no truncation logic is implemented yet

        info = {"time_elapsed": state.time_elapsed}



      
        # --- End Debugging Block ---
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
            ], dtype=np.float32),
            "target": np.array([
                state.target.position.x(), 
                state.target.position.y(), 
                state.target.position.yaw,
            ], dtype=np.float32),
            "future_evader_positions": np.array(state.future_target_positions).reshape(2, -1).T  # Reshape to (2, num_steps) for x, y
        }
        return obs, {"time_elapsed": state.time_elapsed}

    def render(self,window=100) -> None:
            """Renders the current state of the environment using matplotlib."""

            # TODO: Implement the rendering logic with pygame
            if self.render_mode != "human":
                return

            state = self.city_env.get_state()
            drone_pos = state.drone.position
            target_pos = state.target.position

            target_future_positions = state.future_target_positions

            # drone_grid_pos = self.city_env.world_to_map(drone_pos.vector)
            # target_grid_pos = self.city_env.world_to_map(target_pos.vector)


            if self.fig is None:
                plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
                self.ax.imshow(self.obstacle_map_for_render, cmap='gray_r', origin='lower', extent=[-self.world_width/2, self.world_width/2, -self.world_height/2, self.world_height/2])
                self.drone_plot = self.ax.scatter([], [], s=100, marker='>', c='blue', label='Drone')
                self.target_plot = self.ax.scatter([], [], s=100, marker='x', c='red', label='Target')
                self.drone_vel_plot = self.ax.quiver([], [], [], [], color='blue', alpha=0.8, scale=100, width=0.005)
                self.target_vel_plot = self.ax.quiver([], [], [], [], color='red', alpha=0.8, scale=100, width=0.005)
            
                self.future_target_plot, = self.ax.plot([], [], 'r--', label='Future Target Path', alpha=0.6) # Note the comma
                self.ax.legend()
                self.ax.set_xlim(-self.world_width/2, self.world_width/2)
                self.ax.set_ylim(-self.world_height/2, self.world_height/2)
                for sensor in self.sensors:
                    # sensor_grid_pos = self.city_env.world_to_map(sensor.position)
                    circle = Circle((sensor.position[0], sensor.position[1]), sensor.radius, color='green', fill=True, alpha=0.5)
                    self.ax.add_artist(circle)


            self.drone_plot.set_offsets([drone_pos.x(), drone_pos.y()])
            self.target_plot.set_offsets([target_pos.x(), target_pos.y()])


            if target_future_positions:
        # Prepend the target's current position to the start of the path line
                full_path = np.vstack([target_pos.vector, np.array(target_future_positions)])
                path_x, path_y = full_path.T
                self.future_target_plot.set_data(path_x, path_y)
            else:
                self.future_target_plot.set_data([], [])

             # Change 4: Update the velocity quiver plots

            drone_vel = state.drone.velocity

            self.drone_vel_plot.set_offsets([drone_pos.x(), drone_pos.y()])
            self.drone_vel_plot.set_UVC(drone_vel[0], drone_vel[1])

            target_vel = state.target.velocity
            self.target_vel_plot.set_offsets([target_pos.x(), target_pos.y()])
            self.target_vel_plot.set_UVC(target_vel[0], target_vel[1])


            self.ax.set_title(f"City Environment | Sim Time: {state.time_elapsed:.2f}s | Drone Pos : ({drone_pos.x():.2f}, {drone_pos.y():.2f}) | Target Pos: ({target_pos.x():.2f}, {target_pos.y():.2f})")


            plt.pause(1e-9) # A very small, non-zero pause

    def close(self):
        """Close the rendering window."""
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)

    def seed(self, seed: int | None = None) -> list[int]:
        pass
