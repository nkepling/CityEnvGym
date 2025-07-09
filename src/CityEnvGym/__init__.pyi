# CityEnvGym/__init__.pyi
# Type stubs for the CityEnvGym Python package.

from typing import List, Any, Optional, Union
from numpy import ndarray # Assuming Eigen::VectorXd maps to numpy.ndarray

# Version string
__version__: str

# ----------------------------------------------------
# 1. Type Stubs for Data Structures (from binder.cpp)
# ----------------------------------------------------

class Position:
    """
    Python binding for city_env::Position.
    Represents a 3D position vector with a yaw orientation.
    """
    vector: ndarray # Assuming Eigen::Vector3d maps to numpy.ndarray
    yaw: float

    def __init__(self) -> None: ...
    def x(self) -> float: ...
    def y(self) -> float: ...
    def z(self) -> float: ...

class DronePhysics:
    """
    Python binding for city_env::Drone::Physics.
    """
    mass: float
    propulsion_gain: float
    drag_coeff: float

    def __init__(self) -> None: ...

class Drone:
    """
    Python binding for city_env::Drone.
    """
    id: int
    position: Position
    velocity: ndarray # Assuming Eigen::Vector3d maps to numpy.ndarray
    physics: DronePhysics

    def __init__(self) -> None: ...

class Target:
    """
    Python binding for city_env::Target.
    """
    position: Position
    radius: float

    def __init__(self) -> None: ...

class State:
    """
    Python binding for city_env::State.
    """
    drones: List[Drone]
    targets: List[Target]
    time_elapsed: float

    def __init__(self) -> None: ...

# ----------------------------------------------------
# 2. Type Stubs for the Main CityEnv Class (from binder.cpp)
# ----------------------------------------------------

class CityEnv:
    """
    Python binding for city_env::CityEnv.
    The main constructor for the CityEnv class.
    """
    def __init__(
        self,
        num_drones: int,
        obstacle_map: List[List[bool]],
        world_width: float = 1000.0,
        world_height: float = 1000.0,
        time_step: float = 0.016666666666666666, # 1.0f / 60.0f
        fov_angle: float = 90.0,
        fov_distance: float = 100.0,
        drones: Optional[List[Drone]] = None, # Default is empty vector
    ) -> None: ...

    def reset(self) -> State:
        """
        Resets the environment to its initial state and returns the new state.
        """
        ...

    def step(self, actions: ndarray) -> State: # Assuming actions is an Eigen::VectorXd/MatrixXd
        """
        Advances the simulation by one time step with the given actions (target velocities)
        and returns the new state.
        """
        ...

    def get_state(self) -> State:
        """
        Returns the current state of the environment without advancing the simulation.
        """
        ...

    def check_fov(self, drone: Drone, target: Target) -> bool:
        """
        Checks if a target is within a drone's field of view, considering obstacles.
        """
        ...