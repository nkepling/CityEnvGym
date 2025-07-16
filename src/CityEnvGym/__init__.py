

from gymnasium import register

register(
    id="CityEnvGym/CityEnv-v0",
    entry_point="CityEnvGym.CityEnvGym:CityEnvironment",
    max_episode_steps=10000,
    ) 



try:
    from ._CityEnvGym import *

except ImportError as e:

    if os.environ.get("READTHEDOCS") == "True":
        print("WARNING: Could not import CityEnvGym. This is expected on ReadTheDocs.")
    else:

        raise ImportError(
            f"Could not load C++ extension 'CityEnvGym'. "
            f"Ensure it is built and installed correctly. Original error: {e}"
        ) from e
    

from . import utils
from . import wrappers

__all__ = [
    "Position",
    "DronePhysics",
    "Drone",
    "Target",
    "State",
    "CityEnvironment",
    "Sensor",
]


