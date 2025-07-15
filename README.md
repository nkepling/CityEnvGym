# Pursuit Evasion City Environment

Simple puruit evasion environmen with continous dynamics. Uses Eigen3 for simple physics. 



## Install

Clone this repo:

```bash
git clone --recurse-submodules https://github.com/nkepling/CityEnvGym.git
```

This project requires Eigen3, a C++ template library for linear algebra. Please ensure you have Eigen3 installed on your system before proceeding.

* **Eigen3**:

    * **macOS (using Homebrew)**:
        ```bash
        brew install eigen
        ```
    * **Ubuntu/Debian**:
        ```bash
        sudo apt-get update
        sudo apt-get install libeigen3-dev
        ```
    * **Windows**:
        You can typically download Eigen from its official website and manually set up your `EIGEN3_INCLUDE_DIR` environment variable, or use a package manager like `vcpkg` or `Conan`.

        *Using vcpkg:*
        ```bash
        vcpkg install eigen3
        ```


You can install this package with [UV](https://docs.astral.sh/uv/). Regular pip install should work too. 

```{bash}
uv pip install -e . -v
```


## Quickstart

```python
import gymnasium as gym
import CityEnvGym

obstacle_map = [[False for _ in range(int(200))] for _ in range(int(200))]

sensors = [[0.0,0.0,25.0],[-50.0,-50.0,25],[50.0,50.0,25],[-50.0,50.0,25],[50.0,-50.0,25]] # x,y ,radius

env = gym.make("CityEnvGym/CityEnv-v0", render_mode="human",obstacle_map=obstacle_map,sensors=sensors,num_evader_steps=1000,max_episode_steps=10000)

```