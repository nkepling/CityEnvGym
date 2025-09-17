import pytest
import numpy as np
from CityEnvGym.utils import astar_path 




import numpy as np

# --- Helper Function for Testing ---
# It's good practice to have your test helpers return specific failure info.
def find_path_collision(path, obstacle_map):
    """
    Checks a path for obstacles and returns the first collision found.

    Args:
        path (list or np.ndarray): A list of (x, y) coordinates.
        obstacle_map (list[list[bool]]): 2D grid where True is an obstacle.

    Returns:
        tuple or None: The (x, y) coordinate of the first obstacle found, 
                       or None if the path is clear.
    """
    for x, y in path:
        # Check for out-of-bounds access
        if not (0 <= y < len(obstacle_map) and 0 <= x < len(obstacle_map[0])):
            return (x, y) # Return the out-of-bounds point as a collision
            
        # Check for an obstacle
        if obstacle_map[y][x]:
            return (x, y) # Return the colliding point
            
    return None # No obstacles found


# --- Your Pytest Function ---
def test_astar_path():
    """
    Tests the astar_path function for correctness, including an obstacle check.
    """
    # 1. Define the test case environment
    obstacle_map = [
        # x:  0      1      2      3      4
        [False, False, False, False, False], # y = 0
        [False, True,  True,  True,  False], # y = 1
        [False, False, False, True,  False], # y = 2
        [False, True,  False, False, False], # y = 3
        [False, False, False, True,  False]  # y = 4
    ]
    
    start = (0, 0) # (x, y)
    goal = (4, 4)  # (x, y)

    # 2. Run the function being tested
    path = astar_path(start, goal, obstacle_map)

    assert path is not None and len(path) > 0, "A* failed to return a path."
    
    assert np.array_equal(path[0], start), f"Path starts at {path[0]} instead of {start}."
    assert np.array_equal(path[-1], goal), f"Path ends at {path[-1]} instead of {goal}."
    

    colliding_point = find_path_collision(path, obstacle_map)
    assert colliding_point is None, f"Path collides with an obstacle at {colliding_point}!"


