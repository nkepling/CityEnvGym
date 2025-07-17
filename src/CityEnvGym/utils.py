from PIL import Image
import numpy as np


def _load_map_from_image(filepath: str) -> list[list[bool]]:
    """Loads a grayscale image and converts it into a boolean obstacle map."""
    try:
        with Image.open(filepath) as img:
            # Convert to grayscale and then to a NumPy array
            gray_img = img.convert('L')
            pixels = np.array(gray_img, dtype=np.uint8)
            
            # Black pixels (value < 128) are obstacles (True)
            # White pixels are free space (False)
            obstacles = pixels > 128
            return obstacles.tolist()
    except FileNotFoundError:
        print(f"Warning: Default map file not found at {filepath}. Creating an empty map.")
        return []


def _update_physics(drone, properties):
    """Updates the drone's physics based on the action taken."""
    for prop, value in properties.items():
        if hasattr(drone.physics, prop):
            setattr(drone.physics, prop, value)
        else:
            print(f"Warning: Drone does not have property '{prop}' to update.")


def coordinate_transformation(evader_state, pursuer_state):
    """World coords to pursuer coords transformation.
    """
    evader_pos = evader_state[:2]  
    evader_heading_world = evader_state[2] 

    pursuer_pos = pursuer_state[:2]  
    pursuer_heading_world = pursuer_state[2]  

    relative_evader_pos_world = evader_pos - pursuer_pos

    cos_neg_theta = np.cos(-pursuer_heading_world)
    sin_neg_theta = np.sin(-pursuer_heading_world)

    rotation_matrix = np.array([
        [cos_neg_theta, -sin_neg_theta],
        [sin_neg_theta, cos_neg_theta]
    ])

    evader_pos_in_pursuer_frame = rotation_matrix @ relative_evader_pos_world
    evader_heading_in_pursuer_frame = evader_heading_world - pursuer_heading_world
    evader_heading_in_pursuer_frame = np.arctan2(np.sin(evader_heading_in_pursuer_frame), np.cos(evader_heading_in_pursuer_frame))
    return np.concatenate([evader_pos_in_pursuer_frame, [evader_heading_in_pursuer_frame]])

def rotate_vector(vector, angle):
    """Helper function to rotate a 2D vector."""
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return rotation_matrix @ vector

def transform_trajectory_to_ego(trajectory: np.ndarray, pursuer_state: np.ndarray) -> np.ndarray:
    """
    Transforms a world-frame trajectory of (x, y) points into the pursuer's egocentric frame.

    Args:
        trajectory (np.ndarray): An array of future positions, shape (N, 2).
        pursuer_state (np.ndarray): The state of the pursuer [x, y, yaw, ...].

    Returns:
        np.ndarray: The transformed trajectory in the pursuer's frame, shape (N, 2).
    """
    pursuer_pos = pursuer_state[:2]
    pursuer_yaw = pursuer_state[2]

    relative_trajectory = trajectory - pursuer_pos
    angle = -pursuer_yaw
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    ego_trajectory = (rotation_matrix @ relative_trajectory.T).T
    
    return ego_trajectory