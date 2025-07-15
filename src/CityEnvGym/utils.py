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


