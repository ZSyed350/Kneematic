import numpy as np

def compute_camera_vectors(long_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute camera front and up vectors so the long axis appears vertical on screen.
    
    Returns:
        front: viewing direction
        up:    screen-up direction
    """
    up = np.asarray(long_axis, dtype=float)
    up = up / np.linalg.norm(up)

    # Pick a helper vector not parallel to up
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, up)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])

    # Make front perpendicular to up
    front = np.cross(up, helper)
    front = front / np.linalg.norm(front)

    return front, up