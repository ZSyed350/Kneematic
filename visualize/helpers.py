import numpy as np
import open3d as o3d
import matplotlib.cm as cm

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

def process_line(line: str):
    """Chip, S0, S1, S2, S3, S4, S5, pos"""
    data = line.split(',')
    chip = int(data[0])
    s0 = float(data[1])
    s1 = float(data[2])
    s2 = float(data[3])
    s3 = float(data[4])
    s4 = float(data[5])
    s5 = float(data[6])
    pos = float(data[7])
    return chip, s0, s1, s2, s3, s4, s5, pos

def rotation_about_point(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p - R @ p
    return T

def split_thigh_shin(points: np.ndarray, knee_center: np.ndarray, leg_axis: np.ndarray, foot_ref: np.ndarray):
    """
    Split by signed projection along the long axis.
    Then pick the side whose centroid is closer to foot_ref as shin.
    """
    axis = leg_axis / np.linalg.norm(leg_axis)
    rel = points - knee_center
    signed = rel @ axis

    mask_a = signed >= 0
    mask_b = ~mask_a

    pts_a = points[mask_a]
    pts_b = points[mask_b]

    if len(pts_a) == 0 or len(pts_b) == 0:
        raise ValueError("Thigh/shin split failed.")

    cen_a = pts_a.mean(axis=0)
    cen_b = pts_b.mean(axis=0)

    if np.linalg.norm(cen_a - foot_ref) < np.linalg.norm(cen_b - foot_ref):
        shin_mask = mask_a
        thigh_mask = mask_b
    else:
        shin_mask = mask_b
        thigh_mask = mask_a

    return thigh_mask, shin_mask

def make_pcd(points: np.ndarray, color):
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(points)
    out.paint_uniform_color(color)
    return out

def make_line(start: np.ndarray, end: np.ndarray, color=(0, 0, 0)):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack([start, end]))
    ls.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    ls.colors = o3d.utility.Vector3dVector([color])
    return ls

def color_points_by_distance_from_center(pcd, center):
    points = np.asarray(pcd.points)
    center = np.asarray(center, dtype=float)

    dist = np.linalg.norm(points - center, axis=1)

    d_min, d_max = dist.min(), dist.max()
    if d_max - d_min < 1e-8:
        dist_norm = np.zeros_like(dist)
    else:
        dist_norm = (dist - d_min) / (d_max - d_min)

    colors = cm.get_cmap("hsv")(dist_norm)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd