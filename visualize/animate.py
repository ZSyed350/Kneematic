import open3d as o3d
import numpy as np
import matplotlib.cm as cm


class AngleSource:
    def get_angle_deg(self) -> float:
        raise NotImplementedError

class GeneratedAngleSource(AngleSource):
    def __init__(self, angles):
        self.angles = np.asarray(angles, dtype=float)
        self.i = 0

    def get_angle_deg(self) -> float:
        angle = self.angles[self.i % len(self.angles)]
        self.i += 1
        return float(angle)

class LiveAngleSource(AngleSource):
    def __init__(self, initial_angle_deg: float = 0.0):
        self.current_angle_deg = float(initial_angle_deg)

    def update_angle(self, angle_deg: float):
        self.current_angle_deg = float(angle_deg)

    def get_angle_deg(self) -> float:
        return self.current_angle_deg

def rotation_about_point(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p - R @ p
    return T

def generate_motion(angle_max_deg=90.0, ang_speed_deg_per_sec=60.0, cycles=1, fps=60):
    cycle_time = 2.0 * angle_max_deg / ang_speed_deg_per_sec
    frames_per_cycle = int(np.ceil(cycle_time * fps))

    i = np.arange(frames_per_cycle)
    phase = i / max(frames_per_cycle - 1, 1)
    tri = 1.0 - np.abs(2.0 * phase - 1.0)   # 0 -> 1 -> 0
    angles = np.tile(tri * angle_max_deg, cycles)
    return angles

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

