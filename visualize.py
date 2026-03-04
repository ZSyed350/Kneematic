import time
import numpy as np
import open3d as o3d

# -----------------------------
# Parameters (edit as you like)
# -----------------------------
THIGH_LEN = 0.40
SHIN_LEN  = 0.40
RADIUS    = 0.05

FPS = 60
ANGLE_MAX_DEG = 140.0
ANG_SPEED_DEG_PER_SEC = 80.0  # how fast it swings
KNEE_POINT = np.array([0.0, 0.0, 0.0])  # knee at origin

# -----------------------------
# Helpers
# -----------------------------
def rotation_about_point(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Build a 4x4 transform that applies rotation R about a fixed point p.
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p - R @ p
    return T

def rot_x(deg: float) -> np.ndarray:
    """
    Rotation matrix about X axis by deg degrees.
    (Bends the shin in the Y-Z plane.)
    """
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1, 0, 0],
        [0, c,-s],
        [0, s, c]
    ], dtype=float)

# -----------------------------
# Build geometry
# Open3D cylinders are along +Z by default, centered at origin.
# We'll position them so they meet at the knee (z = 0).
# -----------------------------
thigh = o3d.geometry.TriangleMesh.create_cylinder(radius=RADIUS, height=THIGH_LEN, resolution=40, split=4)
thigh.compute_vertex_normals()
thigh.paint_uniform_color([1.0, 0.0, 0.0])  # red

shin = o3d.geometry.TriangleMesh.create_cylinder(radius=RADIUS, height=SHIN_LEN, resolution=40, split=4)
shin.compute_vertex_normals()
shin.paint_uniform_color([0.0, 0.2, 1.0])  # blue

# Position thigh so its bottom face sits at the knee (z=0), extending upward to +Z
# Cylinder center is at z = height/2 after translation.
thigh.translate([0.0, 0.0, THIGH_LEN / 2.0])

# Position shin so its top face sits at the knee (z=0), extending downward to -Z
# Move cylinder center to z = -height/2
shin.translate([0.0, 0.0, -SHIN_LEN / 2.0])

# We'll animate the shin by rotating it about the knee point (origin) around the X axis.
# To avoid accumulating floating point drift, we:
#   - keep a "base" copy of the shin
#   - each frame, reset shin to base and apply the current transform
shin_base = o3d.geometry.TriangleMesh(shin)  # deep copy-ish for mesh data

# Optional: coordinate frame
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

# -----------------------------
# Visualizer setup
# -----------------------------
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Knee Angle Animation (0 ↔ 140 deg)", width=1100, height=800)

vis.add_geometry(thigh)
vis.add_geometry(shin)
vis.add_geometry(axis)

# Improve view a bit
opt = vis.get_render_option()
opt.background_color = np.array([0.05, 0.05, 0.06])
opt.mesh_show_back_face = True

ctr = vis.get_view_control()
ctr.set_front([0.3, -1.0, 0.4])
ctr.set_lookat([0.0, 0.0, 0.0])
ctr.set_up([0.0, 0.0, 1.0])
ctr.set_zoom(0.8)

# -----------------------------
# Animation loop
# 0 -> 140 -> 0 repeating (triangle wave)
# -----------------------------
t0 = time.time()

try:
    while True:
        t = time.time() - t0

        # Triangle wave between 0 and 1:
        # phase increases linearly; reflect to make up-and-down
        period = 2.0 * ANGLE_MAX_DEG / ANG_SPEED_DEG_PER_SEC  # time for 0->max->0
        phase = (t % period) / period  # 0..1
        tri = 1.0 - abs(2.0 * phase - 1.0)  # 0..1..0

        angle = tri * ANGLE_MAX_DEG  # 0..max..0

        # Reset shin to base, then apply rotation about the knee point
        shin.vertices = o3d.utility.Vector3dVector(np.asarray(shin_base.vertices))
        shin.triangles = o3d.utility.Vector3iVector(np.asarray(shin_base.triangles))
        shin.vertex_normals = o3d.utility.Vector3dVector(np.asarray(shin_base.vertex_normals))

        R = rot_x(angle)
        T = rotation_about_point(R, KNEE_POINT)
        shin.transform(T)

        # Update display
        vis.update_geometry(shin)
        vis.poll_events()
        vis.update_renderer()

        time.sleep(1.0 / FPS)

except KeyboardInterrupt:
    pass
finally:
    vis.destroy_window()