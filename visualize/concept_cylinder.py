import time
import numpy as np
import open3d as o3d
import os
import imageio

FPS = 60
KNEE_POINT = np.array([0.0, 0.0, 0.0])  # knee at origin
CAPTURE = "capture"  # capture path

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
def build_default_geometry():
    thigh_len = 0.40
    shin_len  = 0.40
    radius    = 0.05

    thigh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=thigh_len, resolution=40, split=4)
    thigh.compute_vertex_normals()
    thigh.paint_uniform_color([1.0, 0.0, 0.0])  # red

    shin = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=shin_len, resolution=40, split=4)
    shin.compute_vertex_normals()
    shin.paint_uniform_color([0.0, 0.2, 1.0])  # blue

    # Position thigh so its bottom face sits at the knee (z=0), extending upward to +Z
    # Cylinder center is at z = height/2 after translation.
    thigh.translate([0.0, 0.0, thigh_len / 2.0])

    # Position shin so its top face sits at the knee (z=0), extending downward to -Z
    # Move cylinder center to z = -height/2
    shin.translate([0.0, 0.0, -shin_len / 2.0])

    # We'll animate the shin by rotating it about the knee point (origin) around the X axis.
    # To avoid accumulating floating point drift, we:
    #   - keep a "base" copy of the shin
    #   - each frame, reset shin to base and apply the current transform
    shin_base = o3d.geometry.TriangleMesh(shin)  # deep copy-ish for mesh data

    # Optional: coordinate frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    return thigh, shin, shin_base, axis

def setup_visualizer(thigh, shin, axis):
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

    return vis

def generate_default_motion():
    
    angle_max_deg = 140.0
    ang_speed_deg_per_sec = 80.0  # how fast it swings
    cycles = 2

    # time for one full cycle (0->max->0)
    cycle_time = 2.0 * angle_max_deg / ang_speed_deg_per_sec
    frames_per_cycle = int(np.ceil(cycle_time * FPS))

    # build one cycle as a triangle wave: 0..1..0, then scale to degrees
    i = np.arange(frames_per_cycle)
    phase = i / (frames_per_cycle - 1)                  # 0..1
    tri = 1.0 - np.abs(2.0 * phase - 1.0)               # 0..1..0
    angles_one_cycle = tri * angle_max_deg              # 0..max..0

    # repeat for 3 cycles (finite vector)
    angles = np.tile(angles_one_cycle, cycles)

    return angles

def run_visualization(vis, angles, shin, shin_base, video: str | None = None):
    # TODO test video save

    writer = None

    if video is not None:
        os.makedirs(CAPTURE, exist_ok=True)
        path = os.path.join(CAPTURE, video)
        writer = imageio.get_writer(path, fps=FPS)

    for angle in angles:

        # Reset shin to base (avoid drift)
        shin.vertices = o3d.utility.Vector3dVector(np.asarray(shin_base.vertices))
        shin.triangles = o3d.utility.Vector3iVector(np.asarray(shin_base.triangles))
        shin.vertex_normals = o3d.utility.Vector3dVector(np.asarray(shin_base.vertex_normals))

        R = rot_x(angle)
        T = rotation_about_point(R, KNEE_POINT)
        shin.transform(T)

        # Update visualization
        vis.update_geometry(shin)
        vis.poll_events()
        vis.update_renderer()

        # Capture frame if recording
        if writer is not None:
            img = vis.capture_screen_float_buffer(False)
            img = (255 * np.asarray(img)).astype(np.uint8)
            writer.append_data(img)

        time.sleep(1.0 / FPS)

    if writer is not None:
        writer.close()

    vis.destroy_window()

def main():
    thigh, shin, shin_base, axis = build_default_geometry()
    vis = setup_visualizer(thigh=thigh, shin=shin, axis=axis)

    angles = generate_default_motion()
    run_visualization(vis=vis, angles=angles, shin=shin, shin_base=shin_base)

if __name__ == "__main__":
    main()