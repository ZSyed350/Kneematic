import open3d as o3d
import numpy as np
import os

import compute
import camera
import animate

# PATHS
STL_FILE = "models/Leg_Reduced_mesh.STL"
RAW_PCD = "models/leg_pcd.ply"
PROCESSED_PCD = "models/leg_processed.ply"
AXIS = "models/leg_axis.ply"

# PCD
NUM_POINTS = 5000
KNEE_CENTER = [464.18004978, 116.53999796, 640.04481245]

# CAMERA
ZOOM = 0.7
WIDTH = 1000
HEIGHT = 800

# POINT REMOVAL PARAMETERS
KNEE_RADIUS = 100.0
KNEE_DEPTH = 70.0
HIP_RADIUS = 500.0
HIP_DEPTH = 360.0
FOOT_RADIUS = 600.0
FOOT_DEPTH = 150.0

# ANIMATION
FPS = 60
MAX_BEND_DEG = 90.0
CYCLES = 1
ANG_SPEED_DEG_PER_SEC = 60.0
USE_LIVE_ANGLE = False


def apply_knee_angle(
    angle_deg: float,
    shin_base_points: np.ndarray,
    shin_pcd,
    rot_axis: np.ndarray,
    knee_center: np.ndarray,
):
    # Reset shin points from base
    current_shin = shin_base_points.copy()

    # Use negative sign because that is the bend direction you wanted
    theta = -np.deg2rad(angle_deg)
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * theta)
    T = animate.rotation_about_point(R, knee_center)

    # Rotate shin point cloud
    current_shin_h = np.c_[current_shin, np.ones(len(current_shin))]
    current_shin = (T @ current_shin_h.T).T[:, :3]
    shin_pcd.points = o3d.utility.Vector3dVector(current_shin)


if __name__ == "__main__":

    if not os.path.exists(RAW_PCD):
        mesh = o3d.io.read_triangle_mesh(STL_FILE)
        pcd = mesh.sample_points_uniformly(number_of_points=NUM_POINTS)
        o3d.io.write_point_cloud(RAW_PCD, pcd, format='auto', write_ascii=False, compressed=False, print_progress=False)
    else:
        pcd = o3d.io.read_point_cloud(RAW_PCD)

    pcd.paint_uniform_color(color=(0.0, 1.0, 0.0))
    points = np.asarray(pcd.points)

    _, leg_axis, start_point, end_point = compute.get_long_axis(points)
    axis_line = compute.create_line_set(start_point, end_point, color=(0.0, 0.0, 0.0))

    pcd, _, _, _ = compute.remove_points_in_cylinder_volume(
        axis_line, KNEE_CENTER, KNEE_DEPTH, KNEE_RADIUS, pcd
    )

    pcd, _, _, _ = compute.remove_points_in_cylinder_volume(
        axis_line, end_point, HIP_DEPTH, HIP_RADIUS, pcd
    )

    # Save stuff
    if not os.path.exists(PROCESSED_PCD):
        o3d.io.write_point_cloud(PROCESSED_PCD, pcd)
    if not os.path.exists(AXIS):
        o3d.io.write_line_set(AXIS, axis_line)

    # Get foot
    footless_pcd, foot_mask, _, _ = compute.remove_points_in_cylinder_volume(
        axis_line, start_point, FOOT_DEPTH, FOOT_RADIUS, pcd
    )
    foot_pcd = o3d.geometry.PointCloud()
    points = np.asarray(pcd.points)
    foot_pcd.points = o3d.utility.Vector3dVector(points[foot_mask])
    foot_pcd.paint_uniform_color(color=(0.0, 1.0, 1.0))
    foot_points = np.asarray(foot_pcd.points)

    # Get foot axis
    _, foot_axis, foot_start, foot_end = compute.get_long_axis(foot_points)
    foot_line = compute.create_line_set(foot_start, foot_end, color=(0.0, 0.0, 0.0))

    # Get plane of rotation
    plane_point, plane_normal, _ = compute.plane_from_axes(
        leg_axis,
        foot_axis,
        point_on_plane=KNEE_CENTER
    )
    rotation_plane_mesh = compute.create_plane_mesh_from_point_normal(
        plane_point=KNEE_CENTER,
        plane_normal=plane_normal,
        width=700.0,
        height=700.0,
        color=(0.5, 0.5, 0.5),
    )

    knee_center = np.asarray(KNEE_CENTER, dtype=float)
    points = np.asarray(pcd.points)

    # Split remaining leg cloud into thigh and shin
    thigh_mask, shin_mask = animate.split_thigh_shin(
        points=points,
        knee_center=knee_center,
        leg_axis=leg_axis,
        foot_ref=foot_start   # or foot_points.mean(axis=0)
    )

    thigh_points = points[thigh_mask]
    shin_points = points[shin_mask]

    thigh_pcd = animate.make_pcd(thigh_points, (1.0, 0.0, 0.0))
    shin_pcd  = animate.make_pcd(shin_points,  (0.0, 0.2, 1.0))
    thigh_pcd = animate.color_points_by_distance_from_center(thigh_pcd, KNEE_CENTER)
    shin_pcd  = animate.color_points_by_distance_from_center(shin_pcd, KNEE_CENTER)

    # Base copy so rotation never accumulates drift
    shin_base_points = shin_points.copy()

    # Shin long axis line from the shin points, not the foot-only points
    _, shin_axis, shin_start, shin_end = compute.get_long_axis(shin_points)
    shin_line = animate.make_line(shin_start, shin_end, color=(0.0, 0.0, 0.0))
    shin_line_base = np.vstack([shin_start, shin_end]).copy()

    # Thigh line is optional but helpful
    _, thigh_axis, thigh_start, thigh_end = compute.get_long_axis(thigh_points)
    thigh_line = animate.make_line(thigh_start, thigh_end, color=(0.0, 0.0, 0.0))

    # Rotation axis = plane normal
    rot_axis = np.asarray(plane_normal, dtype=float)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    # VISUALIZE ------------------------------------------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="JBOSINO", width=WIDTH, height=HEIGHT)

    vc = vis.get_view_control()
    front, up = camera.compute_camera_vectors(leg_axis)

    vc.set_lookat(KNEE_CENTER)
    vc.set_up(up)
    vc.set_front(front)
    vc.set_zoom(ZOOM)

    geometries = [
        thigh_pcd,
        shin_pcd,
    ]
    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.0, 0.0, 0.0])
    opt.point_size = 3.0
    opt.mesh_show_back_face = True

    angles = animate.generate_motion(
        angle_max_deg=MAX_BEND_DEG,
        ang_speed_deg_per_sec=ANG_SPEED_DEG_PER_SEC,
        cycles=CYCLES,
        fps=FPS,
    )

    if USE_LIVE_ANGLE:
        angle_source = animate.LiveAngleSource(initial_angle_deg=0.0)
        # Arduino listener here
    else:
        angle_source = animate.GeneratedAngleSource(angles)


    def animate_callback(vis):
        angle_deg = angle_source.get_angle_deg()

        apply_knee_angle(
            angle_deg=angle_deg,
            shin_base_points=shin_base_points,
            shin_pcd=shin_pcd,
            rot_axis=rot_axis,
            knee_center=knee_center,
        )

        vis.update_geometry(shin_pcd)
        return False
    
    vis.register_animation_callback(animate_callback)
    vis.run()
    vis.destroy_window()

    # while True:
    #     for angle_deg in angles:
    #         # Reset shin points from base
    #         current_shin = shin_base_points.copy()

    #         # Axis-angle rotation about plane normal
    #         theta = -np.deg2rad(angle_deg)
    #         R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * theta)
    #         T = animate.rotation_about_point(R, knee_center)

    #         # Rotate shin point cloud
    #         current_shin_h = np.c_[current_shin, np.ones(len(current_shin))]
    #         current_shin = (T @ current_shin_h.T).T[:, :3]
    #         shin_pcd.points = o3d.utility.Vector3dVector(current_shin)

    #         # Rotate shin axis line too
    #         shin_line_pts = shin_line_base.copy()
    #         shin_line_h = np.c_[shin_line_pts, np.ones(len(shin_line_pts))]
    #         shin_line_pts = (T @ shin_line_h.T).T[:, :3]
    #         shin_line.points = o3d.utility.Vector3dVector(shin_line_pts)

    #         vis.update_geometry(shin_pcd)
    #         vis.update_geometry(shin_line)
    #         vis.poll_events()
    #         vis.update_renderer()
    #         time.sleep(1.0 / FPS)

    #     vis.run()
    